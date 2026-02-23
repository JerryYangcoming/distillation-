# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import model_dict  # 请确保此导入在您的环境中有效

# --- 基本配置 ---
BASE_SAVE_DIR = 'save'  # Base directory for all saved files
GATING_MODEL_DIR = os.path.join(BASE_SAVE_DIR, 'gating_models')
os.makedirs(GATING_MODEL_DIR, exist_ok=True)

# --- 设备检查 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")
if device == 'cuda':
    if torch.cuda.is_available():
        print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
    else:
        print("CUDA 声称可用，但 torch.cuda.is_available() 返回 False。切换到 CPU。")
        device = 'cpu'

# --- 模型路径与参数 ---
classifier_path = os.path.join(BASE_SAVE_DIR,
                               'classifier_models/classifier_resnet8x4_cifar100_lr_0.05_decay_0.0005_trial_1/resnet8x4_best.pth')
student_base_path = os.path.join(BASE_SAVE_DIR,
                                 'student_model/S_resnet8x4_T_resnet32x4_cifar100_kd_r_1.0_a_0.9_FixedT_4.0_trial_0.95,1,5,10,20_model_')
teacher_path = os.path.join(BASE_SAVE_DIR, 'models/resnet32x4_vanilla/ckpt_epoch_240.pth')

classifier_model_name = 'resnet8x4'
student_model_name = 'resnet8x4'
teacher_model_name = 'resnet32x4'
n_cls = 100
n_groups = 10
num_classes_per_group = n_cls // n_groups
if n_cls % n_groups != 0:
    print(f"警告: 总类别数 {n_cls} 不能被组数 {n_groups} 整除。这会影响类别范围计算。")


# --- 辅助函数：加载模型 ---
def load_model_from_checkpoint(model_cls_fn, path, num_classes, device_to_load):
    model = model_cls_fn(num_classes=num_classes).to(device_to_load)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点文件未找到: {path}")
    try:
        checkpoint = torch.load(path, map_location=device_to_load, weights_only=True)
    except Exception as e:
        print(f"警告: 使用 weights_only=True 加载模型 '{os.path.basename(path)}' 失败 ({e})。尝试不使用 weights_only。")
        print("如果模型来源不受信任，这可能存在安全风险。")
        checkpoint = torch.load(path, map_location=device_to_load, weights_only=False)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    print(f"模型成功从 {os.path.basename(path)} 加载。")
    return model


# --- 可学习门控网络定义 ---
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Softmax will be applied outside if using CrossEntropyLoss, or can be added here if needed elsewhere

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# --- 门控网络特征提取函数 ---
def extract_gating_features(data_batch, student_models_list, device_to_use,
                            num_total_classes, num_model_groups, classes_p_group,
                            feature_type='confidence'):
    """
    从学生模型的输出中提取门控网络的输入特征。
    目前支持 'confidence': 每个学生在其负责域内的最大置信度。
    """
    batch_size_current = data_batch.size(0)
    gating_features = torch.zeros(batch_size_current, num_model_groups, device=device_to_use)

    with torch.no_grad():  # Ensure no gradients are computed for student models here
        all_student_logits_batch = []
        for student_model_item in student_models_list:
            student_logits = student_model_item(data_batch.to(device_to_use))
            all_student_logits_batch.append(student_logits)

        if feature_type == 'confidence':
            all_student_probs_batch = [torch.softmax(logits, dim=1) for logits in all_student_logits_batch]
            for group_idx in range(num_model_groups):
                current_student_probs = all_student_probs_batch[group_idx]
                start_class_idx = group_idx * classes_p_group
                end_class_idx = min(start_class_idx + classes_p_group, num_total_classes)

                if start_class_idx < end_class_idx:  # Check if the slice is valid
                    probs_for_responsible_classes = current_student_probs[:, start_class_idx:end_class_idx]
                    if probs_for_responsible_classes.numel() > 0:
                        score_for_this_group = torch.max(probs_for_responsible_classes, dim=1).values
                    else:  # Should not happen with proper setup and if start_class_idx < end_class_idx
                        score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)
                else:  # If slice is empty (e.g. classes_p_group is 0 or issue with indices)
                    score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)
                gating_features[:, group_idx] = score_for_this_group
        else:
            raise ValueError(f"Unsupported feature_type for gating network: {feature_type}")
    return gating_features


# --- 门控网络训练函数 ---
def train_gating_network(student_models_list, cifar_train_loader, gating_model,
                         optimizer_gating, criterion_gating, num_epochs_gating,
                         device_to_train, n_cls_gating, n_groups_gating, classes_p_group_gating,
                         gating_model_save_path):
    print("\n--- 开始训练门控网络 ---")
    best_gating_acc = 0.0
    for student_model in student_models_list:  # Ensure students are in eval mode and frozen
        student_model.eval()

    for epoch in range(num_epochs_gating):
        gating_model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        epoch_start_time = time.time()

        for batch_idx, (images, true_labels) in enumerate(cifar_train_loader):
            images = images.to(device_to_train)
            true_group_labels_batch = (true_labels // classes_p_group_gating).to(device_to_train)

            optimizer_gating.zero_grad()

            # Extract features for gating network
            # Note: data_batch (images) is already on device_to_train
            gating_input_features = extract_gating_features(images, student_models_list, device_to_train,
                                                            n_cls_gating, n_groups_gating, classes_p_group_gating)

            gating_outputs = gating_model(gating_input_features)
            loss = criterion_gating(gating_outputs, true_group_labels_batch)
            loss.backward()
            optimizer_gating.step()

            running_loss += loss.item() * images.size(0)
            _, predicted_groups = torch.max(gating_outputs, 1)
            correct_preds += (predicted_groups == true_group_labels_batch).sum().item()
            total_preds += images.size(0)

            if (batch_idx + 1) % (max(1, len(cifar_train_loader) // 10)) == 0:
                print(
                    f"  门控训练 Epoch [{epoch + 1}/{num_epochs_gating}], Batch [{batch_idx + 1}/{len(cifar_train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total_preds
        epoch_acc = (correct_preds / total_preds) * 100
        epoch_end_time = time.time()
        print(f"门控训练 Epoch [{epoch + 1}/{num_epochs_gating}] 完成。用时: {epoch_end_time - epoch_start_time:.2f}s, "
              f"平均损失: {epoch_loss:.4f}, 路由准确率: {epoch_acc:.2f}%")

        if epoch_acc > best_gating_acc:
            best_gating_acc = epoch_acc
            torch.save(gating_model.state_dict(), gating_model_save_path)
            print(f"新的最佳门控模型已保存到 {gating_model_save_path} (准确率: {best_gating_acc:.2f}%)")

    print(f"--- 门控网络训练完成 --- 最高准确率: {best_gating_acc:.2f}%")


# --- 数据加载 ---
print("加载 CIFAR-100 数据...")
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
data_dir = './data'  # Ensure this path is correct

# Test DataLoader (as before)
try:
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=cifar_transform)
except Exception as e:
    print(f"自动下载或加载测试数据集时出错: {e}")
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=cifar_transform)

eval_batch_size = 256  # Potentially different from training batch size for gating
num_workers = 4 if device == 'cuda' and os.name != 'nt' and torch.cuda.is_available() else 0
test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
total_images_test = len(test_loader.dataset)
print(f"测试数据集已加载。总图像数: {total_images_test}。批量大小: {eval_batch_size}。")

# Train DataLoader (for Gating Network Training)
try:
    train_dataset_cifar = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=cifar_transform)
except Exception as e:
    print(f"自动下载或加载训练数据集时出错: {e}")
    train_dataset_cifar = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=cifar_transform)

gating_train_batch_size = 128  # Can be different
train_loader_for_gating = DataLoader(train_dataset_cifar, batch_size=gating_train_batch_size, shuffle=True,
                                     num_workers=num_workers)
print(f"训练数据集 (用于门控网络) 已加载。总图像数: {len(train_dataset_cifar)}。批量大小: {gating_train_batch_size}。")

# --- 加载学生模型 ---
students = []
print(f"加载学生模型 ({student_model_name})...")
for i in range(1, n_groups + 1):
    current_student_path = f'{student_base_path}{i}/{student_model_name}_last_model_{i}.pth'
    print(f"加载学生模型 {i} (对应组索引 {i - 1})...")
    student = load_model_from_checkpoint(model_dict[student_model_name], current_student_path, num_classes=n_cls,
                                         device_to_load=device)
    student.eval()  # Set to eval mode
    students.append(student)


# --- 预测函数（支持批量） - Routed Student Only Prediction ---
def predict_batch_student_only_after_routing(data_batch, student_models_list, device_to_use,
                                             num_total_classes, num_model_groups, classes_p_group,
                                             routing_method='confidence',  # Added routing_method
                                             gating_model_instance=None,  # For learnable gating
                                             original_classifier_model=None):  # For original classifier routing
    data_batch = data_batch.to(device_to_use)
    batch_size_current = data_batch.size(0)
    predicted_group_indices = None

    with torch.no_grad():
        # 1. 获取所有学生模型对当前批次的logits输出 (common for many methods or for final prediction)
        all_student_logits_batch = []
        for student_model_item in student_models_list:
            student_logits = student_model_item(data_batch)
            all_student_logits_batch.append(student_logits)

        # --- START ROUTING LOGIC ---
        if routing_method == 'confidence':
            all_student_probs_batch = [torch.softmax(logits, dim=1) for logits in all_student_logits_batch]
            group_scores_for_batch = torch.zeros(batch_size_current, num_model_groups, device=device_to_use)
            for group_idx in range(num_model_groups):
                current_student_probs = all_student_probs_batch[group_idx]
                start_class_idx = group_idx * classes_p_group
                end_class_idx = min(start_class_idx + classes_p_group, num_total_classes)
                if start_class_idx < end_class_idx:
                    probs_for_responsible_classes = current_student_probs[:, start_class_idx:end_class_idx]
                    if probs_for_responsible_classes.numel() > 0:
                        score_for_this_group = torch.max(probs_for_responsible_classes, dim=1).values
                    else:
                        score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)
                else:
                    score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)
                group_scores_for_batch[:, group_idx] = score_for_this_group
            predicted_group_indices = torch.argmax(group_scores_for_batch, dim=1).cpu().numpy()

        elif routing_method == 'sum_logits':
            # (Implementation from your previous code - ensure it's correct for your needs)
            group_scores_for_batch = torch.zeros(batch_size_current, num_model_groups, device=device_to_use)
            for group_idx in range(num_model_groups):
                current_student_logits = all_student_logits_batch[group_idx]  # Logits already computed
                start_class_idx = group_idx * classes_p_group
                end_class_idx = min(start_class_idx + classes_p_group, num_total_classes)
                if start_class_idx < end_class_idx:
                    logits_for_responsible_classes = current_student_logits[:, start_class_idx:end_class_idx]
                    score_for_this_group = torch.sum(logits_for_responsible_classes, dim=1)
                else:
                    score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)
                group_scores_for_batch[:, group_idx] = score_for_this_group
            predicted_group_indices = torch.argmax(group_scores_for_batch, dim=1).cpu().numpy()

        elif routing_method == 'learnable_gating':
            if gating_model_instance is None:
                raise ValueError("Gating model instance must be provided for 'learnable_gating' routing.")
            gating_model_instance.eval()  # Ensure gating model is in eval mode
            # data_batch is already on device_to_use from the start of the function
            gating_input_features = extract_gating_features(data_batch, student_models_list, device_to_use,
                                                            num_total_classes, num_model_groups, classes_p_group)
            gating_outputs = gating_model_instance(gating_input_features)
            predicted_group_indices = torch.argmax(gating_outputs, dim=1).cpu().numpy()

        elif routing_method == 'original_classifier':
            if original_classifier_model is None:
                raise ValueError("Original classifier model must be provided for 'original_classifier' routing.")
            original_classifier_model.eval()
            group_preds_raw_outputs = original_classifier_model(data_batch)  # data_batch is on device
            predicted_group_indices = torch.argmax(group_preds_raw_outputs, dim=1).cpu().numpy()

        else:
            raise ValueError(f"Unsupported routing_method: {routing_method}")
        # --- END ROUTING LOGIC ---

        batch_final_predictions = []
        batch_student_raw_predictions = []

        for i in range(batch_size_current):
            predicted_group_idx_for_image = predicted_group_indices[i]
            selected_student_all_class_logits = all_student_logits_batch[predicted_group_idx_for_image][i, :].unsqueeze(
                0)
            student_pred_class_for_image = torch.argmax(selected_student_all_class_logits, dim=1).item()

            final_prediction_for_image = student_pred_class_for_image

            batch_student_raw_predictions.append(student_pred_class_for_image)
            batch_final_predictions.append(final_prediction_for_image)

    return (np.array(batch_final_predictions),
            predicted_group_indices,
            np.array(batch_student_raw_predictions))


# --- 主要执行流程 ---

# 1. 选择路由方法
# Options: 'confidence', 'sum_logits', 'learnable_gating', 'original_classifier'
SELECTED_ROUTING_METHOD = 'learnable_gating'

# 2. 门控网络特定设置 (如果使用 'learnable_gating')
gating_model_instance = None
gating_model_filename = f"gating_mlp_input{n_groups}_hidden64_output{n_groups}.pth"  # Example filename
gating_model_full_path = os.path.join(GATING_MODEL_DIR, gating_model_filename)

if SELECTED_ROUTING_METHOD == 'learnable_gating':
    gating_input_dim = n_groups  # Based on 'confidence' feature type (one max confidence per group)
    gating_hidden_dim = 64  # Hyperparameter
    gating_output_dim = n_groups

    gating_model_instance = GatingNetwork(gating_input_dim, gating_hidden_dim, gating_output_dim).to(device)

    # 尝试加载预训练的门控模型，否则进行训练
    if os.path.exists(gating_model_full_path):
        print(f"加载预训练的门控网络模型从: {gating_model_full_path}")
        gating_model_instance.load_state_dict(torch.load(gating_model_full_path, map_location=device))
        gating_model_instance.eval()
    else:
        print(f"未找到预训练的门控模型。现在开始训练...")
        # Gating Network Training Hyperparameters
        gating_lr = 0.001
        gating_epochs = 10  # Adjust as needed
        optimizer_gating = optim.Adam(gating_model_instance.parameters(), lr=gating_lr)
        criterion_gating = nn.CrossEntropyLoss()

        train_gating_network(students, train_loader_for_gating, gating_model_instance,
                             optimizer_gating, criterion_gating, gating_epochs,
                             device, n_cls, n_groups, num_classes_per_group,
                             gating_model_full_path)
        # After training, load the best saved model (already done by train_gating_network)
        gating_model_instance.load_state_dict(torch.load(gating_model_full_path, map_location=device))
        gating_model_instance.eval()

# 3. 如果使用 'original_classifier' 路由, 加载它
original_classifier_for_routing = None
if SELECTED_ROUTING_METHOD == 'original_classifier':
    print(f"为路由加载原始分类器 ({classifier_model_name})...")
    try:
        original_classifier_for_routing = load_model_from_checkpoint(
            model_dict[classifier_model_name],
            classifier_path,
            num_classes=n_groups,  # Original classifier predicts groups
            device_to_load=device
        )
        original_classifier_for_routing.eval()
    except FileNotFoundError:
        print(f"错误: 原始分类器模型文件 '{classifier_path}' 未找到。不能使用 'original_classifier' 路由。")
        exit()

# --- 处理测试集进行评估 ---
print(f"\n开始处理测试集 (路由方法: {SELECTED_ROUTING_METHOD}, 最终预测总是来自选定的学生模型)...")

all_true_labels = []
all_final_predictions = []
all_predicted_group_indices_by_router = []
all_student_raw_predictions = []
start_time = time.time()

for batch_idx, (current_data_batch, current_target_batch) in enumerate(test_loader):
    (final_preds_b,
     group_indices_router_b,
     student_raw_preds_b) = predict_batch_student_only_after_routing(
        current_data_batch, students, device, n_cls, n_groups, num_classes_per_group,
        routing_method=SELECTED_ROUTING_METHOD,
        gating_model_instance=gating_model_instance,  # Will be None if not 'learnable_gating'
        original_classifier_model=original_classifier_for_routing  # Will be None if not 'original_classifier'
    )

    all_true_labels.extend(current_target_batch.numpy())
    all_final_predictions.extend(final_preds_b)
    all_predicted_group_indices_by_router.extend(group_indices_router_b)
    all_student_raw_predictions.extend(student_raw_preds_b)

    processed_count = (batch_idx * eval_batch_size) + current_data_batch.size(0)
    if (batch_idx + 1) % (max(1, len(test_loader) // 20)) == 0 or (batch_idx + 1) == len(test_loader):
        current_time_val = time.time()
        elapsed = current_time_val - start_time
        progress_percentage = (processed_count / total_images_test) * 100
        print(
            f"  已处理 {processed_count}/{total_images_test} 张图像 ({progress_percentage:.1f}%) | 已用时间: {elapsed:.1f}秒")

end_time = time.time()
total_processing_time = end_time - start_time
avg_time_per_image = total_processing_time / total_images_test if total_images_test > 0 else 0
print(f"测试集处理完成。总耗时: {total_processing_time:.2f}秒")
print(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒")

# --- 计算指标 ---
true_labels_np = np.array(all_true_labels)
final_predictions_np = np.array(all_final_predictions)
predicted_group_indices_router_np = np.array(all_predicted_group_indices_by_router)
true_group_indices_np = true_labels_np // num_classes_per_group

routing_accuracy_metric_name = f"{SELECTED_ROUTING_METHOD.replace('_', ' ').title()} 路由准确率"
task_assignment_accuracy = (predicted_group_indices_router_np == true_group_indices_np).mean() * 100 if len(
    predicted_group_indices_router_np) > 0 else 0

student_model_decision_percentage = 100.0
overall_system_accuracy = (final_predictions_np == true_labels_np).mean() * 100 if len(final_predictions_np) > 0 else 0
accuracy_when_student_model_used = overall_system_accuracy

# --- 打印结果 ---
print(f"\n--- 联合系统评估结果 (路由: {SELECTED_ROUTING_METHOD}, 最终预测总是来自选定学生) ---")
print(f"{routing_accuracy_metric_name} (任务分配给正确学生组): {task_assignment_accuracy:.2f}%")
print(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}% (设计如此)")
print(f"当使用学生模型决策时的准确率 (即系统整体准确率): {accuracy_when_student_model_used:.2f}%")
print(f"联合模型整体准确率: {overall_system_accuracy:.2f}%")
print(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒")

# --- 保存结果到文件 ---
results_dir = 'evaluation_results'
os.makedirs(results_dir, exist_ok=True)

filename_c_part = f"{SELECTED_ROUTING_METHOD.replace('_', '')}RoutedStudentOnly"
filename_s_part = student_model_name
results_filename = f"eval_{filename_c_part}_s{filename_s_part}.txt"
results_file_path = os.path.join(results_dir, results_filename)

with open(results_file_path, 'w', encoding='utf-8') as f:
    f.write(f"--- 联合系统评估结果 (路由: {SELECTED_ROUTING_METHOD}, 最终预测总是来自选定学生) ---\n")
    f.write(f"{routing_accuracy_metric_name} (任务分配给正确学生组): {task_assignment_accuracy:.2f}%\n")
    f.write(f"  - 定义: 此准确率衡量路由方法是否成功将图像分配给负责其真实类别所属范围的学生模型组。\n")
    f.write(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}% (设计如此)\n")
    f.write(f"当使用学生模型决策时的准确率 (即系统整体准确率): {accuracy_when_student_model_used:.2f}%\n")
    f.write(f"联合模型整体准确率: {overall_system_accuracy:.2f}%\n")
    f.write(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒\n")
    f.write(f"\n--- 配置信息 ---\n")
    f.write(f"路由方法: {SELECTED_ROUTING_METHOD}\n")
    if SELECTED_ROUTING_METHOD == 'learnable_gating' and gating_model_instance:
        f.write(f"门控网络模型: {gating_model_filename}\n")
        f.write(f"  - 输入维度: {gating_input_dim}, 隐藏层维度: {gating_hidden_dim}, 输出维度: {gating_output_dim}\n")
    f.write(f"最终决策: 总是来自被路由选中的学生模型 (无教师回退)\n")
    f.write(f"总类别数: {n_cls}, 组数: {n_groups}, 每组理论类别数: {num_classes_per_group}\n")
    f.write(f"学生模型类型: {student_model_name}, 基础路径: {student_base_path}\n")
    f.write(f"测试数据批次大小: {eval_batch_size}, 工作进程数: {num_workers}\n")
    f.write(f"运行设备: {device}\n")

print(f"结果已保存到: {results_file_path}")
print("评估完毕。")

