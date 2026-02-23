# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import model_dict  # 请确保此导入在您的环境中有效

# --- 设备检查 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")
if device == 'cuda':
    if torch.cuda.is_available():  # 额外检查以确保安全
        print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
    else:  # 理论上不应发生，但为了稳健性
        print("CUDA 声称可用，但 torch.cuda.is_available() 返回 False。切换到 CPU。")
        device = 'cpu'

# --- 模型路径 ---
classifier_path = 'save/classifier_models/classifier_resnet8x4_cifar100_lr_0.05_decay_0.0005_trial_1/resnet8x4_best.pth'
student_base_path = 'save/student_model/S_resnet8x4_T_resnet32x4_cifar100_kd_r_1.0_a_0.9_FixedT_4.0_trial_0.95,1,5,10,20_model_'
teacher_path = 'save/models/resnet32x4_vanilla/ckpt_epoch_240.pth'  # Still loaded for potential reference/comparison, but not for fallback
classifier_model_name = 'resnet8x4'
student_model_name = 'resnet8x4'
teacher_model_name = 'resnet32x4'
n_cls = 100  # CIFAR-100 有 100 个类别
n_groups = 10  # 10 个组，每组理论上 10 个类别


# --- 辅助函数：加载模型 ---
def load_model_from_checkpoint(model_cls_fn, path, num_classes, device):
    """从检查点加载模型状态字典的辅助函数"""
    model = model_cls_fn(num_classes=num_classes).to(device)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点文件未找到: {path}")
    try:
        # Attempting to load with weights_only=True for security
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"警告: 使用 weights_only=True 加载模型 '{os.path.basename(path)}' 失败 ({e})。尝试不使用 weights_only。")
        print("如果模型来源不受信任，这可能存在安全风险。")
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    print(f"模型成功从 {os.path.basename(path)} 加载。")
    return model


# --- 数据加载 ---
print("加载 CIFAR-100 测试数据...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
data_dir = './data'
try:
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
except Exception as e:
    print(f"自动下载或加载数据集时出错，尝试不使用 download=True（请确保数据已存在于 '{data_dir}'）：{e}")
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)

batch_size = 256
num_workers = 4 if device == 'cuda' and os.name != 'nt' and torch.cuda.is_available() else 0
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
total_images = len(test_loader.dataset)
print(f"测试数据集已加载。总图像数: {total_images}。批量大小: {batch_size}。工作进程数: {num_workers}。")

# --- 加载模型 ---
print(f"加载原始分类器 ({classifier_model_name}) 供参考 (当前未使用)...")
# original_classifier = load_model_from_checkpoint(model_dict[classifier_model_name],
#                                                  classifier_path,
#                                                  num_classes=n_groups, device=device)
# original_classifier.eval() # Commented out as it's not strictly needed for this configuration

students = []
print(f"加载学生模型 ({student_model_name})...")
for i in range(1, n_groups + 1):
    current_student_path = f'{student_base_path}{i}/{student_model_name}_last_model_{i}.pth'
    print(f"加载学生模型 {i} (对应组索引 {i - 1})...")
    student = load_model_from_checkpoint(model_dict[student_model_name], current_student_path, num_classes=n_cls,
                                         device=device)
    student.eval()
    students.append(student)


# Teacher model is loaded here but will not be used for fallback in the predict_batch function.
# It could be used for a separate "teacher-only" baseline evaluation if desired.
# print(f"加载教师模型 ({teacher_model_name}) 供参考 (当前未使用于预测决策)...")
# teacher = load_model_from_checkpoint(model_dict[teacher_model_name], teacher_path, num_classes=n_cls, device=device)
# teacher.eval() # Commented out as it's not strictly needed for this configuration


# --- 预测函数（支持批量） - Routed Student Only Prediction ---
def predict_batch_student_only_after_routing(data_batch, student_models_list, device_to_use,
                                             num_total_classes, num_model_groups, classes_p_group,
                                             routing_method='confidence'):  # Added routing_method
    """
    对一个批次的数据进行预测。
    1. 使用指定的路由方法选择一个学生模型。
    2. 最终预测总是来自被选中的学生模型。
    """
    data_batch = data_batch.to(device_to_use)
    batch_size_current = data_batch.size(0)

    with torch.no_grad():
        # 1. 获取所有学生模型对当前批次的logits输出
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
                probs_for_responsible_classes = current_student_probs[:, start_class_idx:end_class_idx]
                if probs_for_responsible_classes.numel() > 0:
                    score_for_this_group = torch.max(probs_for_responsible_classes, dim=1).values
                else:
                    score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)
                group_scores_for_batch[:, group_idx] = score_for_this_group
            predicted_group_indices = torch.argmax(group_scores_for_batch, dim=1).cpu().numpy()
        elif routing_method == 'sum_logits':
            group_scores_for_batch = torch.zeros(batch_size_current, num_model_groups, device=device_to_use)
            for group_idx in range(num_model_groups):
                current_student_logits = all_student_logits_batch[group_idx]
                start_class_idx = group_idx * classes_p_group
                end_class_idx = min(start_class_idx + classes_p_group, num_total_classes)
                logits_for_responsible_classes = current_student_logits[:, start_class_idx:end_class_idx]
                score_for_this_group = torch.sum(logits_for_responsible_classes, dim=1)
                group_scores_for_batch[:, group_idx] = score_for_this_group
            predicted_group_indices = torch.argmax(group_scores_for_batch, dim=1).cpu().numpy()
        # Add other routing methods here if needed e.g. 'original_classifier'
        # elif routing_method == 'original_classifier':
        #     if 'original_classifier_model' not in globals() or original_classifier_model is None:
        #         raise ValueError("Original classifier model not loaded or passed for 'original_classifier' routing.")
        #     group_preds_raw_outputs = original_classifier_model(data_batch)
        #     predicted_group_indices = torch.argmax(group_preds_raw_outputs, dim=1).cpu().numpy()
        else:
            raise ValueError(f"Unsupported routing_method: {routing_method}")
        # --- END ROUTING LOGIC ---

        batch_final_predictions = []
        batch_student_raw_predictions = []  # For storing the class predicted by the chosen student

        for i in range(batch_size_current):  # Iterate over each image in the batch
            predicted_group_idx_for_image = predicted_group_indices[i]

            # The selected student's full logits (for all n_cls) were already computed
            selected_student_all_class_logits = all_student_logits_batch[predicted_group_idx_for_image][i, :].unsqueeze(
                0)  # [1, n_cls]
            student_pred_class_for_image = torch.argmax(selected_student_all_class_logits, dim=1).item()

            # Final prediction is ALWAYS the selected student's prediction
            final_prediction_for_image = student_pred_class_for_image

            batch_student_raw_predictions.append(student_pred_class_for_image)  # Also store this as the "student raw"
            batch_final_predictions.append(final_prediction_for_image)

    # used_student_flags is always True for all samples, so it's 100%
    # teacher_pred_classes is no longer generated here as it's not used for fallback
    return (np.array(batch_final_predictions),
            predicted_group_indices,
            np.array(batch_student_raw_predictions))


# --- 处理测试集 ---
# CHOOSE YOUR ROUTING METHOD HERE: 'confidence', 'sum_logits'
# If you want to use 'original_classifier', uncomment its loading and pass it to the prediction function.
SELECTED_ROUTING_METHOD = 'confidence'  # Or 'sum_logits'
print(f"\n开始处理测试集 (路由方法: {SELECTED_ROUTING_METHOD}, 最终预测总是来自选定的学生模型)...")

all_true_labels = []
all_final_predictions = []
all_predicted_group_indices_by_router = []
all_student_raw_predictions = []  # This will be same as final_predictions
start_time = time.time()

num_classes_per_group = n_cls // n_groups
if n_cls % n_groups != 0:
    print(f"警告: 总类别数 {n_cls} 不能被组数 {n_groups} 整除。这会影响类别范围计算 (尤其用于路由)。")

# If using original_classifier routing, load it:
# original_classifier_model_for_routing = None
# if SELECTED_ROUTING_METHOD == 'original_classifier':
#     print(f"Loading original classifier for routing...")
#     original_classifier_model_for_routing = load_model_from_checkpoint(model_dict[classifier_model_name],
#                                                                          classifier_path,
#                                                                          num_classes=n_groups, device=device)
#     original_classifier_model_for_routing.eval()


for batch_idx, (current_data_batch, current_target_batch) in enumerate(test_loader):
    (final_preds_b,
     group_indices_router_b,
     student_raw_preds_b) = predict_batch_student_only_after_routing(
        current_data_batch, students, device, n_cls, n_groups, num_classes_per_group,
        routing_method=SELECTED_ROUTING_METHOD
        # Pass original_classifier_model_for_routing if that method is chosen
    )

    all_true_labels.extend(current_target_batch.numpy())
    all_final_predictions.extend(final_preds_b)
    all_predicted_group_indices_by_router.extend(group_indices_router_b)
    all_student_raw_predictions.extend(student_raw_preds_b)

    processed_count = (batch_idx * batch_size) + current_data_batch.size(0)
    if (batch_idx + 1) % (max(1, len(test_loader) // 20)) == 0 or (batch_idx + 1) == len(test_loader):
        current_time = time.time()
        elapsed = current_time - start_time
        progress_percentage = (processed_count / total_images) * 100
        print(
            f"  已处理 {processed_count}/{total_images} 张图像 ({progress_percentage:.1f}%) | 已用时间: {elapsed:.1f}秒")

end_time = time.time()
total_processing_time = end_time - start_time
avg_time_per_image = total_processing_time / total_images if total_images > 0 else 0
print(f"测试集处理完成。总耗时: {total_processing_time:.2f}秒")
print(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒")

# --- 计算指标 ---
true_labels_np = np.array(all_true_labels)
final_predictions_np = np.array(all_final_predictions)  # These are the routed student predictions
predicted_group_indices_router_np = np.array(all_predicted_group_indices_by_router)
# student_raw_predictions_np is the same as final_predictions_np in this setup

true_group_indices_np = true_labels_np // num_classes_per_group

# 1. "任务分配"准确率: 衡量选择的路由方法是否将图像分配给了正确的学生模型组。
routing_accuracy_metric_name = ""
if SELECTED_ROUTING_METHOD == 'confidence':
    routing_accuracy_metric_name = "最大置信度路由准确率"
elif SELECTED_ROUTING_METHOD == 'sum_logits':
    routing_accuracy_metric_name = "Logits总和路由准确率"
elif SELECTED_ROUTING_METHOD == 'original_classifier':
    routing_accuracy_metric_name = "原始分类器路由准确率"
else:
    routing_accuracy_metric_name = f"{SELECTED_ROUTING_METHOD}路由准确率"

task_assignment_accuracy = (predicted_group_indices_router_np == true_group_indices_np).mean() * 100 if len(
    predicted_group_indices_router_np) > 0 else 0

# 2. 使用学生模型决策的比例 - this will be 100% by design
student_model_decision_percentage = 100.0

# 3. 联合模型的整体准确率 (now reflects the accuracy of the routed student models)
overall_system_accuracy = (final_predictions_np == true_labels_np).mean() * 100 if len(final_predictions_np) > 0 else 0

# 4. 准确率 when student model used - this is the same as overall_system_accuracy
accuracy_when_student_model_used = overall_system_accuracy

# 5. 当系统决策使用教师模型时 - N/A in this setup
accuracy_when_teacher_model_used = 0.0  # Or mark as N/A

# --- 打印结果 ---
print(f"\n--- 联合系统评估结果 (路由: {SELECTED_ROUTING_METHOD}, 最终预测总是来自选定学生) ---")
print(f"{routing_accuracy_metric_name} (任务分配给正确学生组): {task_assignment_accuracy:.2f}%")
print(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}% (设计如此)")
print(f"当使用学生模型决策时的准确率 (即系统整体准确率): {accuracy_when_student_model_used:.2f}%")
# print(f"当使用教师模型决策时的准确率 (因学生预测越界): N/A (未使用教师模型进行决策)") # This metric is no longer applicable
print(f"联合模型整体准确率: {overall_system_accuracy:.2f}%")
print(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒")

# --- 保存结果到文件 ---
results_dir = 'evaluation_results'
os.makedirs(results_dir, exist_ok=True)

filename_c_part = f"{SELECTED_ROUTING_METHOD.replace('_', '')}RoutedStudentOnly"
filename_s_part = student_model_name
# Teacher part can be omitted from filename if it's not used at all
# filename_t_part = teacher_model_name

results_filename = f"eval_{filename_c_part}_s{filename_s_part}.txt"
results_file_path = os.path.join(results_dir, results_filename)

with open(results_file_path, 'w', encoding='utf-8') as f:
    f.write(f"--- 联合系统评估结果 (路由: {SELECTED_ROUTING_METHOD}, 最终预测总是来自选定学生) ---\n")
    f.write(f"{routing_accuracy_metric_name} (任务分配给正确学生组): {task_assignment_accuracy:.2f}%\n")
    f.write(f"  - 定义: 此准确率衡量路由方法是否成功将图像分配给负责其真实类别所属范围的学生模型组。\n")
    f.write(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}% (设计如此)\n")
    f.write(f"当使用学生模型决策时的准确率 (即系统整体准确率): {accuracy_when_student_model_used:.2f}%\n")
    # f.write(f"当使用教师模型决策时的准确率: N/A (未使用教师模型进行决策)\n")
    f.write(f"联合模型整体准确率: {overall_system_accuracy:.2f}%\n")
    f.write(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒\n")
    f.write(f"\n--- 配置信息 ---\n")
    f.write(f"路由方法: {SELECTED_ROUTING_METHOD}\n")
    f.write(f"最终决策: 总是来自被路由选中的学生模型 (无教师回退)\n")
    f.write(f"总类别数: {n_cls}, 组数: {n_groups}, 每组理论类别数: {num_classes_per_group}\n")
    f.write(f"学生模型类型: {student_model_name}, 基础路径: {student_base_path}\n")
    # f.write(f"教师模型类型: {teacher_model_name}, 路径: {teacher_path} (仅供参考, 未用于预测决策)\n")
    # f.write(f"原始分类器路径: {classifier_path} (仅供参考, 根据路由选择使用)\n")
    f.write(f"测试数据批次大小: {batch_size}, 工作进程数: {num_workers}\n")
    f.write(f"运行设备: {device}\n")

print(f"结果已保存到: {results_file_path}")
print("评估完毕。")

