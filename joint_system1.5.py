# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn  # Added for NLLLoss and Parameter
import torch.optim as optim  # Added for optimizer
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler  # Added for validation split
from models import model_dict  # 请确保此导入在您的环境中有效

# --- 基本配置 ---
BASE_SAVE_DIR = 'save'
CALIBRATION_DIR = os.path.join(BASE_SAVE_DIR, 'calibration_temperatures')
os.makedirs(CALIBRATION_DIR, exist_ok=True)

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

# --- 模型路径 ---
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

    state_dict_key = 'model' if 'model' in checkpoint else 'state_dict' if 'state_dict' in checkpoint else None
    state_dict = checkpoint[state_dict_key] if state_dict_key else checkpoint
    model.load_state_dict(state_dict)
    print(f"模型成功从 {os.path.basename(path)} 加载。")
    return model


# --- 温度缩放辅助类和函数 ---
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification model
        output corresponding to [batch_size, num_classes]
    """

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initial temperature

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should be outside the model, applied to an instance
    def set_temperature(self, valid_loader, device_to_use, cross_validate='NLL', lr=0.01, max_iter=50):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.to(device_to_use)
        nll_criterion = nn.CrossEntropyLoss().to(device_to_use)
        ece_criterion = _ECELoss().to(device_to_use)  # If you want to use ECE

        # First: collect all D_val logits and labels
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input_val, label_val in valid_loader:
                input_val = input_val.to(device_to_use)
                logits = self.model(input_val)  # Get original logits
                logits_list.append(logits)
                labels_list.append(label_val)
        logits_all = torch.cat(logits_list).to(device_to_use)
        labels_all = torch.cat(labels_list).to(device_to_use)

        # Next: optimize temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        print(f"开始校准温度 (初始 T={self.temperature.item():.3f})...")

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits_all), labels_all)
            loss.backward()
            return loss

        optimizer.step(eval)
        print(f"最佳温度已找到: {self.temperature.item():.3f}")
        return self.temperature.item()


# (Optional) For ECE calculation if you want to track it
class _ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


# --- 数据加载 ---
print("加载 CIFAR-100 数据...")
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
data_dir = './data'

# Test DataLoader
try:
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=cifar_transform)
except Exception as e:
    print(f"自动下载或加载测试数据集时出错: {e}")
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=cifar_transform)

eval_batch_size = 256
num_workers = 4 if device == 'cuda' and os.name != 'nt' and torch.cuda.is_available() else 0
test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
total_images_test = len(test_loader.dataset)
print(f"测试数据集已加载。总图像数: {total_images_test}。批量大小: {eval_batch_size}。")

# Validation DataLoader (for Temperature Scaling calibration)
VALID_SPLIT_PERCENT = 0.1  # Use 10% of training data for validation
try:
    train_dataset_full = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=cifar_transform)
except Exception as e:
    print(f"自动下载或加载训练数据集时出错: {e}")
    train_dataset_full = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=cifar_transform)

num_train = len(train_dataset_full)
indices = list(range(num_train))
split = int(np.floor(VALID_SPLIT_PERCENT * num_train))
np.random.shuffle(indices)  # Shuffle once
train_idx, valid_idx = indices[split:], indices[:split]

# We only need the validation subset for calibration here
valid_sampler = SubsetRandomSampler(valid_idx)
calibration_val_loader = DataLoader(train_dataset_full, batch_size=eval_batch_size, sampler=valid_sampler,
                                    num_workers=num_workers)
print(f"校准验证集已创建。大小: {len(valid_idx)} 张图像。")

# --- 加载模型 ---
print(f"加载原始分类器 ({classifier_model_name}) 供参考...")
# original_classifier = load_model_from_checkpoint(model_dict[classifier_model_name], classifier_path,
#  num_classes=n_groups, device_to_load=device)
# original_classifier.eval() # Not used in confidence routing directly

students = []
student_temperatures = []  # To store learned temperatures
print(f"加载学生模型 ({student_model_name}) 并进行温度校准...")
for i in range(1, n_groups + 1):
    student_idx_0_based = i - 1
    current_student_path = f'{student_base_path}{i}/{student_model_name}_last_model_{i}.pth'
    print(f"加载学生模型 {i} (组索引 {student_idx_0_based}) 从 {current_student_path}...")
    student_model_raw = load_model_from_checkpoint(model_dict[student_model_name], current_student_path,
                                                   num_classes=n_cls,
                                                   device_to_load=device)
    student_model_raw.eval()  # Ensure model is in eval mode before wrapping

    # Temperature scaling
    temp_filename = os.path.join(CALIBRATION_DIR, f'student_{student_idx_0_based}_temp.pth')
    scaled_student_model = ModelWithTemperature(student_model_raw)  # Wrap the raw model

    if os.path.exists(temp_filename):
        saved_temp = torch.load(temp_filename, map_location=device)
        scaled_student_model.temperature.data = torch.tensor([saved_temp], device=device)
        print(f"学生 {i}: 已加载保存的温度: {saved_temp:.3f}")
    else:
        print(f"学生 {i}: 开始温度校准...")
        scaled_student_model.set_temperature(calibration_val_loader, device)
        torch.save(scaled_student_model.temperature.item(), temp_filename)
        print(f"学生 {i}: 校准完成。温度 {scaled_student_model.temperature.item():.3f} 已保存到 {temp_filename}")

    students.append(scaled_student_model)  # Store the scaled model
    student_temperatures.append(scaled_student_model.temperature.item())  # Also keep track of T values

print(f"加载教师模型 ({teacher_model_name})...")
teacher = load_model_from_checkpoint(model_dict[teacher_model_name], teacher_path, num_classes=n_cls,
                                     device_to_load=device)
teacher.eval()


# --- 预测函数（支持批量） - Calibrated Confidence-Based Routing ---
def predict_batch_calibrated_confidence_routing(data_batch, scaled_student_models_list, teacher_model_instance,
                                                device_to_use,
                                                num_total_classes, num_model_groups, classes_p_group):
    data_batch = data_batch.to(device_to_use)
    batch_size_current = data_batch.size(0)

    with torch.no_grad():
        all_student_scaled_logits_batch = []
        all_student_calibrated_probs_batch = []
        for scaled_student_model_item in scaled_student_models_list:
            # The ModelWithTemperature wrapper already applies scaling in its forward pass
            # if we call scaled_student_model_item(data_batch) it will give scaled logits
            # So, we get scaled logits directly.
            scaled_logits = scaled_student_model_item(data_batch)  # This now returns logits / T
            all_student_scaled_logits_batch.append(scaled_logits)
            all_student_calibrated_probs_batch.append(torch.softmax(scaled_logits, dim=1))

        group_confidence_scores_for_batch = torch.zeros(batch_size_current, num_model_groups, device=device_to_use)
        for group_idx in range(num_model_groups):
            current_student_calibrated_probs = all_student_calibrated_probs_batch[group_idx]
            start_class_idx = group_idx * classes_p_group
            end_class_idx = min(start_class_idx + classes_p_group, num_total_classes)

            if start_class_idx < end_class_idx:
                probs_for_responsible_classes = current_student_calibrated_probs[:, start_class_idx:end_class_idx]
                if probs_for_responsible_classes.numel() > 0:
                    confidence_score_for_this_group = torch.max(probs_for_responsible_classes, dim=1).values
                else:
                    confidence_score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)
            else:
                confidence_score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)
            group_confidence_scores_for_batch[:, group_idx] = confidence_score_for_this_group

        predicted_group_indices_confidence = torch.argmax(group_confidence_scores_for_batch, dim=1).cpu().numpy()

        teacher_outputs_raw = teacher_model_instance(data_batch)
        teacher_pred_classes = torch.argmax(teacher_outputs_raw, dim=1).cpu().numpy()

        batch_final_predictions = []
        batch_used_student_flags = []
        batch_student_raw_predictions = []

        for i in range(batch_size_current):
            predicted_group_idx_for_image = predicted_group_indices_confidence[i]
            teacher_pred_for_image = teacher_pred_classes[i]

            # Use the scaled logits for student's own prediction
            selected_student_scaled_logits = all_student_scaled_logits_batch[predicted_group_idx_for_image][i,
                                             :].unsqueeze(0)
            student_pred_class_for_image = torch.argmax(selected_student_scaled_logits,
                                                        dim=1).item()  # Prediction from scaled logits
            batch_student_raw_predictions.append(student_pred_class_for_image)

            if student_pred_class_for_image // classes_p_group == predicted_group_idx_for_image:
                final_prediction_for_image = student_pred_class_for_image
                used_student_flag_for_image = True
            else:
                final_prediction_for_image = teacher_pred_for_image
                used_student_flag_for_image = False

            batch_final_predictions.append(final_prediction_for_image)
            batch_used_student_flags.append(used_student_flag_for_image)

    return (np.array(batch_final_predictions),
            np.array(batch_used_student_flags),
            predicted_group_indices_confidence,
            np.array(batch_student_raw_predictions),
            teacher_pred_classes)


# --- 处理测试集 ---
print("\n开始处理测试集 (使用校准的基于学生最大置信度的路由机制)...")
all_true_labels = []
all_final_predictions = []
all_used_student_flags = []
all_predicted_group_indices_by_confidence = []
all_student_raw_predictions = []
all_teacher_raw_predictions = []
start_time = time.time()

for batch_idx, (current_data_batch, current_target_batch) in enumerate(test_loader):
    (final_preds_b,
     used_students_flags_b,
     group_indices_confidence_b,
     student_raw_preds_b,
     teacher_raw_preds_b) = predict_batch_calibrated_confidence_routing(  # Call the new function
        current_data_batch, students, teacher, device, n_cls, n_groups, num_classes_per_group
    )

    all_true_labels.extend(current_target_batch.numpy())
    all_final_predictions.extend(final_preds_b)
    all_used_student_flags.extend(used_students_flags_b)
    all_predicted_group_indices_by_confidence.extend(group_indices_confidence_b)
    all_student_raw_predictions.extend(student_raw_preds_b)
    all_teacher_raw_predictions.extend(teacher_raw_preds_b)

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
used_student_flags_np = np.array(all_used_student_flags)
predicted_group_indices_confidence_np = np.array(all_predicted_group_indices_by_confidence)
student_raw_predictions_np = np.array(all_student_raw_predictions)
teacher_raw_predictions_np = np.array(all_teacher_raw_predictions)

true_group_indices_np = true_labels_np // num_classes_per_group

confidence_routing_accuracy = (predicted_group_indices_confidence_np == true_group_indices_np).mean() * 100 if len(
    predicted_group_indices_confidence_np) > 0 else 0
student_model_decision_percentage = used_student_flags_np.mean() * 100 if len(used_student_flags_np) > 0 else 0
overall_system_accuracy = (final_predictions_np == true_labels_np).mean() * 100 if len(final_predictions_np) > 0 else 0

student_decision_indices = used_student_flags_np.astype(bool)
accuracy_when_student_model_used = 0
if student_decision_indices.sum() > 0:
    accuracy_when_student_model_used = (final_predictions_np[student_decision_indices] == true_labels_np[
        student_decision_indices]).mean() * 100

teacher_decision_indices = ~used_student_flags_np.astype(bool)
accuracy_when_teacher_model_used = 0
if teacher_decision_indices.sum() > 0:
    accuracy_when_teacher_model_used = (final_predictions_np[teacher_decision_indices] == true_labels_np[
        teacher_decision_indices]).mean() * 100

# --- 打印结果 ---
print("\n--- 联合系统评估结果 (使用校准的基于学生最大置信度的路由) ---")
print(f"学生模型的平均校准温度: {np.mean(student_temperatures):.3f} (Std: {np.std(student_temperatures):.3f})")
print(f"各个学生模型的温度: {[float(f'{t:.3f}') for t in student_temperatures]}")
print(f"校准后最大置信度路由准确率 (任务分配给正确学生组): {confidence_routing_accuracy:.2f}%")
print(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}%")
print(f"当使用学生模型决策时的准确率: {accuracy_when_student_model_used:.2f}%")
print(f"当使用教师模型决策时的准确率 (因学生预测越界): {accuracy_when_teacher_model_used:.2f}%")
print(f"联合模型整体准确率: {overall_system_accuracy:.2f}%")
print(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒")

# --- 保存结果到文件 ---
results_dir = 'evaluation_results'
os.makedirs(results_dir, exist_ok=True)

filename_c_part = "CalibratedMaxConfidenceRouted"
filename_t_part = teacher_model_name
filename_s_part = student_model_name
results_filename = f"eval_c{filename_c_part}_t{filename_t_part}_s{filename_s_part}.txt"  # Added eval_ prefix
results_file_path = os.path.join(results_dir, results_filename)

with open(results_file_path, 'w', encoding='utf-8') as f:
    f.write(f"--- 联合系统评估结果 (使用校准的基于学生最大置信度的路由) ---\n")
    f.write(f"学生模型的平均校准温度: {np.mean(student_temperatures):.3f} (Std: {np.std(student_temperatures):.3f})\n")
    f.write(f"各个学生模型的温度: {[float(f'{t:.3f}') for t in student_temperatures]}\n")
    f.write(f"校准后最大置信度路由准确率 (任务分配给正确学生组): {confidence_routing_accuracy:.2f}%\n")
    f.write(
        f"  - 定义: 此准确率衡量基于“校准后学生对其负责类别内的最大预测概率”的路由方法是否成功将图像分配给负责其真实类别所属范围的学生模型组。\n")
    f.write(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}%\n")
    f.write(f"当使用学生模型决策时的准确率: {accuracy_when_student_model_used:.2f}%\n")
    f.write(f"当使用教师模型决策时的准确率 (因学生预测越界): {accuracy_when_teacher_model_used:.2f}%\n")
    f.write(f"联合模型整体准确率: {overall_system_accuracy:.2f}%\n")
    f.write(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒\n")
    f.write(f"\n--- 配置信息 ---\n")
    f.write(f"路由方法: 基于所有学生模型对其负责类别内的校准后最大预测概率 (Calibrated Max Confidence Routing)\n")
    f.write(f"校准方法: 温度缩放 (Temperature Scaling)，优化 NLL\n")
    f.write(f"总类别数: {n_cls}, 组数: {n_groups}, 每组理论类别数: {num_classes_per_group}\n")
    f.write(f"学生模型类型: {student_model_name}, 基础路径: {student_base_path}\n")
    f.write(f"教师模型类型: {teacher_model_name}, 路径: {teacher_path}\n")
    f.write(f"测试数据批次大小: {eval_batch_size}, 工作进程数: {num_workers}\n")
    f.write(f"运行设备: {device}\n")

print(f"结果已保存到: {results_file_path}")
print("评估完毕。")

