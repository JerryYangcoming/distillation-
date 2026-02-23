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
classifier_path = 'save/classifier_models/classifier_resnet8x4_cifar100_opt_adamw_lr_0.001_wd_0.0001_aug_适中_ls_0.1_trial_3/ckpt_epoch_240.pth'
# CORRECTED student_base_path as per previous discussion
student_base_path = 'save/student_model/S_resnet8x4_T_resnet32x4_cifar100_kd_r_1.0_a_0.9_FixedT_4.0_trial_0.95,1,5,10,20_model_'
teacher_path = 'save/models/resnet32x4_vanilla/ckpt_epoch_240.pth'
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

batch_size = 256  # You might want to adjust this based on your GPU memory
num_workers = 4 if device == 'cuda' and os.name != 'nt' and torch.cuda.is_available() else 0
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
total_images = len(test_loader.dataset)
print(f"测试数据集已加载。总图像数: {total_images}。批量大小: {batch_size}。工作进程数: {num_workers}。")

# --- 加载模型 ---
# Original classifier is loaded here for potential comparison or if you want to switch back
# but it won't be used in the new "Confidence-Based Routing" by default.
print(f"加载原始分类器 ({classifier_model_name}) 供参考...")
original_classifier = load_model_from_checkpoint(model_dict[classifier_model_name], classifier_path,
                                                 num_classes=n_groups, device=device)
original_classifier.eval()

students = []
print(f"加载学生模型 ({student_model_name})...")
for i in range(1, n_groups + 1):
    current_student_path = f'{student_base_path}{i}/{student_model_name}_last_model_{i}.pth'
    print(f"加载学生模型 {i} (对应组索引 {i - 1})...")
    student = load_model_from_checkpoint(model_dict[student_model_name], current_student_path, num_classes=n_cls,
                                         device=device)
    student.eval()
    students.append(student)

print(f"加载教师模型 ({teacher_model_name})...")
teacher = load_model_from_checkpoint(model_dict[teacher_model_name], teacher_path, num_classes=n_cls, device=device)
teacher.eval()


# --- 预测函数（支持批量） - Confidence-Based Routing ---
def predict_batch_confidence_routing(data_batch, student_models_list, teacher_model_instance, device_to_use,
                                     num_total_classes, num_model_groups, classes_p_group):
    """
    对一个批次的数据进行预测，使用基于学生对其负责类别最大置信度的路由机制。
    """
    data_batch = data_batch.to(device_to_use)
    batch_size_current = data_batch.size(0)

    with torch.no_grad():
        # 1. 获取所有学生模型对当前批次的logits输出，并转换为概率
        all_student_logits_batch = []
        all_student_probs_batch = []
        for student_model_item in student_models_list:
            student_logits = student_model_item(data_batch)
            all_student_logits_batch.append(student_logits)
            all_student_probs_batch.append(torch.softmax(student_logits, dim=1))  # Convert to probabilities

        # 2. 计算每个组的“最大置信度得分”
        # group_confidence_scores_for_batch 存储每个学生对其负责类别内的最大预测概率
        group_confidence_scores_for_batch = torch.zeros(batch_size_current, num_model_groups, device=device_to_use)

        for group_idx in range(num_model_groups):
            current_student_probs = all_student_probs_batch[
                group_idx]  # Probs from student for this group [batch_size, n_cls]

            start_class_idx = group_idx * classes_p_group
            end_class_idx = min(start_class_idx + classes_p_group, num_total_classes)

            # Probs for the classes this student is responsible for
            probs_for_responsible_classes = current_student_probs[:,
                                            start_class_idx:end_class_idx]  # [batch_size, classes_p_group]

            if probs_for_responsible_classes.numel() > 0:  # Ensure there are classes in this group
                confidence_score_for_this_group = torch.max(probs_for_responsible_classes, dim=1).values  # [batch_size]
            else:  # Should not happen with proper setup
                confidence_score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)

            group_confidence_scores_for_batch[:, group_idx] = confidence_score_for_this_group

        # 3. 根据最高置信度得分确定预测的组索引
        predicted_group_indices_confidence = torch.argmax(group_confidence_scores_for_batch, dim=1).cpu().numpy()
        # --- 新决策逻辑结束 ---

        teacher_outputs_raw = teacher_model_instance(data_batch)  # Teacher processes the whole batch once
        teacher_pred_classes = torch.argmax(teacher_outputs_raw, dim=1).cpu().numpy()

        batch_final_predictions = []
        batch_used_student_flags = []
        batch_student_raw_predictions = []  # For storing the class predicted by the chosen student

        for i in range(batch_size_current):  # Iterate over each image in the batch
            predicted_group_idx_for_image = predicted_group_indices_confidence[i]
            teacher_pred_for_image = teacher_pred_classes[i]

            # The selected student's full logits (for all n_cls) were already computed
            selected_student_all_class_logits = all_student_logits_batch[predicted_group_idx_for_image][i, :].unsqueeze(
                0)  # [1, n_cls]
            student_pred_class_for_image = torch.argmax(selected_student_all_class_logits, dim=1).item()
            batch_student_raw_predictions.append(student_pred_class_for_image)

            # Check if the student's prediction is within its assigned group's responsibility
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
            predicted_group_indices_confidence,  # Group indices from confidence routing
            np.array(batch_student_raw_predictions),  # Student's actual class prediction
            teacher_pred_classes)  # Teacher's actual class prediction


# --- 处理测试集 ---
print("\n开始处理测试集 (使用新的基于学生最大置信度的路由机制)...")
all_true_labels = []
all_final_predictions = []
all_used_student_flags = []
all_predicted_group_indices_by_confidence = []
all_student_raw_predictions = []
all_teacher_raw_predictions = []
start_time = time.time()

num_classes_per_group = n_cls // n_groups
if n_cls % n_groups != 0:
    print(f"警告: 总类别数 {n_cls} 不能被组数 {n_groups} 整除。这会影响类别范围计算。")

for batch_idx, (current_data_batch, current_target_batch) in enumerate(test_loader):
    (final_preds_b,
     used_students_flags_b,
     group_indices_confidence_b,
     student_raw_preds_b,
     teacher_raw_preds_b) = predict_batch_confidence_routing(  # Call the new function
        current_data_batch, students, teacher, device, n_cls, n_groups, num_classes_per_group
    )

    all_true_labels.extend(current_target_batch.numpy())
    all_final_predictions.extend(final_preds_b)
    all_used_student_flags.extend(used_students_flags_b)
    all_predicted_group_indices_by_confidence.extend(group_indices_confidence_b)
    all_student_raw_predictions.extend(student_raw_preds_b)
    all_teacher_raw_predictions.extend(teacher_raw_preds_b)

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
final_predictions_np = np.array(all_final_predictions)
used_student_flags_np = np.array(all_used_student_flags)
predicted_group_indices_confidence_np = np.array(all_predicted_group_indices_by_confidence)
student_raw_predictions_np = np.array(all_student_raw_predictions)
teacher_raw_predictions_np = np.array(all_teacher_raw_predictions)

true_group_indices_np = true_labels_np // num_classes_per_group

# 1. "任务分配"准确率: 衡量新的置信度路由方法是否将图像分配给了正确的学生模型组。
confidence_routing_accuracy = (predicted_group_indices_confidence_np == true_group_indices_np).mean() * 100 if len(
    predicted_group_indices_confidence_np) > 0 else 0

# 2. 使用学生模型决策的比例
student_model_decision_percentage = used_student_flags_np.mean() * 100 if len(used_student_flags_np) > 0 else 0

# 3. 联合模型的整体准确率
overall_system_accuracy = (final_predictions_np == true_labels_np).mean() * 100 if len(final_predictions_np) > 0 else 0

# 4. 当系统决策使用学生模型时，这些决策的准确率
student_decision_indices = used_student_flags_np.astype(bool)
accuracy_when_student_model_used = 0
if student_decision_indices.sum() > 0:
    accuracy_when_student_model_used = (final_predictions_np[student_decision_indices] == true_labels_np[
        student_decision_indices]).mean() * 100

# 5. 当系统决策使用教师模型时 (因学生预测越界), 这些决策的准确率
teacher_decision_indices = ~used_student_flags_np.astype(bool)
accuracy_when_teacher_model_used = 0
if teacher_decision_indices.sum() > 0:
    accuracy_when_teacher_model_used = (final_predictions_np[teacher_decision_indices] == true_labels_np[
        teacher_decision_indices]).mean() * 100

# --- 打印结果 ---
print("\n--- 联合系统评估结果 (使用基于学生最大置信度的路由) ---")
print(f"最大置信度路由准确率 (任务分配给正确学生组): {confidence_routing_accuracy:.2f}%")
print(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}%")
print(f"当使用学生模型决策时的准确率: {accuracy_when_student_model_used:.2f}%")
print(f"当使用教师模型决策时的准确率 (因学生预测越界): {accuracy_when_teacher_model_used:.2f}%")
print(f"联合模型整体准确率: {overall_system_accuracy:.2f}%")
print(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒")

# --- 保存结果到文件 ---
results_dir = 'evaluation_results'
os.makedirs(results_dir, exist_ok=True)

filename_c_part = "MaxConfidenceRouted"  # Indicate new routing
filename_t_part = teacher_model_name
filename_s_part = student_model_name

results_filename = f"c{filename_c_part}_t{filename_t_part}_s{filename_s_part}.txt"
results_file_path = os.path.join(results_dir, results_filename)

with open(results_file_path, 'w', encoding='utf-8') as f:
    f.write(f"--- 联合系统评估结果 (使用基于学生最大置信度的路由) ---\n")
    f.write(f"最大置信度路由准确率 (任务分配给正确学生组): {confidence_routing_accuracy:.2f}%\n")
    f.write(
        f"  - 定义: 此准确率衡量基于“学生对其负责类别内的最大预测概率”的路由方法是否成功将图像分配给负责其真实类别所属范围的学生模型组。\n")
    f.write(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}%\n")
    f.write(f"当使用学生模型决策时的准确率: {accuracy_when_student_model_used:.2f}%\n")
    f.write(f"当使用教师模型决策时的准确率 (因学生预测越界): {accuracy_when_teacher_model_used:.2f}%\n")
    f.write(f"联合模型整体准确率: {overall_system_accuracy:.2f}%\n")
    f.write(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒\n")
    f.write(f"\n--- 配置信息 ---\n")
    f.write(f"路由方法: 基于所有学生模型对其负责类别内的最大预测概率 (Max Confidence Routing)\n")
    f.write(f"总类别数: {n_cls}, 组数: {n_groups}, 每组理论类别数: {num_classes_per_group}\n")
    f.write(f"学生模型类型: {student_model_name}, 基础路径: {student_base_path}\n")
    f.write(f"教师模型类型: {teacher_model_name}, 路径: {teacher_path}\n")
    f.write(f"测试数据批次大小: {batch_size}, 工作进程数: {num_workers}\n")
    f.write(f"运行设备: {device}\n")

print(f"结果已保存到: {results_file_path}")
print("评估完毕。")

