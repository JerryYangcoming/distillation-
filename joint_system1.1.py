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
    if torch.cuda.is_available(): # 额外检查以确保安全
        print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
    else: # 理论上不应发生，但为了稳健性
        print("CUDA 声称可用，但 torch.cuda.is_available() 返回 False。切换到 CPU。")
        device = 'cpu'


# --- 模型路径 ---
classifier_path = 'save/classifier_models/classifier_resnet8x4_cifar100_lr_0.05_decay_0.0005_trial_1/resnet8x4_best.pth'
student_base_path = 'save/student_model/S_resnet8x4_T_resnet32x4_cifar100_kd_r_1.0_a_0.9_FixedT_4.0_trial_0.95,1,5,10,20_model_'
teacher_path = 'save/models/resnet32x4_vanilla/ckpt_epoch_240.pth'
classifier_model_name = 'resnet8x4'
student_model_name = 'resnet8x4'
teacher_model_name = 'resnet32x4'
n_cls = 100  # CIFAR-100 有 100 个类别
n_groups = 10  # 10 个组，每组理论上 10 个类别 (例如, 学生模型0 负责类别 0-9, 学生模型1 负责 10-19, 等等)

# --- 辅助函数：加载模型 ---
def load_model_from_checkpoint(model_cls, path, num_classes, device):
    """从检查点加载模型状态字典的辅助函数"""
    model = model_cls(num_classes=num_classes).to(device)  # 直接在目标设备上创建模型
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点文件未找到: {path}")
    try:
        # 尝试使用 weights_only=True 以增强安全性, 如果您的 .pth 文件兼容
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"警告: 使用 weights_only=True 加载模型 '{os.path.basename(path)}' 失败 ({e})。尝试不使用 weights_only。")
        print("如果模型来源不受信任，这可能存在安全风险。")
        checkpoint = torch.load(path, map_location=device)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    print(f"模型成功从 {os.path.basename(path)} 加载。")
    return model

# --- 数据加载（循环外部，只加载一次） ---
print("加载 CIFAR-100 测试数据...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 归一化参数
])
data_dir = './data' # 数据集根目录
try:
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
except Exception as e:
    print(f"自动下载或加载数据集时出错，尝试不使用 download=True（请确保数据已存在于 '{data_dir}'）：{e}")
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)

batch_size = 256
# 根据操作系统和CUDA可用性设置 num_workers
num_workers = 4 if device == 'cuda' and os.name != 'nt' and torch.cuda.is_available() else 0
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
total_images = len(test_loader.dataset)
print(f"测试数据集已加载。总图像数: {total_images}。批量大小: {batch_size}。工作进程数: {num_workers}。")

# --- 加载模型 ---
# 定义用于文件名的模型名称 (基于 model_dict 的键名)


print(f"加载分类器 ({classifier_model_name})...")
classifier = load_model_from_checkpoint(model_dict[classifier_model_name], classifier_path, num_classes=n_groups, device=device)
classifier.eval()

students = []
print(f"加载学生模型 ({student_model_name})...")
# 假设学生模型的文件夹编号从1开始，并且与n_groups数量一致
# 学生模型的索引将是 0 到 n_groups-1
for i in range(1, n_groups + 1):
    # 构建学生模型路径，假设 student_base_path 以 "_model_" 结尾
    # 并且内部的 .pth 文件名也包含数字 i
    # 例如: ..._model_1/resnet20_last_model_1.pth
    current_student_path = f'{student_base_path}{i}/resnet8x4_last_model_{i}.pth' # 注意: 'resnet20' 硬编码自原始路径
    print(f"加载学生模型 {i} (对应组索引 {i-1})...") # 组索引从0开始
    student = load_model_from_checkpoint(model_dict[student_model_name], current_student_path, num_classes=n_cls, device=device)
    student.eval()
    students.append(student)

print(f"加载教师模型 ({teacher_model_name})...")
teacher = load_model_from_checkpoint(model_dict[teacher_model_name], teacher_path, num_classes=n_cls, device=device)
teacher.eval()

# --- 预测函数（支持批量） ---
def predict_batch(data_batch, classifier_model, student_models, teacher_model, device_to_use, classes_per_group):
    """
    对一个批次的数据进行预测。
    返回最终预测、是否使用了学生模型、分类器预测的组、学生模型的原始预测、教师模型的原始预测。
    """
    data_batch = data_batch.to(device_to_use)
    with torch.no_grad():
        # 分类器预测组别 (原始输出 -> argmax)
        group_preds_raw_outputs = classifier_model(data_batch)
        # predicted_group_indices 对应 student_models 的索引 (0 到 n_groups-1)
        predicted_group_indices = torch.argmax(group_preds_raw_outputs, dim=1).cpu().numpy()

        # 教师模型对整个批次进行预测 (原始输出 -> argmax)
        teacher_outputs_raw = teacher_model(data_batch)
        teacher_pred_classes = torch.argmax(teacher_outputs_raw, dim=1).cpu().numpy()

        batch_final_predictions = []
        batch_used_student_flags = [] # 标记每张图片是否最终使用了学生模型的预测
        batch_student_raw_predictions = [] # 存储对应学生模型的原始预测（即使未使用）

        for i in range(data_batch.size(0)): # 遍历批次中的每张图片
            current_image_data = data_batch[i].unsqueeze(0) # (1, C, H, W)
            predicted_group_idx_for_image = predicted_group_indices[i]
            teacher_pred_for_image = teacher_pred_classes[i]

            # 选择对应的学生模型
            if 0 <= predicted_group_idx_for_image < len(student_models):
                selected_student = student_models[predicted_group_idx_for_image]
                student_output_raw_for_image = selected_student(current_image_data)
                student_pred_class_for_image = torch.argmax(student_output_raw_for_image, dim=1).item()
                batch_student_raw_predictions.append(student_pred_class_for_image)

                # 判断学生预测的类别是否在其专注的组内
                # 例如：如果 classes_per_group 是 10,
                # 学生模型0 (predicted_group_idx_for_image=0) 专注于类别 0-9,
                # 学生模型1 (predicted_group_idx_for_image=1) 专注于类别 10-19, 等等.
                # 所以, student_pred_class_for_image // classes_per_group 应等于 predicted_group_idx_for_image
                if student_pred_class_for_image // classes_per_group == predicted_group_idx_for_image:
                    final_prediction_for_image = student_pred_class_for_image
                    used_student_flag_for_image = True
                else:
                    # 如果学生预测的类别超出了其负责的范围，则使用教师模型的预测
                    final_prediction_for_image = teacher_pred_for_image
                    used_student_flag_for_image = False
            else:
                # Fallback: 如果预测的组索引无效 (理论上不应发生，如果n_groups设置正确)
                print(f"警告: 无效的组预测索引 {predicted_group_idx_for_image} (应为 0-{len(student_models)-1})。对图像 {i} 使用教师预测。")
                final_prediction_for_image = teacher_pred_for_image
                used_student_flag_for_image = False
                batch_student_raw_predictions.append(teacher_pred_for_image) # 或使用占位符如 -1

            batch_final_predictions.append(final_prediction_for_image)
            batch_used_student_flags.append(used_student_flag_for_image)

    return (np.array(batch_final_predictions),
            np.array(batch_used_student_flags),
            predicted_group_indices, # 分类器对批次中每个样本预测的组索引
            np.array(batch_student_raw_predictions), # 学生模型对每个样本的预测（在被选中时）
            teacher_pred_classes) # 教师模型对批次中每个样本的预测

# --- 处理测试集 ---
print("\n开始处理测试集...")
all_true_labels = []
all_final_predictions = []
all_used_student_flags = []
all_predicted_group_indices = []
all_student_raw_predictions = []
all_teacher_raw_predictions = [] # 存储教师模型对所有样本的预测，用于分析或作为回退
start_time = time.time()

# 计算每组的类别数量, 假设 n_cls 可以被 n_groups 整除
num_classes_per_group = n_cls // n_groups
if n_cls % n_groups != 0:
    print(f"警告: 总类别数 {n_cls} 不能被组数 {n_groups} 整除。组的类别划分可能不均匀，这会影响学生模型的“专注组内”判断逻辑。")


for batch_idx, (current_data_batch, current_target_batch) in enumerate(test_loader):
    (final_preds_b,
    used_students_flags_b,
    group_indices_b,
    student_raw_preds_b,
    teacher_raw_preds_b) = predict_batch(
        current_data_batch, classifier, students, teacher, device, num_classes_per_group
    )

    all_true_labels.extend(current_target_batch.numpy())
    all_final_predictions.extend(final_preds_b)
    all_used_student_flags.extend(used_students_flags_b)
    all_predicted_group_indices.extend(group_indices_b)
    all_student_raw_predictions.extend(student_raw_preds_b)
    all_teacher_raw_predictions.extend(teacher_raw_preds_b) # 保存每个样本的教师预测

    # 使用 current_data_batch.size(0) 获取当前批次的实际大小，以防最后一个批次不完整
    processed_count = (batch_idx + 1) * current_data_batch.size(0)
    # 更稳健的进度打印逻辑
    if (batch_idx + 1) % (max(1, len(test_loader) // 20)) == 0 or (batch_idx + 1) == len(test_loader): # 每处理5%或最后一个批次时打印
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
predicted_group_indices_np = np.array(all_predicted_group_indices) # 分类器预测的组索引
student_raw_predictions_np = np.array(all_student_raw_predictions) # 学生模型的原始预测（在其被选中时）
teacher_raw_predictions_np = np.array(all_teacher_raw_predictions) # 教师模型的原始预测（对所有样本）

# 确定每个标签的真实组别
# 例如, 如果 num_classes_per_group=10, 类别 0-9 属于组 0, 10-19 属于组 1, 等等。
true_group_indices_np = true_labels_np // num_classes_per_group

# 1. 分类器准确率: 检查分类器是否将图像分配给了正确的学生模型组。
#    即，分类器预测的组索引 (predicted_group_indices_np) 是否等于图像真实标签对应的组索引 (true_group_indices_np)。
classifier_task_assignment_accuracy = (predicted_group_indices_np == true_group_indices_np).mean() * 100 if len(predicted_group_indices_np) > 0 else 0

# 2. 使用学生模型决策的比例
student_model_decision_percentage = used_student_flags_np.mean() * 100 if len(used_student_flags_np) > 0 else 0

# 3. 联合模型的整体准确率 (最终输出的预测与真实标签的比较)
overall_system_accuracy = (final_predictions_np == true_labels_np).mean() * 100 if len(final_predictions_np) > 0 else 0

# 4. 当系统决策使用学生模型时，这些决策的准确率
#    只考虑 final_predictions_np 中那些由学生模型贡献的部分 (即 used_student_flags_np 为 True 的样本)
student_decision_indices = used_student_flags_np.astype(bool)
accuracy_when_student_model_used = 0
if student_decision_indices.sum() > 0:
    accuracy_when_student_model_used = (final_predictions_np[student_decision_indices] == true_labels_np[student_decision_indices]).mean() * 100

# 5. 当系统决策使用教师模型时 (即学生预测越界), 这些决策的准确率
#    只考虑 final_predictions_np 中那些由教师模型贡献的部分 (即 used_student_flags_np 为 False 的样本)
teacher_decision_indices = ~used_student_flags_np.astype(bool)
accuracy_when_teacher_model_used = 0
if teacher_decision_indices.sum() > 0:
    accuracy_when_teacher_model_used = (final_predictions_np[teacher_decision_indices] == true_labels_np[teacher_decision_indices]).mean() * 100

# --- 打印结果 ---
print("\n--- 联合系统评估结果 ---")
print(f"分类器 ({classifier_model_name}) 准确率 (任务分配给正确学生组): {classifier_task_assignment_accuracy:.2f}%")
print(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}%")
print(f"当使用学生模型决策时的准确率: {accuracy_when_student_model_used:.2f}%")
print(f"当使用教师模型决策时的准确率 (因学生预测越界): {accuracy_when_teacher_model_used:.2f}%")
print(f"联合模型整体准确率: {overall_system_accuracy:.2f}%")
print(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒")

# --- 保存结果到文件 ---
results_dir = 'evaluation_results'
os.makedirs(results_dir, exist_ok=True) # 如果目录不存在则创建

# 根据加载模型时定义的名称构建文件名
filename_c_part = classifier_model_name
filename_t_part = teacher_model_name
filename_s_part = student_model_name # 假设所有学生模型类型相同

results_filename = f"c{filename_c_part}_t{filename_t_part}_s{filename_s_part}.txt"
results_file_path = os.path.join(results_dir, results_filename)

with open(results_file_path, 'w', encoding='utf-8') as f:
    f.write(f"--- 联合系统评估结果 ---\n")
    f.write(f"分类器 ({classifier_model_name}) 准确率: {classifier_task_assignment_accuracy:.2f}%\n")
    f.write(f"  - 定义: 此准确率衡量分类器是否成功将图像（任务）分配给负责其真实类别所属范围的学生模型组（子模型）。\n")
    f.write(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}%\n")
    f.write(f"当使用学生模型决策时的准确率: {accuracy_when_student_model_used:.2f}%\n")
    f.write(f"当使用教师模型决策时的准确率 (因学生预测越界): {accuracy_when_teacher_model_used:.2f}%\n")
    f.write(f"联合模型整体准确率: {overall_system_accuracy:.2f}%\n")
    f.write(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒\n")
    f.write(f"\n--- 配置信息 ---\n")
    f.write(f"总类别数: {n_cls}, 组数: {n_groups}, 每组理论类别数: {num_classes_per_group}\n")
    f.write(f"分类器模型类型: {classifier_model_name}, 路径: {classifier_path}\n")
    f.write(f"学生模型类型: {student_model_name}, 基础路径: {student_base_path}\n")
    f.write(f"教师模型类型: {teacher_model_name}, 路径: {teacher_path}\n")
    f.write(f"测试数据批次大小: {batch_size}, 工作进程数: {num_workers}\n")
    f.write(f"运行设备: {device}\n")

print(f"结果已保存到: {results_file_path}")
print("评估完毕。")
