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
    if torch.cuda.is_available(): # Extra check to be safe
        print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
    else: # Should not happen if device is 'cuda' but good for robustness
        print("CUDA 声称可用，但 torch.cuda.is_available() 返回 False。切换到 CPU。")
        device = 'cpu'


# --- 模型路径 ---
classifier_path = 'save/classifier_models/classifier_resnet20_cifar100_lr_0.05_decay_0.0005_trial_1/resnet20_best.pth'
student_base_path = 'save/student_model/S_resnet20_T_resnet56_cifar100_kd_r_1.0_a_0.9_FixedT_4.0_trial_0.95,1,5,10,20_model_'
teacher_path = 'save/models/resnet56_vanilla/ckpt_epoch_240.pth'
n_cls = 100  # CIFAR-100 有 100 个类别
n_groups = 10  # 10 个组，每组 10 个类别

# --- 辅助函数：加载模型 ---
def load_model_from_checkpoint(model_cls, path, num_classes, device):
    """从检查点加载模型状态字典的辅助函数"""
    model = model_cls(num_classes=num_classes).to(device)  # 直接在目标设备上创建模型
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点文件未找到: {path}")
    # Added weights_only=True for security, if your .pth files are compatible
    # If not, revert to: checkpoint = torch.load(path, map_location=device)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"警告: 使用 weights_only=True 加载模型失败 ({e})。尝试不使用 weights_only。")
        print("这可能存在安全风险，如果模型来源不受信任。")
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
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 归一化
])
data_dir = './data'
try:
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
except Exception as e:
    print(f"加载数据集时出错，尝试不使用 download=True（请确保数据存在于 {data_dir}）：{e}")
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)

batch_size = 256
# Ensure num_workers is appropriate for your OS and setup
num_workers = 4 if device == 'cuda' and os.name != 'nt' and torch.cuda.is_available() else 0
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
total_images = len(test_loader.dataset)
print(f"测试数据集已加载。总图像数: {total_images}。批量大小: {batch_size}。")

# --- 加载模型 ---
# Define model names for filename based on the model_dict keys used
classifier_model_name = 'resnet20'
student_model_name = 'resnet20' # Assuming all student models are of the same type
teacher_model_name = 'resnet56'

# 加载分类器
print(f"加载分类器 ({classifier_model_name})...")
classifier = load_model_from_checkpoint(model_dict[classifier_model_name], classifier_path, num_classes=n_groups, device=device)
classifier.eval()

# 加载学生模型（10 个）
students = []
print(f"加载学生模型 ({student_model_name})...")
for i in range(1, 11):
    student_path = f'{student_base_path}{i}/resnet20_last_model_{i}.pth' # Note: resnet20 is hardcoded here from original path
    print(f"加载学生模型 {i}...")
    student = load_model_from_checkpoint(model_dict[student_model_name], student_path, num_classes=n_cls, device=device)
    student.eval()
    students.append(student)

# 加载教师模型
print(f"加载教师模型 ({teacher_model_name})...")
teacher = load_model_from_checkpoint(model_dict[teacher_model_name], teacher_path, num_classes=n_cls, device=device)
teacher.eval()

# --- 预测函数（支持批量） ---
def predict_batch(data, classifier, students, teacher, device):
    data = data.to(device)
    with torch.no_grad():
        # 分类器预测组别
        group_preds = torch.argmax(classifier(data), dim=1).cpu().numpy()
        # 教师模型预测
        teacher_preds = torch.argmax(teacher(data), dim=1).cpu().numpy()

        final_preds = []
        used_students = []
        student_preds_list = []

        for i, group_pred in enumerate(group_preds):
            # 选择对应的学生模型
            # Ensure group_pred is a valid index for students list
            if 0 <= group_pred < len(students):
                student = students[group_pred]
                student_output = student(data[i].unsqueeze(0)) # Process one image at a time
                student_pred = torch.argmax(student_output, dim=1).item()
                student_preds_list.append(student_pred)

                # 判断学生预测是否在其专注组内
                if student_pred // 10 == group_pred: # Assuming n_groups = 10 means 10 classes per group
                    final_pred = student_pred
                    used_student = True
                else:
                    final_pred = teacher_preds[i]
                    used_student = False
            else:
                # Fallback if group_pred is out of bounds (should not happen with correct n_groups)
                print(f"警告: 无效的组预测 {group_pred}。对图像 {i} 使用教师预测。")
                final_pred = teacher_preds[i]
                used_student = False
                student_preds_list.append(teacher_preds[i]) # Or some placeholder like -1

            final_preds.append(final_pred)
            used_students.append(used_student)

    return np.array(final_preds), np.array(used_students), group_preds, np.array(student_preds_list), teacher_preds

# --- 处理测试集 ---
print("\n开始处理测试集...")
true_labels = []
predictions = []
used_students_list = []
predicted_groups_list = []
student_preds_list = []
teacher_preds_list = []
start_time = time.time()

for batch_idx, (data, target) in enumerate(test_loader):
    final_preds_batch, used_students_batch, group_preds_batch, student_preds_batch, teacher_preds_batch = predict_batch(
        data, classifier, students, teacher, device
    )
    true_labels.extend(target.numpy())
    predictions.extend(final_preds_batch)
    used_students_list.extend(used_students_batch)
    predicted_groups_list.extend(group_preds_batch)
    student_preds_list.extend(student_preds_batch)
    teacher_preds_list.extend(teacher_preds_batch)

    processed_count = (batch_idx + 1) * data.size(0) # Use data.size(0) for actual batch size
    # More robust progress printing logic
    if (batch_idx + 1) % (max(1, len(test_loader) // 10)) == 0 or (batch_idx + 1) == len(test_loader):
        current_time = time.time()
        elapsed = current_time - start_time
        progress_percentage = (processed_count / total_images) * 100
        print(
            f"  已处理 {processed_count}/{total_images} 张图像 ({progress_percentage:.1f}%) | 时间: {elapsed:.1f}秒")

end_time = time.time()
total_time = end_time - start_time
avg_time_per_image = total_time / total_images if total_images > 0 else 0
print(f"测试集处理完成。耗时: {total_time:.2f}秒")
print(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒")

# --- 计算指标 ---
true_labels = np.array(true_labels)
predictions = np.array(predictions)
used_students_arr = np.array(used_students_list) # Renamed to avoid conflict
predicted_groups_arr = np.array(predicted_groups_list) # Renamed
student_preds_arr = np.array(student_preds_list) # Renamed
teacher_preds_arr = np.array(teacher_preds_list) # Renamed
true_groups = true_labels // 10 # Assuming 10 classes per group

# 分类器准确率
classifier_accuracy = (predicted_groups_arr == true_groups).mean() * 100 if len(predicted_groups_arr) > 0 else 0

# 使用学生模型的比例
student_used_percentage = used_students_arr.mean() * 100 if len(used_students_arr) > 0 else 0

# 整体准确率
overall_accuracy = (predictions == true_labels).mean() * 100 if len(predictions) > 0 else 0

# 使用学生模型时的准确率
student_used_indices = used_students_arr.astype(bool)
student_accuracy = 0
if student_used_indices.sum() > 0:
    student_accuracy = (student_preds_arr[student_used_indices] == true_labels[student_used_indices]).mean() * 100

# 使用教师模型时的准确率
teacher_used_indices = ~used_students_arr.astype(bool)
teacher_accuracy = 0
if teacher_used_indices.sum() > 0:
    teacher_accuracy = (teacher_preds_arr[teacher_used_indices] == true_labels[teacher_used_indices]).mean() * 100

# --- 打印结果 ---
print("\n--- 联合系统评估结果 ---")
print(f"分类器 ({classifier_model_name}) 准确率: {classifier_accuracy:.2f}%")
print(f"使用学生模型 ({student_model_name}) 的比例: {student_used_percentage:.2f}%")
print(f"使用学生模型时的准确率: {student_accuracy:.2f}%")
print(f"使用教师模型 ({teacher_model_name}) 时的准确率: {teacher_accuracy:.2f}%")
print(f"联合模型整体准确率: {overall_accuracy:.2f}%")
print(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒")

# --- 保存结果到文件 ---
# Define results directory and filename
results_dir = 'evaluation_results'
os.makedirs(results_dir, exist_ok=True) # Create directory if it doesn't exist

# Construct the filename
# These names are based on the model_dict keys used for loading each component
filename_c_part = classifier_model_name
filename_t_part = teacher_model_name
filename_s_part = student_model_name # Assuming all student models are of this type

results_filename = f"c{filename_c_part}_t{filename_t_part}_s{filename_s_part}.txt"
results_file_path = os.path.join(results_dir, results_filename)

with open(results_file_path, 'w') as f:
    f.write(f"分类器 ({classifier_model_name}) 准确率: {classifier_accuracy:.2f}%\n")
    f.write(f"使用学生模型 ({student_model_name}) 的比例: {student_used_percentage:.2f}%\n")
    f.write(f"使用学生模型时的准确率: {student_accuracy:.2f}%\n")
    f.write(f"使用教师模型 ({teacher_model_name}) 时的准确率: {teacher_accuracy:.2f}%\n")
    f.write(f"联合模型整体准确率: {overall_accuracy:.2f}%\n")
    f.write(f"每张图像的平均处理时间: {avg_time_per_image:.6f} 秒\n")
    f.write(f"\n--- 模型配置 ---\n")
    f.write(f"分类器模型: {classifier_model_name}\n")
    f.write(f"分类器路径: {classifier_path}\n")
    f.write(f"学生模型: {student_model_name}\n")
    f.write(f"学生模型基础路径: {student_base_path}\n")
    f.write(f"教师模型: {teacher_model_name}\n")
    f.write(f"教师路径: {teacher_path}\n")

print(f"结果已保存到: {results_file_path}")

print("评估完毕。")
