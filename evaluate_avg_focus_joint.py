# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import model_dict  # 请确保此导入在您的环境中有效

# --- 设备检查 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")
if device == 'cuda':
    print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")
    cudnn.benchmark = True
# --- 设备检查结束 ---

# --- 模型路径模板和教师模型路径 ---
base_path = './save/student_model/S_wrn_40_1_T_wrn_40_2_cifar100_kd_r_1.0_a_0.9_FixedT_4.0_trial_0.95,1,5,10,20_model_{}/wrn_40_1_last_model_{}.pth'
teacher_model_path = './save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth'  # 假设的教师模型路径
n_cls = 100  # CIFAR-100 有 100 个类别

# --- 模型路径结束 ---

# --- 辅助函数：加载模型 ---
def load_model_from_checkpoint(model_cls, path, device):
    """从检查点加载模型状态字典的辅助函数"""
    model = model_cls(num_classes=n_cls)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点文件未找到: {path}")
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
num_workers = 4 if device == 'cuda' and os.name != 'nt' else 0
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
total_images = len(test_loader.dataset)
print(f"测试数据集已加载。总图像数: {total_images}。批量大小: {batch_size}。")

# --- 数据加载结束 ---

# --- 评估函数 ---
def evaluate_model(model_index, specialist_student_model_path, teacher_model_path, device, test_loader):
    # 加载专家学生模型
    print(f"\n加载专家学生模型 {model_index}...")
    specialist_student_model = load_model_from_checkpoint(model_dict['wrn_40_1'], specialist_student_model_path, device)

    # 加载教师模型
    print("加载教师模型...")
    teacher_model = load_model_from_checkpoint(model_dict['wrn_40_2'], teacher_model_path, device)

    # 设置评估模式并移动到设备
    specialist_student_model.eval().to(device)
    teacher_model.eval().to(device)
    print("模型已设置为评估模式并移动到设备。")

    # 确定专注类别
    focused_classes = list(range((model_index - 1) * 10, model_index * 10))
    focused_classes_set = set(focused_classes)
    print(f"专家学生模型 {model_index} 关注的类别: {focused_classes}")

    # 初始化准确率计数器
    joint_correct = 0
    joint_total = 0
    specialist_focused_correct = 0
    specialist_focused_total = 0
    teacher_correct = 0
    teacher_total = 0
    print("准确率计数器已初始化。")

    # 评估循环
    print("\n开始评估循环...")
    processed_count = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            specialist_output = specialist_student_model(data)
            specialist_pred = specialist_output.argmax(dim=1)
            teacher_output = teacher_model(data)
            teacher_pred = teacher_output.argmax(dim=1)

            for i in range(data.size(0)):
                target_item = target[i].item()
                specialist_pred_item = specialist_pred[i].item()
                teacher_pred_item = teacher_pred[i].item()

                # 联合模型预测逻辑
                if specialist_pred_item in focused_classes_set:
                    final_pred = specialist_pred_item
                else:
                    final_pred = teacher_pred_item
                joint_total += 1
                if final_pred == target_item:
                    joint_correct += 1

                # 专才学生在专注类别上的准确率
                if target_item in focused_classes_set:
                    specialist_focused_total += 1
                    if specialist_pred_item == target_item:
                        specialist_focused_correct += 1

                # 教师模型在总体类别上的准确率
                teacher_total += 1
                if teacher_pred_item == target_item:
                    teacher_correct += 1

            processed_count += data.size(0)
            if (batch_idx + 1) % (len(test_loader) // 10 + 1) == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                print(
                    f"  已处理 {processed_count}/{total_images} 张图像 ({processed_count / total_images * 100:.1f}%) | 时间: {elapsed:.1f}秒")

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    print(f"评估循环完成。耗时: {total_time:.2f}秒")
    print(f"每张图像的平均评估时间: {avg_time_per_image:.6f} 秒")

    # 计算最终准确率
    joint_accuracy = (100. * joint_correct / joint_total) if joint_total > 0 else 0
    specialist_focused_accuracy = (
                100. * specialist_focused_correct / specialist_focused_total) if specialist_focused_total > 0 else 0
    teacher_accuracy = (100. * teacher_correct / teacher_total) if teacher_total > 0 else 0

    # 打印结果
    print("\n--- 最终评估结果 ---")
    print(f"联合模型整体准确率: {joint_accuracy:.2f}% ({joint_correct}/{joint_total})")
    print(
        f"专才学生模型在专注类别上的准确率 (类别 {focused_classes}): {specialist_focused_accuracy:.2f}% ({specialist_focused_correct}/{specialist_focused_total})")
    print(f"教师模型在总体类别上的准确率: {teacher_accuracy:.2f}% ({teacher_correct}/{teacher_total})")
    print(f"每张图像的平均评估时间: {avg_time_per_image:.6f} 秒")

    # 保存结果到TXT文件
    student_model_dir = os.path.dirname(specialist_student_model_path)
    results_file = os.path.join(student_model_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"专才学生模型在专注类别上的准确率: {specialist_focused_accuracy:.2f}%\n")
        f.write(f"教师模型在总体类别上的准确率: {teacher_accuracy:.2f}%\n")
        f.write(f"学生-教师联合模型在总体类别上的准确率: {joint_accuracy:.2f}%\n")
        f.write(f"每张图像的平均评估时间: {avg_time_per_image:.6f} 秒\n")
    print(f"结果已保存到: {results_file}")

# --- 主循环：从 model_1 到 model_10 ---
for model_index in range(1, 11):  # 评估 model_1 到 model_10
    specialist_student_model_path = base_path.format(model_index, model_index)
    if not os.path.isfile(specialist_student_model_path):
        print(f"警告: 模型文件未找到: {specialist_student_model_path}")
        continue
    evaluate_model(model_index, specialist_student_model_path, teacher_model_path, device, test_loader)

print("所有模型评估完毕。")
