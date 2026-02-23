# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import model_dict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json  # For saving structured results


# --- 新增：可靠性评估相关 ---
def get_ece(preds, targets, n_bins=15):
    """Calculates Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(preds, 1)
    accuracies = predictions.eq(targets)

    ece = torch.zeros(1, device=preds.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def reliability_diagram(confidences, accuracies, n_bins=15, ax=None):
    """Plots a reliability diagram"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    avg_conf_in_bin = []
    acc_in_bin = []
    bin_counts = []

    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin_mask = (confidences > bl) & (confidences <= bu)
        bin_counts.append(np.sum(in_bin_mask))
        if np.sum(in_bin_mask) > 0:
            avg_conf_in_bin.append(np.mean(confidences[in_bin_mask]))
            acc_in_bin.append(np.mean(accuracies[in_bin_mask]))
        else:
            avg_conf_in_bin.append(None)  # Or some placeholder like (bl+bu)/2
            acc_in_bin.append(None)  # Or 0

    # Plotting
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')

    # Bar for proportion of samples in each bin (optional, can make plot busy)
    # for i, count in enumerate(bin_counts):
    #     ax.bar((bin_lowers[i] + bin_uppers[i])/2, acc_in_bin[i] if acc_in_bin[i] is not None else 0,
    #            width=(bin_uppers[i]-bin_lowers[i])*0.9, color='lightskyblue', alpha=0.5, edgecolor='black')

    # Line plot for calibration
    valid_bins_x = [x for x in avg_conf_in_bin if x is not None]
    valid_bins_y = [y for x, y in zip(avg_conf_in_bin, acc_in_bin) if x is not None]  # y corresponding to valid x

    if valid_bins_x:  # only plot if there are valid bins
        ax.plot(valid_bins_x, valid_bins_y, marker='o', linestyle='-', color='blue', label='Model calibration')

    # Add gaps indication
    # for i in range(len(avg_conf_in_bin)):
    #     if avg_conf_in_bin[i] is not None and acc_in_bin[i] is not None:
    #         ax.errorbar(avg_conf_in_bin[i], acc_in_bin[i],
    #                     yerr=np.abs(avg_conf_in_bin[i] - acc_in_bin[i]),
    #                     fmt='none', ecolor='red', capsize=5, alpha=0.5)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.7)
    return ax


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

# --- 模型路径 (请确保这些路径正确) ---
# 注意：这里的 classifier_path 应该是指训练好的10分类的组分类器
classifier_path = 'save/classifier_models/classifier_resnet8x4_cifar100_opt_adamw_lr_0.001_wd_0.0001_aug_适中_ls_0.1_trial_3/ckpt_epoch_240.pth'
student_base_path = 'save/student_model/S_resnet8x4_T_resnet32x4_cifar100_kd_r_1.0_a_0.9_FixedT_4.0_trial_0.95,1,5,10,20_model_'
teacher_path = 'save/models/resnet32x4_vanilla/ckpt_epoch_240.pth'

classifier_model_name = 'resnet8x4'  # 组分类器模型名
student_model_name = 'resnet8x4'  # 学生模型名
teacher_model_name = 'resnet32x4'  # 教师模型名
n_cls = 100
n_groups = 10


def load_model_from_checkpoint(model_cls_fn, path, num_classes, device_to_load_on):
    """从检查点加载模型状态字典的辅助函数"""
    model = model_cls_fn(num_classes=num_classes).to(device_to_load_on)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点文件未找到: {path}")

    print(f"正在从 {path} 加载模型...")
    try:
        # PyTorch 1.6+ supports weights_only for security.
        checkpoint = torch.load(path, map_location=device_to_load_on, weights_only=True)
    except RuntimeError:  # Fallback for older PyTorch or if it's not a 'weights_only' checkpoint
        print(f"警告: 使用 weights_only=True 加载模型 '{os.path.basename(path)}' 失败。尝试不使用 weights_only。")
        print("如果模型来源不受信任，这可能存在安全风险。")
        checkpoint = torch.load(path, map_location=device_to_load_on)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # 假设检查点本身就是 state_dict

    # 处理可能的 DataParallel 包装:
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("检测到模型是 DataParallel 包装的，正在移除 'module.' 前缀...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

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

batch_size_eval = 128
num_workers_eval = 4 if device == 'cuda' and os.name != 'nt' else 0
test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers_eval,
                         pin_memory=True)
total_images = len(test_loader.dataset)
print(f"测试数据集已加载。总图像数: {total_images}。批量大小: {batch_size_eval}。工作进程数: {num_workers_eval}。")

# --- 加载模型 ---
# 'original_classifier' 实际上是组分类器 (输出10个组的logits)
# 当前的 predict_batch_confidence_routing 并未使用这个组分类器。
# 如果需要使用它来预先判断组，则需要修改路由逻辑。
group_classifier = None
# print(f"加载原始组分类器 ({classifier_model_name})...")
# try:
#     group_classifier = load_model_from_checkpoint(model_dict[classifier_model_name], classifier_path,
#                                                  num_classes=n_groups, device_to_load_on=device)
#     group_classifier.eval()
# except FileNotFoundError:
#     print(f"警告: 组分类器路径 {classifier_path} 未找到。它将不会被用于路由。")
#     group_classifier = None


students = []
print(f"加载学生模型 ({student_model_name})...")
for i in range(1, n_groups + 1):
    current_student_path = f'{student_base_path}{i}/{student_model_name}_last_model_{i}.pth'
    print(f"加载学生模型 {i} (对应组索引 {i - 1})...")
    try:
        student = load_model_from_checkpoint(model_dict[student_model_name], current_student_path, num_classes=n_cls,
                                             device_to_load_on=device)
        student.eval()
        students.append(student)
    except FileNotFoundError:
        print(f"错误: 学生模型 {current_student_path} 未找到！评估无法继续。")
        exit()

if len(students) != n_groups:
    print(f"错误: 期望加载 {n_groups} 个学生模型，但实际加载了 {len(students)} 个。请检查路径。")
    exit()

print(f"加载教师模型 ({teacher_model_name})...")
try:
    teacher = load_model_from_checkpoint(model_dict[teacher_model_name], teacher_path, num_classes=n_cls,
                                         device_to_load_on=device)
    teacher.eval()
except FileNotFoundError:
    print(f"错误: 教师模型 {teacher_path} 未找到！评估无法继续。")
    exit()


# --- 预测函数 (Confidence-Based Routing) ---
# 此函数与您提供的版本基本一致，主要关注其输出的收集和后续分析
def predict_batch_confidence_routing(data_batch, student_models_list, teacher_model_instance, device_to_use,
                                     num_total_classes, num_model_groups, classes_p_group):
    data_batch = data_batch.to(device_to_use)
    batch_size_current = data_batch.size(0)

    all_final_predictions = []
    all_used_student_flags = []
    all_predicted_group_indices_confidence = []
    all_student_raw_predictions = []
    all_teacher_raw_predictions = []

    # --- 新增：收集所有模型的原始 logits/probs 用于后续的 ECE 计算 ---
    all_final_probs_for_ece = []  # 存储最终选择的模型的概率输出

    with torch.no_grad():
        all_student_logits_batch = []
        all_student_probs_batch = []  # 存储每个学生模型对所有100类的概率
        for student_model_item in student_models_list:
            student_logits = student_model_item(data_batch)
            all_student_logits_batch.append(student_logits)
            all_student_probs_batch.append(torch.softmax(student_logits, dim=1))

        group_confidence_scores_for_batch = torch.zeros(batch_size_current, num_model_groups, device=device_to_use)
        for group_idx in range(num_model_groups):
            current_student_probs = all_student_probs_batch[group_idx]
            start_class_idx = group_idx * classes_p_group
            end_class_idx = min(start_class_idx + classes_p_group, num_total_classes)
            probs_for_responsible_classes = current_student_probs[:, start_class_idx:end_class_idx]
            if probs_for_responsible_classes.numel() > 0:
                confidence_score_for_this_group, _ = torch.max(probs_for_responsible_classes, dim=1)
            else:
                confidence_score_for_this_group = torch.zeros(batch_size_current, device=device_to_use)
            group_confidence_scores_for_batch[:, group_idx] = confidence_score_for_this_group

        predicted_group_indices_confidence_batch = torch.argmax(group_confidence_scores_for_batch, dim=1)

        teacher_outputs_raw = teacher_model_instance(data_batch)
        teacher_probs_batch = torch.softmax(teacher_outputs_raw, dim=1)
        teacher_pred_classes_batch = torch.argmax(teacher_outputs_raw, dim=1)

        for i in range(batch_size_current):
            predicted_group_idx_for_image = predicted_group_indices_confidence_batch[i].item()
            teacher_pred_for_image = teacher_pred_classes_batch[i].item()

            selected_student_all_class_logits = all_student_logits_batch[predicted_group_idx_for_image][i, :]
            selected_student_probs_for_image = all_student_probs_batch[predicted_group_idx_for_image][i, :]  # 对应学生的全部概率
            student_pred_class_for_image = torch.argmax(selected_student_all_class_logits).item()

            all_student_raw_predictions.append(student_pred_class_for_image)

            if student_pred_class_for_image // classes_p_group == predicted_group_idx_for_image:
                all_final_predictions.append(student_pred_class_for_image)
                all_used_student_flags.append(True)
                all_final_probs_for_ece.append(selected_student_probs_for_image.unsqueeze(0))  # 保存选择的模型的概率
            else:
                all_final_predictions.append(teacher_pred_for_image)
                all_used_student_flags.append(False)
                all_final_probs_for_ece.append(teacher_probs_batch[i, :].unsqueeze(0))  # 保存选择的模型的概率

            all_predicted_group_indices_confidence.append(predicted_group_idx_for_image)
            all_teacher_raw_predictions.append(teacher_pred_for_image)

    final_probs_tensor_for_ece = torch.cat(all_final_probs_for_ece, dim=0) if all_final_probs_for_ece else torch.empty(
        0)

    return (np.array(all_final_predictions),
            np.array(all_used_student_flags),
            np.array(all_predicted_group_indices_confidence),
            np.array(all_student_raw_predictions),
            np.array(all_teacher_raw_predictions),
            final_probs_tensor_for_ece)  # 新增返回项


# --- 处理测试集 ---
print("\n开始处理测试集 (使用新的基于学生最大置信度的路由机制)...")
all_true_labels_list = []
all_final_predictions_list = []
all_used_student_flags_list = []
all_predicted_group_indices_by_confidence_list = []
all_student_raw_predictions_list = []
all_teacher_raw_predictions_list = []
all_final_probs_for_ece_list = []  # 收集所有批次的最终概率

start_time = time.time()
num_classes_per_group = n_cls // n_groups
if n_cls % n_groups != 0:
    print(f"警告: 总类别数 {n_cls} 不能被组数 {n_groups} 整除。这会影响类别范围计算。")

for batch_idx, (current_data_batch, current_target_batch) in enumerate(test_loader):
    (final_preds_b, used_students_flags_b, group_indices_confidence_b,
     student_raw_preds_b, teacher_raw_preds_b, final_probs_b) = predict_batch_confidence_routing(
        current_data_batch, students, teacher, device, n_cls, n_groups, num_classes_per_group
    )

    all_true_labels_list.extend(current_target_batch.numpy())
    all_final_predictions_list.extend(final_preds_b)
    all_used_student_flags_list.extend(used_students_flags_b)
    all_predicted_group_indices_by_confidence_list.extend(group_indices_confidence_b)
    all_student_raw_predictions_list.extend(student_raw_preds_b)
    all_teacher_raw_predictions_list.extend(teacher_raw_preds_b)
    if final_probs_b.numel() > 0:  # 确保不为空
        all_final_probs_for_ece_list.append(final_probs_b.cpu())

    processed_count = (batch_idx * batch_size_eval) + current_data_batch.size(0)
    if (batch_idx + 1) % (max(1, len(test_loader) // 20)) == 0 or (batch_idx + 1) == len(test_loader):
        current_time_iter = time.time()
        elapsed = current_time_iter - start_time
        progress_percentage = (processed_count / total_images) * 100
        print(
            f"  已处理 {processed_count}/{total_images} 张图像 ({progress_percentage:.1f}%) | 已用时间: {elapsed:.1f}秒")

end_time = time.time()
total_processing_time = end_time - start_time
avg_time_per_image_ms = (total_processing_time / total_images * 1000) if total_images > 0 else 0
print(f"测试集处理完成。总耗时: {total_processing_time:.2f}秒")
print(f"每张图像的平均处理时间: {avg_time_per_image_ms:.3f} 毫秒")

# --- 转换列表为Numpy数组或Torch张量 ---
true_labels_np = np.array(all_true_labels_list)
final_predictions_np = np.array(all_final_predictions_list)
used_student_flags_np = np.array(all_used_student_flags_list)
predicted_group_indices_confidence_np = np.array(all_predicted_group_indices_by_confidence_list)
student_raw_predictions_np = np.array(all_student_raw_predictions_list)
teacher_raw_predictions_np = np.array(all_teacher_raw_predictions_list)

# 合并所有批次的概率张量
if all_final_probs_for_ece_list:
    all_final_probs_tensor = torch.cat(all_final_probs_for_ece_list, dim=0)
else:
    all_final_probs_tensor = torch.empty((0, n_cls))  # 如果没有数据，则为空张量

true_group_indices_np = true_labels_np // num_classes_per_group

# --- 计算指标 ---
# 1. "任务分配"准确率
confidence_routing_accuracy = (predicted_group_indices_confidence_np == true_group_indices_np).mean() * 100 if len(
    predicted_group_indices_confidence_np) > 0 else 0
# 2. 使用学生模型决策的比例
student_model_decision_percentage = used_student_flags_np.mean() * 100 if len(used_student_flags_np) > 0 else 0
# 3. 联合模型的整体 Top-1 准确率
overall_system_accuracy_top1 = (final_predictions_np == true_labels_np).mean() * 100 if len(
    final_predictions_np) > 0 else 0

# --- 新增：计算 Top-5 准确率 ---
# `all_final_probs_tensor` 包含了最终决策模型的softmax输出
if all_final_probs_tensor.numel() > 0:
    _, top5_preds = torch.topk(all_final_probs_tensor, 5, dim=1)
    correct_top5 = top5_preds.eq(torch.from_numpy(true_labels_np).view(-1, 1).expand_as(top5_preds))
    overall_system_accuracy_top5 = correct_top5.any(dim=1).float().mean().item() * 100
else:
    overall_system_accuracy_top5 = 0.0

# 4. 当系统决策使用学生模型时，这些决策的准确率
student_decision_indices = used_student_flags_np.astype(bool)
accuracy_when_student_model_used = 0
if student_decision_indices.sum() > 0:
    accuracy_when_student_model_used = (final_predictions_np[student_decision_indices] == true_labels_np[
        student_decision_indices]).mean() * 100
# 5. 当系统决策使用教师模型时, 这些决策的准确率
teacher_decision_indices = ~used_student_flags_np.astype(bool)
accuracy_when_teacher_model_used = 0
if teacher_decision_indices.sum() > 0:
    accuracy_when_teacher_model_used = (final_predictions_np[teacher_decision_indices] == true_labels_np[
        teacher_decision_indices]).mean() * 100

# --- 新增：计算 ECE ---
ece_score = 0.0
if all_final_probs_tensor.numel() > 0 and len(true_labels_np) > 0:
    ece_score = get_ece(all_final_probs_tensor.cpu(), torch.from_numpy(true_labels_np).cpu())

# --- 打印结果 ---
print("\n--- 联合系统评估结果 (使用基于学生最大置信度的路由) ---")
print(f"最大置信度路由准确率 (任务分配给正确学生组): {confidence_routing_accuracy:.2f}%")
print(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}%")
print(f"  当使用学生模型决策时的准确率: {accuracy_when_student_model_used:.2f}%")
print(f"  当使用教师模型决策时的准确率 (因学生预测越界): {accuracy_when_teacher_model_used:.2f}%")
print(f"联合模型整体 Top-1 准确率: {overall_system_accuracy_top1:.2f}%")
print(f"联合模型整体 Top-5 准确率: {overall_system_accuracy_top5:.2f}%")  # 新增
print(f"联合模型期望校准误差 (ECE): {ece_score:.4f}")  # 新增
print(f"每张图像的平均处理时间: {avg_time_per_image_ms:.3f} 毫秒")

# --- 保存结果 ---
results_dir = 'evaluation_results_improved'  # 使用新的目录名
os.makedirs(results_dir, exist_ok=True)

run_identifier = f"ConfRouted_T_{teacher_model_name}_S_{student_model_name}"
timestamp = time.strftime("%Y%m%d-%H%M%S")
results_filename_base = f"{run_identifier}_{timestamp}"

# 混淆矩阵
cm_path = os.path.join(results_dir, f"{results_filename_base}_confusion_matrix.png")
if len(true_labels_np) > 0 and len(final_predictions_np) > 0:
    cm = confusion_matrix(true_labels_np, final_predictions_np, labels=list(range(n_cls)))
    plt.figure(figsize=(24, 20) if n_cls > 20 else (10, 8))  # 根据类别数调整
    sns.heatmap(cm, annot=(n_cls <= 20), fmt='d', cmap='Blues', cbar=True)  # 类别少时显示数字
    plt.title(f'整体系统混淆矩阵\nTop-1 Acc: {overall_system_accuracy_top1:.2f}%', fontsize=16)
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    try:
        plt.savefig(cm_path)
        print(f"混淆矩阵图已保存到: {cm_path}")
    except Exception as e:
        print(f"无法保存混淆矩阵图: {e}")
    plt.close()

# --- 新增：可靠性图 ---
reliability_diagram_path = os.path.join(results_dir, f"{results_filename_base}_reliability_diagram.png")
if all_final_probs_tensor.numel() > 0 and len(true_labels_np) > 0:
    confidences_rd, predictions_rd = torch.max(all_final_probs_tensor.cpu(), 1)
    accuracies_rd = predictions_rd.eq(torch.from_numpy(true_labels_np).cpu())
    fig_rd, ax_rd = plt.subplots(figsize=(7, 7))
    reliability_diagram(confidences_rd.numpy(), accuracies_rd.numpy(), ax=ax_rd)
    ax_rd.set_title(f'Reliability Diagram (ECE: {ece_score:.4f})')
    try:
        fig_rd.savefig(reliability_diagram_path)
        print(f"可靠性图已保存到: {reliability_diagram_path}")
    except Exception as e:
        print(f"无法保存可靠性图: {e}")
    plt.close(fig_rd)

# 分类报告
report_str = "没有足够的样本进行分类报告。"
if len(true_labels_np) > 0 and len(final_predictions_np) > 0:
    report_str = classification_report(
        true_labels_np,
        final_predictions_np,
        labels=list(range(n_cls)),
        target_names=[f'Class_{i}' for i in range(n_cls)],
        zero_division=0
    )
print("\n整体系统分类报告:")
print(report_str)

# 保存文本和JSON结果
results_summary_path = os.path.join(results_dir, f"{results_filename_base}_summary.txt")
results_json_path = os.path.join(results_dir, f"{results_filename_base}_summary.json")

summary_data = {
    "run_identifier": run_identifier,
    "timestamp": timestamp,
    "config": {
        "routing_method": "Max Confidence in Responsible Group",
        "total_classes": n_cls,
        "num_groups": n_groups,
        "classes_per_group_ideal": num_classes_per_group,
        "group_classifier_model": classifier_model_name if group_classifier else "Not Used/Loaded",
        "group_classifier_path": classifier_path if group_classifier else "N/A",
        "student_model_type": student_model_name,
        "student_base_path": student_base_path,
        "teacher_model_type": teacher_model_name,
        "teacher_path": teacher_path,
        "eval_batch_size": batch_size_eval,
        "eval_num_workers": num_workers_eval,
        "device": device,
    },
    "metrics": {
        "confidence_routing_accuracy_percent": confidence_routing_accuracy,
        "student_model_decision_percentage": student_model_decision_percentage,
        "accuracy_when_student_model_used_percent": accuracy_when_student_model_used,
        "accuracy_when_teacher_model_used_percent": accuracy_when_teacher_model_used,
        "overall_system_accuracy_top1_percent": overall_system_accuracy_top1,
        "overall_system_accuracy_top5_percent": overall_system_accuracy_top5,
        "expected_calibration_error": ece_score,
        "avg_processing_time_ms_per_image": avg_time_per_image_ms,
    },
    "classification_report": report_str
}

with open(results_summary_path, 'w', encoding='utf-8') as f:
    f.write(f"--- 联合系统评估结果 (使用基于学生最大置信度的路由) ---\n")
    f.write(f"运行标识: {run_identifier}\n")
    f.write(f"时间戳: {timestamp}\n\n")

    f.write(f"最大置信度路由准确率 (任务分配给正确学生组): {confidence_routing_accuracy:.2f}%\n")
    f.write(
        f"  - 定义: 此准确率衡量基于“学生对其负责类别内的最大预测概率”的路由方法是否成功将图像分配给负责其真实类别所属范围的学生模型组。\n")
    f.write(f"使用学生模型 ({student_model_name}) 决策的比例: {student_model_decision_percentage:.2f}%\n")
    f.write(f"  当使用学生模型决策时的准确率: {accuracy_when_student_model_used:.2f}%\n")
    f.write(f"  当使用教师模型决策时的准确率 (因学生预测越界): {accuracy_when_teacher_model_used:.2f}%\n")
    f.write(f"联合模型整体 Top-1 准确率: {overall_system_accuracy_top1:.2f}%\n")
    f.write(f"联合模型整体 Top-5 准确率: {overall_system_accuracy_top5:.2f}%\n")
    f.write(f"联合模型期望校准误差 (ECE): {ece_score:.4f}\n")
    f.write(f"每张图像的平均处理时间: {avg_time_per_image_ms:.3f} 毫秒\n\n")

    f.write(f"--- 配置信息 ---\n")
    json.dump(summary_data["config"], f, indent=4, ensure_ascii=False)
    f.write("\n\n")

    f.write(f"--- 整体系统分类报告 ---\n")
    f.write(report_str)

with open(results_json_path, 'w', encoding='utf-8') as f_json:
    json.dump(summary_data, f_json, indent=4, ensure_ascii=False)

print(f"详细结果和分类报告已保存到: {results_summary_path}")
print(f"JSON 格式的总结已保存到: {results_json_path}")
print("评估完毕。")

