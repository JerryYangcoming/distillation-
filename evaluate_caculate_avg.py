import os
import re

# 定义文件路径模板
base_path_template = 'save/student_model/S_wrn_40_1_T_wrn_40_2_cifar100_kd_r_1.0_a_0.9_FixedT_4.0_trial_0.95,1,5,10,20_model_{}/evaluation_results.txt'

# 初始化累加器和计数器
specialist_accuracy_sum = 0.0  # 专才学生模型准确率总和
joint_accuracy_sum = 0.0  # 学生-教师联合模型准确率总和
valid_file_count = 0  # 有效文件计数

# 循环处理 model_1 到 model_10
for model_index in range(1, 11):
    # 生成文件路径
    file_path = base_path_template.format(model_index)

    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"警告: 文件未找到: {file_path}")
        continue

    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取“专才学生模型在专注类别上的准确率”
        specialist_match = re.search(r'专才学生模型在专注类别上的准确率: ([\d.]+)%', content)
        if specialist_match:
            specialist_accuracy = float(specialist_match.group(1))
        else:
            print(f"警告: 在文件 {file_path} 中未找到专才学生模型准确率")
            continue

        # 提取“学生-教师联合模型在总体类别上的准确率”
        joint_match = re.search(r'学生-教师联合模型在总体类别上的准确率: ([\d.]+)%', content)
        if joint_match:
            joint_accuracy = float(joint_match.group(1))
        else:
            print(f"警告: 在文件 {file_path} 中未找到联合模型准确率")
            continue

        # 累加准确率
        specialist_accuracy_sum += specialist_accuracy
        joint_accuracy_sum += joint_accuracy
        valid_file_count += 1

    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时发生错误: {e}")
        continue

# 计算并输出平均值
if valid_file_count > 0:
    avg_specialist_accuracy = specialist_accuracy_sum / valid_file_count
    avg_joint_accuracy = joint_accuracy_sum / valid_file_count
    print(f"\n--- 平均准确率结果 ---")
    print(f"专才学生模型在专注类别上的平均准确率: {avg_specialist_accuracy:.2f}%")
    print(f"学生-教师联合模型在总体类别上的平均准确率: {avg_joint_accuracy:.2f}%")
    print(f"基于 {valid_file_count} 个有效文件计算")
else:
    print("错误: 没有找到任何有效的文件")
