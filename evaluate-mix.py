import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import model_dict  # 从 RepDistiller 的 models 模块导入 model_dict

# 显示显卡信息（如果使用 GPU）
if torch.cuda.is_available():
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU 内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
else:
    print("未检测到 GPU，使用 CPU 运行。")

# 定义模型路径（相对路径）
student_model_path = './save/student_model/S_resnet32_T_resnet110_cifar100_kd_r_1.0_a_0.9_b_0.1_target_10_classes_model_1/resnet32_best_model_1.pth'
teacher_model_path = './save/models/resnet110_vanilla/ckpt_epoch_240.pth'

# 实例化学生模型和教师模型
student_model = model_dict['resnet32'](num_classes=100)  # CIFAR-100 有 100 个类别
teacher_model = model_dict['resnet110'](num_classes=100)

# 加载检查点文件
student_checkpoint = torch.load(student_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
teacher_checkpoint = torch.load(teacher_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

# 从检查点中提取 state_dict 并加载到模型中
student_model.load_state_dict(student_checkpoint['model'])
teacher_model.load_state_dict(teacher_checkpoint['model'])

# 将模型设置为评估模式
student_model.eval()
teacher_model.eval()

# 如果有 GPU，将模型移动到 GPU 上
if torch.cuda.is_available():
    student_model = student_model.cuda()
    teacher_model = teacher_model.cuda()

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 的标准化参数
])

# 加载 CIFAR-100 测试数据集
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义学生模型聚焦的 10 个类别（根据 model_1 假设为 0-9）
focused_classes = list(range(10))

# 测试并计算准确率
correct = 0
total = 0
total_images = len(test_loader.dataset)  # 测试集总图片数量

with torch.no_grad():  # 不需要计算梯度
    for i, (data, target) in enumerate(test_loader):
        # 将数据和标签移动到适当的设备（GPU 或 CPU）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data, target = data.to(device), target.to(device)

        # 学生模型进行预测
        student_output = student_model(data)
        student_pred = student_output.argmax(dim=1, keepdim=True)  # 获取预测类别

        # 判断学生模型预测的类别是否在聚焦的 10 个类别内
        if student_pred.item() in focused_classes:
            pred = student_pred  # 使用学生模型结果
        else:
            # 使用教师模型进行预测
            teacher_output = teacher_model(data)
            pred = teacher_output.argmax(dim=1, keepdim=True)  # 获取教师模型预测类别

        # 更新正确预测数和总数
        total += 1
        correct += pred.eq(target.view_as(pred)).sum().item()

        # 每1000张图片显示当前进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{total_images} 张图片 ({(i + 1) / total_images * 100:.2f}%)")

# 计算并打印准确率
accuracy = 100. * correct / total
print(f'测试集准确率: {accuracy:.2f}%')
