import torch
from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from helper.util import accuracy

# 教师模型和学生模型路径
teacher_model_path = "save/models/resnet110_vanilla/ckpt_epoch_240.pth"
student_model_path = "save/student_model/S_wrn_40_1_T_wrn_40_2_cifar100_kd_r_1.0_a_0.9_FixedT_4.0_trial_5,25,1,5,15,30,batch64/wrn_40_1_last.pth"
def load_dataloader():
    _, test_loader, _ = get_cifar100_dataloaders(batch_size=64, num_workers=4, is_instance=True)
    return test_loader

# 加载模型
def load_model(model_path, model_name, num_classes=100):
    checkpoint = torch.load(model_path)
    model = model_dict[model_name](num_classes=num_classes)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 设置为评估模式
    return model

# 模型评估函数
def evaluate_model(model, data_loader):
    correct_1 = 0
    correct_5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:  # 修改为仅接收两个值
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            correct_1 += acc1.item() * images.size(0)
            correct_5 += acc5.item() * images.size(0)
            total += images.size(0)

    top1_acc = correct_1 / total
    top5_acc = correct_5 / total
    return top1_acc, top5_acc


# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 测试推断时间
def measure_inference_time(model, input_size=(64, 3, 32, 32)):
    data = torch.randn(*input_size).cuda()
    model.cuda()
    model.eval()

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    model(data)
    end_time.record()

    torch.cuda.synchronize()
    return start_time.elapsed_time(end_time)

if __name__ == '__main__':
    test_loader = load_dataloader()

    # 加载教师模型
    model_t = load_model(teacher_model_path, 'resnet110')
    model_t.cuda()
    teacher_top1, teacher_top5 = evaluate_model(model_t, test_loader)
    teacher_params = count_parameters(model_t)
    teacher_time = measure_inference_time(model_t)
    print(f"Teacher Model - Top-1 Accuracy: {teacher_top1:.2f}%, Top-5 Accuracy: {teacher_top5:.2f}%")
    print(f"Teacher Model Parameters: {teacher_params}")
    print(f"Teacher Model Inference Time: {teacher_time:.2f} ms")

    # 加载学生模型
    model_s = load_model(student_model_path, 'wrn_40_1')
    model_s.cuda()
    student_top1, student_top5 = evaluate_model(model_s, test_loader)
    student_params = count_parameters(model_s)
    student_time = measure_inference_time(model_s)
    print(f"Student Model - Top-1 Accuracy: {student_top1:.2f}%, Top-5 Accuracy: {student_top5:.2f}%")
    print(f"Student Model Parameters: {student_params}")
    print(f"Student Model Inference Time: {student_time:.2f} ms")
