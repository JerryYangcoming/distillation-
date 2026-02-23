import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['font.size'] = 14  # 设置全局字体大小为18

# 数据：年份、参数量（百万）、模型名称
years = [2012, 2014, 2015, 2016, 2018, 2020, 2023]
params = [60, 138, 25.6, 60.2, 66, 175000, 500000]
models = ['AlexNet', 'VGG16', 'ResNet50', 'ResNet152', 'EfficientNet', 'GPT3', 'GPT4']

# 根据模型类型分配颜色：CNN用蓝色，LLM用橙色
colors = ['blue' if model in ['AlexNet', 'VGG-16', 'ResNet-50', 'ResNet-152', 'EfficientNet'] else 'orange' for model in models]

# 创建图表
plt.figure(figsize=(10, 6))  # 设置图像尺寸为10x6英寸
plt.bar(years, params, color=colors)  # 绘制柱状图

# 设置横轴刻度为年份
plt.xticks(years, [str(year) for year in years])

# 设置纵轴为对数刻度
plt.yscale('log')

# 添加标题和轴标签

plt.xlabel('年份')
plt.ylabel('参数量（百万）')

# 在每个柱子上添加模型名称和年份标签
for i, (year, param, model) in enumerate(zip(years, params, models)):
    plt.text(year, param, f'{model}', ha='center', va='bottom')

# 保存图像为高分辨率PNG文件
plt.savefig('figure1-1.png', dpi=300)

# 可选：显示图表（用于调试，论文中可注释掉）
plt.show()
