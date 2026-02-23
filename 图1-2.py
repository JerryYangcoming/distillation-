import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据（假设 AGENT SYSTEM 提供以下数据，需替换为真实数据）
data = {
    '模型名称': ['AlexNet', 'VGG16', 'ResNet50', 'InceptionV3', 'DenseNet121', 'MobileNetV2'],
    '参数量（百万）': [60, 138, 25.6, 23.8, 8.0, 3.5],
    'Top-1 准确率（%）': [57.2, 71.5, 74.9, 78.8, 75.0, 71.8]
}

# 将数据转换为 Pandas DataFrame
df = pd.DataFrame(data)

# 设置 Seaborn 绘图风格
sns.set(style="whitegrid")

# 创建散点图
plt.figure(figsize=(10, 6))  # 设置图表尺寸
scatter = sns.scatterplot(x='参数量（百万）', y='Top-1 准确率（%）', data=df, s=100)  # 绘制散点图，s 参数控制点的大小

# 添加趋势线，显示复杂度与性能的正相关性
sns.regplot(x='参数量（百万）', y='Top-1 准确率（%）', data=df, scatter=False, ax=scatter)

# 在每个散点旁标注模型名称
for i in range(df.shape[0]):
    plt.text(df['参数量（百万）'][i] + 0.5, df['Top-1 准确率（%）'][i], df['模型名称'][i],
             horizontalalignment='left', size='medium', color='black')

# 设置图表标题和轴标签
plt.title('智能识别模型的复杂度和性能关系', fontsize=14)
plt.xlabel('参数量（百万）', fontsize=12)
plt.ylabel('Top-1 准确率（%）', fontsize=12)

# 显示图表
plt.show()

# 保存图表为图片文件，以便在文档中引用
plt.savefig('model_complexity_performance.png')
