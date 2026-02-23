import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_resnet_flowchart():
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))

    # 定义模型结构中的节点（层）
    nodes = [
        ("Input", 0, 0),              # 输入图像
        ("Conv1\n(BN + ReLU)", 0, 1), # 初始卷积层
        ("Layer1\n(Residual Blocks)", 0, 2), # 第一个残差块组
        ("Layer2\n(Residual Blocks)", 0, 3), # 第二个残差块组
        ("Layer3\n(Residual Blocks)", 0, 4), # 第三个残差块组
        ("AvgPool", 0, 5),            # 平均池化层
        ("FC", 0, 6),                 # 全连接层
        ("Output", 0, 7)              # 分类输出
    ]

    # 定义 layer_idx 对应的标注
    annotations = {
        1: "layer_idx=0",  # Conv1 后的特征
        2: "layer_idx=1",  # Layer1 后的特征
        3: "layer_idx=2",  # Layer2 后的特征
        4: "layer_idx=3"   # Layer3 后的特征
    }

    # 绘制节点（矩形框）
    for i, (label, x, y) in enumerate(nodes):
        ax.add_patch(patches.Rectangle((x - 0.4, y - 0.2), 0.8, 0.4, fill=True, color='lightblue', edgecolor='black'))
        ax.text(x, y, label, ha='center', va='center', fontsize=10)

        # 添加 layer_idx 标注
        if i in annotations:
            ax.text(x + 0.5, y, annotations[i], ha='left', va='center', color='red', fontsize=10)

    # 绘制箭头（表示数据流向）
    for i in range(len(nodes) - 1):
        ax.arrow(nodes[i][1], nodes[i][2] + 0.2, 0, 0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # 设置画布范围并隐藏坐标轴
    ax.set_xlim(-1, 1.5)
    ax.set_ylim(-0.5, 7.5)
    ax.axis('off')

    # 添加标题
    plt.title("ResNet Structure (ResNet56 & ResNet20) with layer_idx Annotations", fontsize=12, pad=20)

    # 保存图像并显示
    plt.savefig("resnet_structure.png", bbox_inches='tight', dpi=300)
    plt.show()

# 运行函数生成流程图
if __name__ == "__main__":
    draw_resnet_flowchart()
