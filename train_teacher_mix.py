from graphviz import Digraph

def create_flowchart():
    # 创建有向图对象，设置从上到下的布局
    dot = Digraph(comment='Training Process Flowchart')
    dot.attr(rankdir='TB')  # Top to Bottom

    # 开始节点
    dot.node('Start', 'Start', shape='ellipse')

    # 解析命令行参数
    dot.node('Parse Options', 'Parse Command Line Options', shape='box')
    dot.edge('Start', 'Parse Options')

    # 定义目标类别组
    dot.node('Define Groups', 'Define 10 Target Class Groups', shape='box')
    dot.edge('Parse Options', 'Define Groups')

    # 循环训练10个模型
    dot.node('Loop Models', 'For Each Model (1 to 10)', shape='diamond')
    dot.edge('Define Groups', 'Loop Models')

    # 训练模型
    dot.node('Train Model', 'Train Model for Target Classes', shape='box')
    dot.edge('Loop Models', 'Train Model', label='Yes')

    # 训练模型内部流程
    dot.node('Set Weights', 'Set Class Weights (Target Classes Higher)', shape='box')
    dot.edge('Train Model', 'Set Weights')

    dot.node('Set Paths', 'Set Model Name and Paths', shape='box')
    dot.edge('Set Weights', 'Set Paths')

    dot.node('Init Logger', 'Initialize TensorBoard Logger', shape='box')
    dot.edge('Set Paths', 'Init Logger')

    dot.node('Load Data', 'Load CIFAR-100 Data Loaders', shape='box')
    dot.edge('Init Logger', 'Load Data')

    dot.node('Load Teacher', 'Load Teacher Model', shape='box')
    dot.edge('Load Data', 'Load Teacher')

    dot.node('Load Student', 'Load Student Model', shape='box')
    dot.edge('Load Teacher', 'Load Student')

    dot.node('Transfer Learning', 'Transfer Weights from Teacher to Student', shape='box')
    dot.edge('Load Student', 'Transfer Learning')

    dot.node('Set Criteria', 'Set Loss Criteria (CLS, KL, KD)', shape='box')
    dot.edge('Transfer Learning', 'Set Criteria')

    dot.node('Set Optimizer', 'Set SGD Optimizer', shape='box')
    dot.edge('Set Criteria', 'Set Optimizer')

    dot.node('Validate Teacher', 'Validate Teacher Model (Target Classes)', shape='box')
    dot.edge('Set Optimizer', 'Validate Teacher')

    # 训练循环
    dot.node('Training Loop', 'For Each Epoch (1 to 240)', shape='diamond')
    dot.edge('Validate Teacher', 'Training Loop')

    dot.node('Adjust LR', 'Adjust Learning Rate', shape='box')
    dot.edge('Training Loop', 'Adjust LR', label='Yes')

    dot.node('Train Epoch', 'Train One Epoch with Distillation', shape='box')
    dot.edge('Adjust LR', 'Train Epoch')

    dot.node('Validate Student', 'Validate Student Model (Target Classes)', shape='box')
    dot.edge('Train Epoch', 'Validate Student')

    dot.node('Log Metrics', 'Log Metrics to TensorBoard', shape='box')
    dot.edge('Validate Student', 'Log Metrics')

    dot.node('Save Best Model', 'Save Best Model if Improved', shape='box')
    dot.edge('Log Metrics', 'Save Best Model')

    dot.node('Save Periodic', 'Save Model Periodically (Every 40 Epochs)', shape='box')
    dot.edge('Save Best Model', 'Save Periodic')

    # 结束Epoch循环
    dot.edge('Save Periodic', 'Training Loop', label='Next Epoch')

    # 结束模型循环
    dot.edge('Training Loop', 'Loop Models', label='No', style='dashed')

    # 结束节点
    dot.node('End', 'End', shape='ellipse')
    dot.edge('Loop Models', 'End', label='No')

    return dot

if __name__ == '__main__':
    # 生成流程图并保存为PNG文件
    dot = create_flowchart()
    dot.render('training_process_flowchart', format='png', view=False)
    print("流程图已生成：training_process_flowchart.png")
