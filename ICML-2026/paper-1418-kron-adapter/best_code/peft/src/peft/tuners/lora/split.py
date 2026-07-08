import torch
from typing import List, Tuple

def rebuild(grad, block_size: [int, int]):
    new_matrix_rows = []
    if grad.dim() == 2:  # 只处理二维梯度（矩阵）
        # 获取梯度矩阵的尺寸
        rows, cols = grad.size()
        # 遍历分块
        for j in range(0, cols, block_size[1]):
            for i in range(0, rows, block_size[0]):
                # 获取当前块
                block = grad[i:i + block_size[0], j:j + block_size[1]]
                # 如果块的大小不足，填充零
                if block.size(0) < block_size[0] or block.size(1) < block_size[1]:
                    padding = (
                        0, block_size[1] - block.size(1),  # 列填充
                        0, block_size[0] - block.size(0)  # 行填充
                    )
                    block = torch.nn.functional.pad(block, padding, "constant", 0)
                # 向量化并添加到新矩阵的行中
                new_matrix_rows.append(block.flatten())

        # 将所有行堆叠成一个新矩阵
    if new_matrix_rows:  # 如果有数据
        new_gad = torch.stack(new_matrix_rows)
    else:
        new_gad = torch.empty(0)  # 如果没有梯度数据，返回空矩阵

    return new_gad

if __name__ == "__main__":
    # 定义一个简单的模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.fc2 = torch.nn.Linear(20, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # 初始化模型和损失函数
    model = SimpleModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 模拟输入数据
    inputs = torch.randn(5, 10)  # batch_size=5, input_size=10
    targets = torch.randn(5, 10)  # batch_size=5, output_size=10

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 反向传播计算梯度
    loss.backward()

    # 调用函数将梯度分块并构造新矩阵
    for param in model.parameters():
        if param.grad is not None:  # 检查是否有梯度
            grad = param.grad  # 获取梯度
            print("旧矩阵的内容:\n", grad)
            print("新矩阵的形状:", grad.shape)
            block_size = (2, 2)  # 分块大小为 2x2
            new_matrix = rebuild(grad, block_size)
            print("新矩阵的形状:", new_matrix.shape)
            print("新矩阵的内容:\n", new_matrix)

