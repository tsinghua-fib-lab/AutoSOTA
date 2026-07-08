import torch
from typing import List, Tuple


# def vec(K):
#     return K.T.flatten().reshape(-1, 1)

# def rebuild(K, r1, r2):
#     """
#     Implements the R(K) operation from the image.
#     K: input matrix (k x d)
#     r1: block height
#     r2: number of block columns
#     """
#     k, d = K.shape
#     num_block_rows = k // r1
#     num_block_cols = r2
#     bw = d // r2 # block width
    
#     blocks = []
#     # R(K) stacks vec(Ki,j) as columns. 
#     # The image shows column-major order through the blocks.
#     for j in range(num_block_cols):
#         for i in range(num_block_rows):
#             # Extract block Ki,j
#             Ki_j = K[i*r1:(i+1)*r1, j*bw:(j+1)*bw]
#             # Vectorize (column-major) and add to list
#             blocks.append(vec(Ki_j))
    
#     return torch.hstack(blocks)


def rebuild(K, r1, r2):
    k, d = K.shape
    Br = k // r1      # number of block rows
    bw = d // r2      # block width

    # Step 1: reshape to (Br, r1, r2, bw)
    K_view = K.view(Br, r1, r2, bw)

    # Step 2: we want to vectorize each (r1, bw) block in COLUMN-MAJOR order.
    # That is equivalent to transposing the block and flattening in row-major.
    # So we permute to (Br, r2, bw, r1) and then flatten last two dims.
    
    # But better: move r1 and bw to end, then transpose those two
    # Actually: to get column-major flatten of (r1, bw), we can do:
    #   block.transpose(-2, -1).contiguous().view(-1)
    # So let's transpose the last two dims of the block

    # Current: (Br, r1, r2, bw) → we want to treat (r1, bw) as block → transpose to (bw, r1)
    # So permute to (Br, r2, bw, r1)
    K_transposed_blocks = K_view.permute(0, 2, 3, 1)  # (Br, r2, bw, r1)

    # Now flatten the last two dims (bw, r1) → (bw * r1,) → this is column-major of original block
    vecs = K_transposed_blocks.reshape(Br, r2, bw * r1)  # (Br, r2, vec_len)

    # Now, we have vecs[i, j] = vectorized block (i,j)
    # But we want to output columns in order: j=0: i=0,1,...,Br-1; j=1: i=0,...
    # So we need to **transpose the first two dimensions** and then **flatten in row-major**
    
    # Transpose to (r2, Br, vec_len)
    vecs = vecs.permute(1, 0, 2)  # (r2, Br, vec_len)

    # Now flatten first two dims: (r2*Br, vec_len), then transpose to (vec_len, r2*Br)
    result = vecs.reshape(r2 * Br, -1).t()

    return result

# def rebuild(grad, block_size: [int, int]):
#     new_matrix_rows = []
#     if grad.dim() == 2:  # 只处理二维梯度（矩阵）
#         # 获取梯度矩阵的尺寸
#         rows, cols = grad.size()
#         # 遍历分块
#         for j in range(0, cols, block_size[1]):
#             for i in range(0, rows, block_size[0]):
#                 # 获取当前块
#                 block = grad[i:i + block_size[0], j:j + block_size[1]]
#                 # 如果块的大小不足，填充零
#                 if block.size(0) < block_size[0] or block.size(1) < block_size[1]:
#                     padding = (
#                         0, block_size[1] - block.size(1),  # 列填充
#                         0, block_size[0] - block.size(0)  # 行填充
#                     )
#                     block = torch.nn.functional.pad(block, padding, "constant", 0)
#                 # 向量化并添加到新矩阵的行中
#                 new_matrix_rows.append(block.T.flatten())

#         # 将所有行堆叠成一个新矩阵
#     if new_matrix_rows:  # 如果有数据
#         new_gad = torch.stack(new_matrix_rows)
#     else:
#         new_gad = torch.empty(0)  # 如果没有梯度数据，返回空矩阵

#     return new_gad

# if __name__ == "__main__":
#     # 定义一个简单的模型
#     class SimpleModel(torch.nn.Module):
#         def __init__(self):
#             super(SimpleModel, self).__init__()
#             self.fc1 = torch.nn.Linear(10, 20)
#             self.fc2 = torch.nn.Linear(20, 10)

#         def forward(self, x):
#             x = self.fc1(x)
#             x = self.fc2(x)
#             return x

#     # 初始化模型和损失函数
#     model = SimpleModel()
#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#     # 模拟输入数据
#     inputs = torch.randn(5, 10)  # batch_size=5, input_size=10
#     targets = torch.randn(5, 10)  # batch_size=5, output_size=10

#     # 前向传播
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)

#     # 反向传播计算梯度
#     loss.backward()

#     # 调用函数将梯度分块并构造新矩阵
#     for param in model.parameters():
#         if param.grad is not None:  # 检查是否有梯度
#             grad = param.grad  # 获取梯度
#             print("旧矩阵的内容:\n", grad)
#             print("新矩阵的形状:", grad.shape)
#             block_size = (2, 2)  # 分块大小为 2x2
#             new_matrix = rebuild(grad, block_size)
#             print("新矩阵的形状:", new_matrix.shape)
#             print("新矩阵的内容:\n", new_matrix)

