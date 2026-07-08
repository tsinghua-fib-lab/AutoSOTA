import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs, hidden_dim=32, revin=True):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dim = hidden_dim
        self.revin = revin
        self.relu  = nn.ReLU()

        self.input_layer = nn.Linear(self.pred_len, self.hidden_dim)
        # self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.hidden_layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x, y):   # x: [Batch, Input length, Channel]
        
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        # print(x.shape)

        # if self.revin:
        #     means = x.mean(-1, keepdim=True)
        #     stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        #     x = x - means
        #     x = x / stdev

        z = self.input_layer(x)
        # print(z.shape)
        z = self.relu(z)
        x = self.output_layer(z)
        
        # tmp = [z1, z2]
        # hidden_features = []
        # for z in tmp:
        #     hidden_features.append(z.unsqueeze(2))
        # hidden_features = torch.cat(hidden_features, dim=2)  # z: [bs x nvars x scale_num x Input length]

        # if self.revin:
        #     x = x * stdev + means
            
        x = x.permute(0,2,1) 
        
        return x, z   # [Batch, Output length, Channel]
    
    
# import torch
# import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, configs, num_channels=8, kernel_size=5, stride=2, revin=True):
#         super(Model, self).__init__()

#         # print(f"padding: {kernel_size//2}")
        
#         self.context_len = configs.seq_len
#         self.revin = revin
#         input_channels = 1  # 因为输入是 1D 序列，所以初始输入通道为 1
        
#         self.relu = nn.ReLU()
#         self.cnn1d_input_layer = nn.Conv1d(input_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=0)
#         self.cnn1d_hidden_layer1 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=0)
#         self.cnn1d_hidden_layer2 = nn.Conv1d(num_channels, num_channels, kernel_size, stride=stride, padding=0)
#         self.output_layer = None
        
    
#     def forward(self, x, y):
        
#         x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
#         # print(x.shape)
#         B, C, L = x.shape
#         if self.revin:
#             means = x.mean(-1, keepdim=True)
#             stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
#             x = x - means
#             x = x / stdev
#         x = x.reshape(B * C, L)
#         # print(x.shape)
        
#         x = x.unsqueeze(1)  # 添加一个通道维度，使输入的形状从 [B, context_len] 变为 [B, 1, context_len]
#         # print(x.shape)
#         z1 = self.relu(self.cnn1d_input_layer(x))
#         z2 = self.relu(self.cnn1d_hidden_layer1(z1))
#         z3 = self.relu(self.cnn1d_hidden_layer2(z2))
#         z3_flatten = torch.flatten(z3, start_dim=1)
#         # print(z3.shape)

#         if self.output_layer is None:
#             self.output_layer = nn.Linear(z3_flatten.shape[1], self.context_len).to(x.device)
        
#         x = self.output_layer(z3_flatten)
#         # print(x.shape)
#         x = x.reshape(B, C, L)
#         # print(x.shape)
        
#         if self.revin:
#             x = x * stdev + means
#         x = x.permute(0, 2, 1) 
        
#         features = [z1, z2, z3]
        
        # return x, features
    
    
# # 创建模型实例
# context_len = 96  # 输入序列的长度
# model = Model(context_len)

# # 示例输入
# input_tensor = torch.randn(32, context_len, 7)  # batch size = 32 channel = 7
# output_tensor, features = model(input_tensor)

# print("Input shape:", input_tensor.shape)
# print("Output shape:", output_tensor.shape)

# for feature in features:
#     print(feature.shape)
    