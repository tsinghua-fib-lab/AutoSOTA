import torch
from torch import nn, optim
import torch.nn.functional as F


##ASI_block
class ASI_block_att(nn.Module):
    def __init__(self, Input_len, num_id, num_hi, num_head, dropout, IF_Chanel):
        super(ASI_block_att, self).__init__()

        self.final_len = Input_len // 2
        self.IF_Chanel = IF_Chanel

        ### temporal attention
        self.Time_att_1 = Time_att(self.final_len, num_head, dropout)
        self.Time_att_2 = Time_att(self.final_len, num_head, dropout)

        ### Interactive attention
        self.Interaction_att_1 = Interaction_att(self.final_len, num_head, dropout)
        self.Interaction_att_2 = Interaction_att(self.final_len, num_head, dropout)

        ### output
        self.laynorm_1 = nn.LayerNorm([num_hi, self.final_len])
        self.laynorm_2 = nn.LayerNorm([num_hi, self.final_len])
        self.linear_1 = nn.Linear(self.final_len, 1)
        self.linear_2 = nn.Linear(self.final_len, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        ### down sampling
        if self.IF_Chanel:
            B, N, C, L = x.size()
            x = x.reshape((B * N, C, L))
            x_1 = x[:, :, 0::2]
            x_2 = x[:, :, 1::2]
        else:
            x_1 = x[:, :, :, 0::2]
            x_2 = x[:, :, :, 1::2]

        ### temporal attention
        x_1 = self.Time_att_1(x_1)
        x_2 = self.Time_att_2(x_2)

        ### Interactive learning
        x_11 = self.Interaction_att_1(x_2, x_1)
        x_22 = self.Interaction_att_2(x_1, x_2)

        ### MLP
        x_11 = x_11 + self.dropout(self.linear_1(x_11))
        x_22 = x_22 + self.dropout(self.linear_2(x_22))
        x_11 = self.laynorm_1(x_11)
        x_22 = self.laynorm_2(x_22)

        ### output
        x_11 = x_11.unsqueeze(-1)
        x_22 = x_22.unsqueeze(-1)

        if self.IF_Chanel:
            x = torch.cat([x_11, x_22], dim=-1)
            x = x.reshape((B, N, C, L // 2, -1))
        else:
            x = torch.cat([x_11, x_22], dim=-1)
        return x


##MRI_block
class MRI_block_att(nn.Module):
    def __init__(self, Input_len, num_id, num_hi, num_head, dropout, IF_Chanel):
        super(MRI_block_att, self).__init__()
        ### Sequence length
        self.embed = nn.Linear(1, num_hi)
        self.len_2 = Input_len // 2
        self.len_3 = Input_len // 4
        self.IF_Chanel = IF_Chanel

        ###ASI_block
        self.ASI_1 = ASI_block_att(Input_len, num_id, num_hi, num_head, dropout, IF_Chanel)
        self.ASI_2 = ASI_block_att(self.len_2, num_id, num_hi, num_head, dropout, IF_Chanel)
        self.ASI_3 = ASI_block_att(self.len_3, num_id, num_hi, num_head, dropout, IF_Chanel)

        ### Result fusion
        self.Time_att = Time_att(Input_len, num_head, dropout)
        self.laynorm = nn.LayerNorm([num_id, num_hi, Input_len])
        self.dropout = nn.Dropout(dropout)
        self.decode = nn.Linear(num_hi, 1)

    def forward(self, x):

        x = self.embed(x.unsqueeze(-1))
        x = x.transpose(-2, -1)

        ###first layer
        result_1 = self.ASI_1(x)

        ### two layer
        result_2 = 0.0
        for i in range(2):
            line = result_1[:, :, :, :, i]
            line = self.ASI_2(line)
            if i == 0:
                result_2 = line
            else:
                result_2 = torch.cat([result_2, line], dim=-1)

        """
        ###three layer
        result_3 = 0.0
        for i in range(4):
            line = result_2[:,:,:,:,i]
            line = self.ASI_3(line)
            if i == 0:
                result_3 = line
            else:
                result_3 = torch.cat([result_3,line],dim=-1)
        """

        B, N, C = result_2.shape[0], result_2.shape[1], result_2.shape[2]
        result_2 = result_2.reshape((B, N, C, -1))
        x = x + result_2
        x = self.laynorm(x)

        if self.IF_Chanel:
            B, N, C, L = x.size()
            x = x.reshape((B * N, C, L))
            x = self.Time_att(x)
            x = x.reshape((B, N, C, L))
        else:
            x = self.Time_att(x)

        x = x.transpose(-2, -1)
        x = self.decode(x)
        x = x.squeeze(-1)
        return x


### temporal_att
class Time_att(nn.Module):
    def __init__(self, dim_input, num_head, dropout):
        super(Time_att, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.dropout(self.query(x))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
        result = 0.0
        for i in range(self.num_head):
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            line = line.unsqueeze(-1)
            if i < 1:
                result = line
            else:
                result = torch.cat([result, line], dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        x = x + result
        x = self.laynorm(x)
        return x


### Interaction_att
class Interaction_att(nn.Module):
    def __init__(self, dim_input, num_head, dropout):
        super(Interaction_att, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        q = self.dropout(self.query(x1))
        k = self.dropout(self.key(x2))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x2))
        kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
        result = 0.0
        for i in range(self.num_head):
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            line = line.unsqueeze(-1)
            if i < 1:
                result = line
            else:
                result = torch.cat([result, line], dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        return result