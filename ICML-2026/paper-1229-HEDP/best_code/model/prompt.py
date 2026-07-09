
import torch.nn as nn

def tensor_prompt(a, b, c=None):
    #将这个张量转换为 PyTorch Parameter 对象，并将 requires_grad 设为 True，表示在反向传播过程中将对这个张量计算梯度。
    if c is None:
        p=nn.Linear(b,a,bias=False)
        
    else:#torch.Size([5, 10, 768])
        p=nn.ModuleList([
            nn.Linear(c,b, bias=False)
            for i in range(a)
        ])
    return p



class PrefixOne(nn.Module):
    def __init__(self, emb_d, e_p_length, e_layers):
        super().__init__()
        #self.task_count = 0
        self.emb_d = emb_d # 输入特征向量维度
        
        self.e_layers = e_layers
        
        self.e_p_length = e_p_length

        
        for e in self.e_layers:
            # 按池数量去初始化，对于dualPrompt每个任务对应一个e-prompt,对于L2p来说不需要一对一
            # 采用按任务数量去创建p，便于锁定不是当前任务的参数
            p = tensor_prompt(self.e_p_length, self.emb_d)
            setattr(self, f'e_p_{e}', p)

    
    def forward(self, l,batch_size):

        p_return = None
        #判断当前层是否需要使用 e-prompt
        if l in self.e_layers:
            
            p = getattr(self, f'e_p_{l}')  # 0 based indexing here 取出该层的e-prompt
            p_return = p.weight.expand(batch_size, -1, -1)
    
        return p_return

