# config.py
"""SciNO配置类"""

class Config:
    seed = 42
    device = 'cuda'

    # 数据参数
    n_nodes = 100
    n_samples = 5000

    # 分数网络参数 (DiffAN-style)
    score_hidden = 256
    score_lr = 1e-4
    score_epochs = 100
    batch_size = 128

    # 扩散噪声调度
    sigma_min = 0.01
    sigma_max = 50.0

    # FNO算子参数
    op_modes = 32
    op_width = 128
    op_lr = 3e-4
    op_epochs = 60
    small_dim_for_supervision = 16

    # 训练工具
    use_amp = True
    grad_clip = 1.0

    # 拓扑排序
    n_votes = 3
    masking = True
    residue = False  # 高维时自动禁用

    # 剪枝
    cutoff = 0.001
    pruning_method = 'auto'
