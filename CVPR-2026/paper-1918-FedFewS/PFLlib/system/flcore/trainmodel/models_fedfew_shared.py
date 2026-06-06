"""
FedFew Model Set with Shared Representation

支持可调节的参数共享策略：
- sharing_degree = 0.0: 完全独立（K 个完整模型）
- sharing_degree = 1.0: 共享特征提取器，K 个独立分类器
- sharing_degree ∈ (0, 1): 部分层共享
"""
import torch
import torch.nn as nn
import copy

# 浮点数比较精度阈值
FLOAT_EPSILON = 1e-6


class FedFewModelSetShared(nn.Module):
    """
    FedFew with Configurable Sharing Degree

    Args:
        base_model: 基础模型（如 ResNet18）
        K: 模型数量
        mu: 平滑参数（用于 logsumexp）
        aggregation: 聚合方法（'logsumexp', 'polynomial', 'power_mean'）
        poly_power: 多项式/幂平均的幂次参数
        sharing_degree: 共享程度 [0, 1]
            - 0.0: 完全独立（当前 FedFew）
            - 1.0: 共享 backbone，K 个分类器头（类似 FedRep）
            - 0.5: 共享一半层
    """

    def __init__(self, base_model, K, mu=0.001, aggregation='logsumexp',
                 poly_power=2.0, sharing_degree=0.0, init='same'):
        super().__init__()
        self.K = K
        self.mu = mu
        self.aggregation = aggregation
        self.poly_power = poly_power
        self.sharing_degree = sharing_degree
        self.init = init  # 'same', 'same_noise', 'independent'

        # 根据模型类型分离特征提取器和分类器
        self.model_type = self._detect_model_type(base_model)

        # 统一初始化接口：根据 sharing_degree 自动选择初始化策略
        self._init_models(base_model, sharing_degree)

        # 用于记录最近一次 forward 的 K 个模型 loss
        self.last_losses = None

        # 为了统一访问K个模型，创建一个辅助属性
        # 对于独立模型（sd=0.0）：直接是 private_features 列表
        # 对于共享模型（sd>0.0）：创建包装器列表
        if not self._is_no_sharing():
            self.models = [self.get_model(k) for k in range(self.K)]
        else:
            self.models = self.private_features

    def _is_full_sharing(self):
        """判断是否为完全共享模式 (sharing_degree ≈ 1.0)"""
        return abs(self.sharing_degree - 1.0) < FLOAT_EPSILON

    def _is_no_sharing(self):
        """判断是否为完全独立模式 (sharing_degree ≈ 0.0)"""
        return abs(self.sharing_degree - 0.0) < FLOAT_EPSILON

    def _detect_model_type(self, model):
        """检测模型类型"""
        # 检测 PFLlib 的自定义 ResNet（使用 layer_0, layer_1, ...）
        if hasattr(model, 'fc') and hasattr(model, 'layer_0'):
            return 'resnet_pfllib'
        # 检测标准 torchvision ResNet（使用 layer1, layer2, layer3, layer4）
        elif hasattr(model, 'fc') and hasattr(model, 'layer1'):
            return 'resnet'
        elif hasattr(model, 'classifier') and hasattr(model, 'features'):
            return 'vgg'
        else:
            return 'unknown'

    def _init_models(self, base_model, sharing_degree):
        """
        统一初始化接口：根据 sharing_degree 选择初始化策略

        Args:
            base_model: 基础模型
            sharing_degree: 共享程度 [0, 1]
        """
        if self._is_full_sharing():
            # 完全共享：共享 backbone + K 个分类器
            self._init_full_sharing(base_model)
        elif self._is_no_sharing():
            # 完全独立：K 个完整模型
            self._init_no_sharing(base_model)
        else:
            # 部分共享：共享前 N 层 + K 个私有后续层
            self._init_partial_sharing(base_model, sharing_degree)

    def _init_full_sharing(self, base_model):
        """初始化完全共享模式（sd=1.0）"""
        if self.model_type == 'resnet_pfllib':
            # PFLlib ResNet: layer_0 ~ layer_7 (8 blocks for ResNet18)
            # 共享：conv1, bn1, relu, maxpool, layer_0~7, avgpool
            backbone_layers = [
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
            ]
            # 添加所有 layer_i
            for i in range(len(base_model.layers)):
                backbone_layers.append(getattr(base_model, f'layer_{i}'))
            backbone_layers.append(base_model.avgpool)

            self.shared_backbone = nn.Sequential(*backbone_layers)

            # K 个独立分类器（根据 init 策略）
            self.classifiers = nn.ModuleList()
            for k in range(self.K):
                fc = copy.deepcopy(base_model.fc)
                # 根据初始化策略处理
                if self.init == 'same':
                    pass  # 保持相同
                elif self.init == 'same_noise' and k > 0:
                    self._add_noise_to_model(fc, std=0.01)
                elif self.init == 'independent' and k > 0 and hasattr(fc, 'reset_parameters'):
                    fc.reset_parameters()
                self.classifiers.append(fc)

            self.private_features = None

        elif self.model_type == 'resnet':
            # 标准 torchvision ResNet: layer1, layer2, layer3, layer4
            self.shared_backbone = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool
            )

            # K 个独立分类器（根据 init 策略）
            self.classifiers = nn.ModuleList()
            for k in range(self.K):
                fc = copy.deepcopy(base_model.fc)
                # 根据初始化策略处理
                if self.init == 'same':
                    pass  # 保持相同
                elif self.init == 'same_noise' and k > 0:
                    self._add_noise_to_model(fc, std=0.01)
                elif self.init == 'independent' and k > 0 and hasattr(fc, 'reset_parameters'):
                    fc.reset_parameters()
                self.classifiers.append(fc)

            self.private_features = None

        else:
            raise NotImplementedError(f"Full sharing not implemented for {self.model_type}")

    def _init_no_sharing(self, base_model):
        """初始化完全独立模式（sd=0.0）"""
        self.shared_backbone = None
        self.private_features = nn.ModuleList()

        if self.init == 'same':
            # 策略1: 所有模型完全相同初始化
            for k in range(self.K):
                self.private_features.append(copy.deepcopy(base_model))

        elif self.init == 'same_noise':
            # 策略2: 相同初始化 + 高斯噪声（std=0.01）
            for k in range(self.K):
                model_k = copy.deepcopy(base_model)
                if k > 0:  # 第一个模型不加噪
                    self._add_noise_to_model(model_k, std=0.01)
                self.private_features.append(model_k)

        elif self.init == 'independent':
            # 策略3: 独立随机初始化（原逻辑）
            self.private_features.append(copy.deepcopy(base_model))  # 第一个保留原始
            for k in range(1, self.K):
                model_k = copy.deepcopy(base_model)
                self._reinitialize_model(model_k)
                self.private_features.append(model_k)

        else:
            raise ValueError(f"Unknown init strategy: {self.init}. Must be 'same', 'same_noise', or 'independent'.")

        self.classifiers = None

    def _init_partial_sharing(self, base_model, sharing_degree):
        """初始化部分共享模式（0 < sd < 1）"""
        if self.model_type == 'resnet_pfllib':
            # PFLlib ResNet18: layer_0~7 (8 blocks = 4 groups of 2 blocks each)
            # 分组策略：
            # Group 0: conv_stem (conv1, bn1, relu, maxpool)
            # Group 1: layer_0, layer_1 (64 channels)
            # Group 2: layer_2, layer_3 (128 channels)
            # Group 3: layer_4, layer_5 (256 channels)
            # Group 4: layer_6, layer_7 (512 channels)

            groups = [
                ('conv_stem', nn.Sequential(base_model.conv1, base_model.bn1,
                                           base_model.relu, base_model.maxpool)),
            ]

            # 将 layer_i 按每2个分组
            num_layers = len(base_model.layers)
            for i in range(0, num_layers, 2):
                if i + 1 < num_layers:
                    # 两个 layer 一组
                    group_layers = nn.Sequential(
                        getattr(base_model, f'layer_{i}'),
                        getattr(base_model, f'layer_{i+1}')
                    )
                    groups.append((f'layer_group_{i//2+1}', group_layers))
                else:
                    # 最后一个 layer（如果总数是奇数）
                    groups.append((f'layer_{i}', getattr(base_model, f'layer_{i}')))

            # 计算共享组数
            num_shared = int(len(groups) * sharing_degree)
            num_shared = max(1, num_shared)  # 至少共享一层

            # 构建共享部分
            shared_modules = []
            for i in range(num_shared):
                shared_modules.append(groups[i][1])
            self.shared_backbone = nn.Sequential(*shared_modules)

            # 构建私有部分（每个模型独立）
            self.private_features = nn.ModuleList()
            for k in range(self.K):
                private_modules = []
                for i in range(num_shared, len(groups)):
                    private_modules.append(copy.deepcopy(groups[i][1]))
                private_modules.append(base_model.avgpool)

                # 根据初始化策略处理
                private_seq = nn.Sequential(*private_modules)
                if self.init == 'same':
                    pass  # 保持相同
                elif self.init == 'same_noise' and k > 0:
                    self._add_noise_to_model(private_seq, std=0.01)
                elif self.init == 'independent' and k > 0:
                    self._reinitialize_model(private_seq)
                self.private_features.append(private_seq)

            # K 个独立分类器（根据 init 策略）
            self.classifiers = nn.ModuleList()
            for k in range(self.K):
                fc = copy.deepcopy(base_model.fc)
                # 根据初始化策略处理
                if self.init == 'same':
                    pass  # 保持相同
                elif self.init == 'same_noise' and k > 0:
                    self._add_noise_to_model(fc, std=0.01)
                elif self.init == 'independent' and k > 0 and hasattr(fc, 'reset_parameters'):
                    fc.reset_parameters()
                self.classifiers.append(fc)

        elif self.model_type == 'resnet':
            # 标准 torchvision ResNet: layer1, layer2, layer3, layer4
            layers = [
                ('conv_stem', nn.Sequential(base_model.conv1, base_model.bn1,
                                           base_model.relu, base_model.maxpool)),
                ('layer1', base_model.layer1),
                ('layer2', base_model.layer2),
                ('layer3', base_model.layer3),
                ('layer4', base_model.layer4),
            ]

            # 计算共享层数
            num_shared = int(len(layers) * sharing_degree)
            num_shared = max(1, num_shared)  # 至少共享一层

            # 构建共享部分
            shared_modules = []
            for i in range(num_shared):
                shared_modules.append(layers[i][1])
            self.shared_backbone = nn.Sequential(*shared_modules)

            # 构建私有部分（每个模型独立）
            self.private_features = nn.ModuleList()
            for k in range(self.K):
                private_modules = []
                for i in range(num_shared, len(layers)):
                    private_modules.append(copy.deepcopy(layers[i][1]))
                private_modules.append(base_model.avgpool)

                # 根据初始化策略处理
                private_seq = nn.Sequential(*private_modules)
                if self.init == 'same':
                    pass  # 保持相同
                elif self.init == 'same_noise' and k > 0:
                    self._add_noise_to_model(private_seq, std=0.01)
                elif self.init == 'independent' and k > 0:
                    self._reinitialize_model(private_seq)
                self.private_features.append(private_seq)

            # K 个独立分类器（根据 init 策略）
            self.classifiers = nn.ModuleList()
            for k in range(self.K):
                fc = copy.deepcopy(base_model.fc)
                # 根据初始化策略处理
                if self.init == 'same':
                    pass  # 保持相同
                elif self.init == 'same_noise' and k > 0:
                    self._add_noise_to_model(fc, std=0.01)
                elif self.init == 'independent' and k > 0 and hasattr(fc, 'reset_parameters'):
                    fc.reset_parameters()
                self.classifiers.append(fc)

        else:
            raise NotImplementedError(f"Partial sharing not implemented for {self.model_type}")

    def _add_noise_to_model(self, model, std=0.01):
        """
        为模型参数添加高斯噪声

        Args:
            model: 模型
            std: 高斯噪声标准差（默认0.01）
        """
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * std
                param.add_(noise)

    def _reinitialize_model(self, model):
        """重新初始化模型参数（增加多样性）"""
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x, y, criterion):
        """
        Forward pass: 计算 smooth min loss

        根据 sharing_degree 选择不同的前向传播路径
        """
        losses = []

        if self._is_full_sharing():
            # 完全共享：shared_backbone + K 个 classifiers
            shared_feat = self.shared_backbone(x)
            shared_feat = torch.flatten(shared_feat, 1)

            for k in range(self.K):
                output = self.classifiers[k](shared_feat)
                loss = criterion(output, y)
                losses.append(loss)

        elif self._is_no_sharing():
            # 完全独立：K 个完整模型
            for k in range(self.K):
                output = self.private_features[k](x)
                loss = criterion(output, y)
                losses.append(loss)

        else:
            # 部分共享：shared_backbone + K 个 private_features + K 个 classifiers
            shared_feat = self.shared_backbone(x)

            for k in range(self.K):
                private_feat = self.private_features[k](shared_feat)
                private_feat = torch.flatten(private_feat, 1)
                output = self.classifiers[k](private_feat)
                loss = criterion(output, y)
                losses.append(loss)

        # 聚合损失
        losses_tensor = torch.stack(losses)
        smooth_min = self._aggregate_losses(losses_tensor)

        # 保存最近一次的 loss 值
        self.last_losses = losses_tensor.detach()

        return smooth_min

    def _aggregate_losses(self, losses_tensor):
        """聚合损失（支持多种方法）"""
        if self.aggregation == 'logsumexp':
            smooth_min = -self.mu * torch.logsumexp(-losses_tensor / self.mu, dim=0)

        elif self.aggregation == 'polynomial':
            eps = 1e-8
            inv_losses = 1.0 / (losses_tensor + eps)
            weights = torch.pow(inv_losses, self.poly_power)
            weights = weights / (weights.sum() + eps)
            smooth_min = (weights * losses_tensor).sum()

        elif self.aggregation == 'power_mean':
            eps = 1e-8
            p = -self.poly_power
            if abs(p) < 1e-6:
                smooth_min = torch.exp(torch.log(losses_tensor + eps).mean())
            else:
                power_sum = torch.pow(losses_tensor + eps, p).mean()
                smooth_min = torch.pow(power_sum, 1.0 / p)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return smooth_min

    def get_model(self, k):
        """
        获取第 k 个模型（用于评估）

        Returns:
            一个 nn.Module 包装器，支持 .eval(), .train(), forward()
        """
        # 创建包装器类
        class ModelWrapper(nn.Module):
            def __init__(self, parent, model_idx):
                super().__init__()
                self.parent = parent
                self.k = model_idx

            def forward(self, x):
                if self.parent._is_full_sharing():
                    # 完全共享模式
                    feat = self.parent.shared_backbone(x)
                    feat = torch.flatten(feat, 1)
                    return self.parent.classifiers[self.k](feat)

                elif self.parent._is_no_sharing():
                    # 完全独立模式
                    return self.parent.private_features[self.k](x)

                else:
                    # 部分共享模式
                    shared_feat = self.parent.shared_backbone(x)
                    private_feat = self.parent.private_features[self.k](shared_feat)
                    private_feat = torch.flatten(private_feat, 1)
                    return self.parent.classifiers[self.k](private_feat)

            def eval(self):
                """设置为评估模式"""
                if self.parent.shared_backbone is not None:
                    self.parent.shared_backbone.eval()
                if self.parent.private_features is not None:
                    self.parent.private_features[self.k].eval()
                if self.parent.classifiers is not None:
                    self.parent.classifiers[self.k].eval()
                return super().eval()

            def train(self, mode=True):
                """设置为训练模式"""
                if self.parent.shared_backbone is not None:
                    self.parent.shared_backbone.train(mode)
                if self.parent.private_features is not None:
                    self.parent.private_features[self.k].train(mode)
                if self.parent.classifiers is not None:
                    self.parent.classifiers[self.k].train(mode)
                return super().train(mode)

        return ModelWrapper(self, k)

    def eval_all_models(self):
        """将所有模型设置为评估模式"""
        if self.shared_backbone is not None:
            self.shared_backbone.eval()
        if self.private_features is not None:
            for model in self.private_features:
                model.eval()
        if self.classifiers is not None:
            for classifier in self.classifiers:
                classifier.eval()

    def train_all_models(self):
        """将所有模型设置为训练模式"""
        if self.shared_backbone is not None:
            self.shared_backbone.train()
        if self.private_features is not None:
            for model in self.private_features:
                model.train()
        if self.classifiers is not None:
            for classifier in self.classifiers:
                classifier.train()

    def get_model_parameters(self, k):
        """
        获取第 k 个模型的参数（用于冻结/解冻）

        Returns:
            generator of parameters belonging to model k
        """
        if self._is_full_sharing():
            # 完全共享模式：只冻结分类器（共享backbone不冻结）
            return self.classifiers[k].parameters()

        elif self._is_no_sharing():
            # 完全独立模式：冻结整个模型
            return self.private_features[k].parameters()

        else:
            # 部分共享模式：冻结私有层和分类器（共享层不冻结）
            import itertools
            return itertools.chain(
                self.private_features[k].parameters(),
                self.classifiers[k].parameters()
            )

    # ========================================================================
    # ⚠️ NEW API: 统一接口 - 消除客户端和服务器端重复的 if-else 判断
    # ========================================================================

    def get_phase1_parameters(self):
        """
        获取 Phase 1 需要优化的参数（共享部分）

        Returns:
            generator of parameters or None (如果 sd=0.0)

        Example:
            # 服务器端创建 Phase 1 优化器
            if model_set.needs_phase1():
                optimizer_shared = torch.optim.SGD(
                    model_set.get_phase1_parameters(),
                    lr=learning_rate
                )
        """
        if self.sharing_degree > 0.0:
            return self.shared_backbone.parameters()
        else:
            return None

    def needs_phase1(self):
        return not self._is_no_sharing()

    def has_shared_backbone(self):
        """
        是否有共享 backbone

        Returns:
            bool: True if sd>0, False if sd=0
        """
        return self.shared_backbone is not None

    def has_private_features(self):
        """
        是否有私有特征层（区分 sd=1.0 和 0<sd<1.0）

        Returns:
            bool: True if 0<sd<1.0 (部分共享), False if sd=0 or sd=1.0

        Example:
            if model_set.has_private_features():
                # 需要经过 private_features 层
                private_feat = model_set.private_features[k](shared_feat)
        """
        return (self.private_features is not None and
                self.sharing_degree > 1e-6 and
                abs(self.sharing_degree - 1.0) > 1e-6)

    def freeze_for_phase1(self):
        """
        Phase 1: 冻结私有部分，准备训练共享部分

        - sd=1.0: 冻结 K 个 classifiers
        - 0<sd<1.0: 冻结 K 个 (private_features + classifiers)
        - sd=0.0: 无操作（无共享部分）

        Example:
            # 客户端 Phase 1 训练前
            model_set.freeze_for_phase1()
            model_set.unfreeze_for_phase1()
            # ... 训练代码 ...
        """
        if self._is_no_sharing():
            return  # 无共享，无需冻结

        # 冻结所有私有部分
        if self._is_full_sharing():
            # sd=1.0: 只冻结 classifiers
            for k in range(self.K):
                for param in self.classifiers[k].parameters():
                    param.requires_grad = False
        else:
            # 0<sd<1.0: 冻结 private_features + classifiers
            for k in range(self.K):
                for param in self.private_features[k].parameters():
                    param.requires_grad = False
                for param in self.classifiers[k].parameters():
                    param.requires_grad = False

    def unfreeze_for_phase1(self):
        """
        Phase 1: 解冻共享部分，准备训练

        - sd>0: 解冻 shared_backbone
        - sd=0.0: 无操作

        Example:
            model_set.freeze_for_phase1()
            model_set.unfreeze_for_phase1()
        """
        if self._is_no_sharing():
            return

        for param in self.shared_backbone.parameters():
            param.requires_grad = True

    def freeze_for_phase2(self):
        """
        Phase 2: 冻结共享部分，准备训练私有部分

        - sd>0: 冻结 shared_backbone
        - sd=0.0: 无操作（无共享部分）

        Example:
            # 客户端 Phase 2 训练前
            model_set.freeze_for_phase2()
            for k in range(K):
                model_set.unfreeze_for_phase2(k)
            # ... 训练代码 ...
        """
        if self._is_no_sharing():
            return

        for param in self.shared_backbone.parameters():
            param.requires_grad = False

    def unfreeze_for_phase2(self, k):
        """
        Phase 2: 解冻第k个模型的私有部分，准备训练

        Args:
            k: 模型索引

        - sd=1.0: 解冻 classifiers[k]
        - 0<sd<1.0: 解冻 private_features[k] + classifiers[k]
        - sd=0.0: 解冻完整模型 models[k]

        Example:
            model_set.freeze_for_phase2()
            for k in range(K):
                model_set.unfreeze_for_phase2(k)
        """
        for param in self.get_model_parameters(k):
            param.requires_grad = True
