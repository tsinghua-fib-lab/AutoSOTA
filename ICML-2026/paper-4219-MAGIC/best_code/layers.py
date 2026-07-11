import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


# ====================== AutoEncoder / AutoDecoder ======================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoEncoder, self).__init__()
        layers = []
        for i in range(len(dims) + 1):
            if i == 0:
                layers += [nn.Linear(input_dim, dims[i])]
            elif i == len(dims):
                layers += [nn.Linear(dims[i - 1], feature_dim)]
            else:
                layers += [nn.Linear(dims[i - 1], dims[i])]
            layers += [nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoDecoder, self).__init__()
        layers = []
        rev = list(reversed(dims))
        for i in range(len(rev) + 1):
            if i == 0:
                layers += [nn.Linear(feature_dim, rev[i])]
            elif i == len(rev):
                layers += [nn.Linear(rev[i - 1], input_dim)]
            else:
                layers += [nn.Linear(rev[i - 1], rev[i])]
            layers += [nn.ReLU()]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# ====================== MAGIC Network ======================
class MAGICNetwork(nn.Module):
    """
    MAGIC network for incomplete multi-view clustering.

    It contains view-specific autoencoders, fused, per-view, and masked-fusion
    paths, consensus posterior aggregation, and confidence-gated semantic
    imputation.
    """

    def __init__(self, num_views, input_sizes, dims,
                 dim_high_feature, dim_low_feature, num_clusters):
        super().__init__()
        self.num_views = num_views
        self.num_clusters = num_clusters
        self.dim_high = dim_high_feature

        # encoders / decoders per view
        self.encoders = nn.ModuleList([
            AutoEncoder(input_sizes[v], dim_high_feature, dims) for v in range(num_views)
        ])
        self.decoders = nn.ModuleList([
            AutoDecoder(input_sizes[v], dim_high_feature, dims) for v in range(num_views)
        ])

        # （保留）原 label learning 模块（不直接用作最终 Q）
        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )

        # ====== Transformer (fuse / uni) ======
        def choose_nhead(d_model: int) -> int:
            # 自动选择能整除 d_model 的 nhead，优先较大
            for h in [8, 6, 5, 4, 3, 2, 1]:
                if d_model % h == 0:
                    return h
            return 1

        def build_trm(d_model, nhead=None, ff=4, nl=2):
            if nhead is None:
                nhead = choose_nhead(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=ff * d_model,
                dropout=0.1, batch_first=False, activation='gelu'
            )
            return nn.TransformerEncoder(layer, num_layers=nl)

        self.fuse_trm = build_trm(dim_high_feature)  # 输入 [T=V, B, D]
        self.uni_trm = build_trm(dim_high_feature)   # 输入 [1, B, D]

        # ====== 三头预测器 (输出 K 维概率) ======
        def head(in_dim, hid, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.GELU(),
                nn.Linear(hid, out_dim),
                nn.Softmax(dim=1)
            )
        self.F_fuse = head(dim_high_feature, dim_low_feature, num_clusters)
        self.F_uni  = head(dim_high_feature, dim_low_feature, num_clusters)
        self.F_mask = head(dim_high_feature, dim_low_feature, num_clusters)


        # 三头融合权重 + 温度
        self.tau_bar = 0.8
        self.w_fuse = 1.0 / 3.0
        self.w_uni  = 1.0 / 6.0
        self.w_mask = 1.0 / 2.0

        # 随机掩码参数（InfMasking 风格）
        self.mask_ratio = 0.7
        self.num_mask_views = 6
        self.p_drop = 0.0

        # 轻量增强强度（可预热）
        self.aug_sigma = 0.01
        self.aug_scale = 0.02
        self.aug_keep  = 0.10

        # 掩码缓存（contrastive label sim）
        self.masks: Dict[int, torch.Tensor] = {}

        # ==== 分阶段填补配置（默认关闭）====
        self.impute_enabled = False        # 总开关（阶段1后再开）
        self.impute_conf_thres = 0.55
        self.impute_enable_knn = True     # 阶段2再开 kNN 特征填补

    # === 控制项（训练循环里可随 epoch 调用）===
    def set_impute(self, flag: bool):
        self.impute_enabled = bool(flag)

    def set_impute_conf(self, thres: float):
        self.impute_conf_thres = float(thres)

    def set_impute_knn(self, flag: bool):
        self.impute_enable_knn = bool(flag)

    def set_aug_strength(self, epoch: int, warmup_epochs: int = 25,
                         max_sigma: float = 0.01, max_scale: float = 0.02,
                         keep: float = 0.10):
        t = 0.0 if warmup_epochs <= 0 else min(1.0, max(0.0, epoch / float(warmup_epochs)))
        self.aug_sigma = max_sigma * t
        self.aug_scale = max_scale * t
        self.aug_keep  = keep

    def reconstruct_only(self, data_views):
        """Run only encoders and decoders for reconstruction pretraining."""
        reconstructions = []
        features = []

        for v in range(self.num_views):
            h_v = self.encoders[v](data_views[v])
            x_hat_v = self.decoders[v](h_v)

            features.append(h_v)
            reconstructions.append(x_hat_v)

        return reconstructions, features

    # ---------- 轻量潜空间增强 ----------
    def aug_H(self, H: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return H
        noise  = self.aug_sigma * torch.randn_like(H)
        scale  = 1.0 + self.aug_scale * torch.tanh(torch.randn_like(H[..., :1]))
        keepid = torch.rand_like(H[..., :1]).lt(self.aug_keep).float()
        H_aug  = (H + noise) * scale
        return H_aug * (1.0 - keepid) + H * keepid

    # ---------- 视图级序列堆叠 ----------
    def concat_modalities(self, H_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(H_list, dim=0)  # [V,B,D]

    # ---------- 子集保留（旧接口） ----------
    def mask_modalities(self, H_list: List[torch.Tensor], keep_mask: List[bool]) -> torch.Tensor:
        seq = []
        for v, keep in enumerate(keep_mask):
            Hv = H_list[v]
            seq.append(Hv if keep else torch.zeros_like(Hv))
        return torch.stack(seq, dim=0)  # [V,B,D]

    # ---------- token 池化 ----------
    def pool_tokens(self, Z_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Z_tokens: [V,B,D], mask: [V,B] (1=keep)
        if mask is None:
            return Z_tokens.mean(dim=0)
        denom = mask.sum(dim=0).clamp_min(1.0).float()
        num = (Z_tokens * mask.unsqueeze(-1).float()).sum(dim=0)
        return num / denom.unsqueeze(-1)

    # ---------- 三头 logit 均值重心 ----------
    def logits_barycenter(self, Qf: torch.Tensor, Qu: torch.Tensor, Qm: torch.Tensor,
                          tau: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        zf = torch.log(Qf + eps)
        zu = torch.log(Qu + eps)
        zm = torch.log(Qm + eps)
        z  = (self.w_fuse * zf + self.w_uni * zu + self.w_mask * zm) / (self.w_fuse + self.w_uni + self.w_mask)
        return torch.softmax(z / tau, dim=1)

    # ---------- DEC 尖化 ----------
    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        col_sum = q.sum(dim=0, keepdim=True)
        weight = (q ** 2) / (col_sum + 1e-8)
        return weight / (weight.sum(dim=1, keepdim=True) + 1e-8)

    # ---------- 对比掩码（去掉对角与正对） ----------
    def mask_correlated_samples2(self, N: int) -> torch.Tensor:
        if N in self.masks:
            return self.masks[N]
        m = torch.ones((N, N), dtype=torch.bool)
        m.fill_diagonal_(False)
        half = N // 2
        for i in range(half):
            m[i, half + i] = False
            m[half + i, i] = False
        self.masks[N] = m
        return m

    # ---------- 视图相似度（标签空间） ----------
    def compute_label_similarity(self, q_i, q_j, mask_i, mask_j, temperature_l=0.5, normalized=False):
        device = q_i.device
        mask_i = torch.as_tensor(mask_i, dtype=torch.bool, device=device)
        mask_j = torch.as_tensor(mask_j, dtype=torch.bool, device=device)
        common = mask_i & mask_j
        Bc = int(common.sum().item())
        if Bc == 0:
            return torch.tensor(0.0, device=device)
        qi = q_i[common]  # [Bc,K]
        qj = q_j[common]  # [Bc,K]
        qi_t = self.target_distribution(qi).t()  # [K,Bc]
        qj_t = self.target_distribution(qj).t()  # [K,Bc]
        q = torch.cat((qi_t, qj_t), dim=0)       # [2K,Bc]
        if normalized:
            qn = q / (q.norm(dim=1, keepdim=True) + 1e-8)
            sim = (qn @ qn.t()) / temperature_l
        else:
            sim = (q @ q.t()) / temperature_l
        K = self.num_clusters
        N = 2 * K
        sim_i_j = torch.diag(sim,  K)
        sim_j_i = torch.diag(sim, -K)
        pos = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples2(N).to(device)
        with torch.no_grad():
            pred_labels = torch.argmax(q, dim=1)
        label_mask = pred_labels.unsqueeze(0) != pred_labels.unsqueeze(1)
        final_mask = mask & label_mask
        neg_vals = sim[final_mask]
        if neg_vals.numel() == 0:
            return torch.tensor(0.0, device=device)
        M = neg_vals.numel() // N
        neg_vals = neg_vals[:N * M].reshape(N, M)
        logits = torch.cat((pos, neg_vals), dim=1)    # [N,1+M]
        probs = torch.softmax(logits, dim=1)[:, 0]
        return probs.sum() / max(Bc, 1)

    # ---------- 计算每视图的簇原型（按软分配加权） ----------
    def _compute_centroids(self, features_list, probs_list, all_masks):
        K = self.num_clusters
        V = self.num_views
        centroids_per_view = []
        for v in range(V):
            H = features_list[v]                   # [B,D]
            Q = probs_list[v]                      # [B,K]
            m = all_masks[v].float().unsqueeze(1)  # [B,1]
            Qm = Q * m                             # [B,K]
            mass = Qm.sum(dim=0, keepdim=True) + 1e-8
            C = (Qm.t() @ H) / mass.t()            # [K,D]
            centroids_per_view.append(C)
        return centroids_per_view

    # ---------- Sinkhorn (K×K) ----------
    def _sinkhorn(self, cost, a=None, b=None, eps=0.1, iters=30):
        K = cost.shape[0]
        if a is None:
            a = torch.full((K,), 1.0 / K, device=cost.device, dtype=cost.dtype)
        if b is None:
            b = torch.full((K,), 1.0 / K, device=cost.device, dtype=cost.dtype)
        Kmat = torch.exp(-cost / eps) + 1e-12      # [K,K]
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(iters):
            u = a / (Kmat @ v + 1e-12)
            v = b / (Kmat.t() @ u + 1e-12)
        T = torch.diag(u) @ Kmat @ torch.diag(v)   # [K,K]
        T = T / (T.sum(dim=1, keepdim=True) + 1e-12)
        return T

    # ---------- 源/目的分配置信度 ----------
    def _conf(self, q: torch.Tensor) -> torch.Tensor:
        # q:[N,K] softmax；conf = 1 - H(q)/logK
        ent = -(q * (q.clamp_min(1e-12)).log()).sum(dim=1)
        return 1.0 - ent / math.log(self.num_clusters + 1e-12)

    # ---------- OT 软对齐做“分配填补”（置信门控） ----------
    def _impute_probs_ot(self, probs_list, features_list, all_masks, view_sim_matrix, eps=0.1, iters=30):
        V = self.num_views
        C_list = self._compute_centroids(features_list, probs_list, all_masks)  # List[K×D]
        a_list = []
        for v in range(V):
            m = all_masks[v].float().unsqueeze(1)
            mass = (probs_list[v] * m).sum(dim=0)  # [K]
            mass = mass / (mass.sum() + 1e-12)
            a_list.append(mass)

        imputed = []
        for i in range(V):
            Qi = probs_list[i].clone()     # [B,K]
            miss = ~all_masks[i]           # [B]
            if miss.any():
                # 选择源视图 j（基于视图相似度）
                sims = view_sim_matrix[i].clone()
                sims[i] = -1e9
                j = torch.argmax(sims).item()

                # 类-类传输矩阵 T（全局）
                Ci, Cj = C_list[i], C_list[j]      # [K,D]
                Ci_n = F.normalize(Ci, dim=1); Cj_n = F.normalize(Cj, dim=1)
                cos = Ci_n @ Cj_n.t()              # [K,K]
                cost = (1.0 - cos).clamp_min(0)    # [K,K]
                ai, aj = a_list[i], a_list[j]
                T = self._sinkhorn(cost, a=ai, b=aj, eps=eps, iters=iters)   # [K,K]

                # 对缺视图样本做置信门控（源视图 j 的分配置信度）
                qj = probs_list[j][miss]           # [Bmiss,K]
                conf_src = self._conf(qj)          # [Bmiss]
                good = conf_src > self.impute_conf_thres
                if good.any():
                    Qi_miss = Qi[miss]
                    Qi_miss[good] = (qj[good] @ T)
                    Qi_miss[good] = Qi_miss[good] / (Qi_miss[good].sum(dim=1, keepdim=True) + 1e-12)
                    Qi[miss] = Qi_miss
            imputed.append(Qi)
        return imputed

    # ---------- kNN + 原型混合做“特征填补”（置信门控 + 可切换） ----------
    def _impute_features_knn(self, features_list, probs_list, all_masks, k=5):
        V = self.num_views
        centroids = self._compute_centroids(features_list, probs_list, all_masks)  # List[K×D]
        imputed_feats = []
        for v in range(V):
            H = features_list[v].clone()    # [B,D]
            Q = probs_list[v]               # [B,K] （已经过分配填补）
            visible = all_masks[v]          # [B]

            if (~visible).any():
                # 目的视图 v 的分配置信度（只在高置信时填特征）
                conf_tgt = self._conf(Q)           # [B]
                do_pos = (~visible) & (conf_tgt > self.impute_conf_thres)

                if do_pos.any():
                    argmax_c = torch.argmax(Q, dim=1)  # [B]
                    for pos in torch.where(do_pos)[0]:
                        c = argmax_c[pos].item()
                        if self.impute_enable_knn:
                            pool_idx = torch.where((argmax_c == c) & visible)[0]
                            if pool_idx.numel() > 0:
                                ref = H[pool_idx]                # [M,D]
                                q = centroids[v][c].unsqueeze(0) # [1,D]
                                ref_n = F.normalize(ref, dim=1); q_n = F.normalize(q, dim=1)
                                sim = (ref_n @ q_n.t()).squeeze(1)  # [M]
                                topk = torch.topk(sim, k=min(k, sim.numel()), largest=True).indices
                                H[pos] = ref[topk].mean(dim=0)
                            else:
                                H[pos] = centroids[v][c]
                        else:
                            # 阶段1：只用类原型，不做 kNN
                            H[pos] = centroids[v][c]
            imputed_feats.append(H)
        return imputed_feats

    # ---------- 随机特征掩码（叠加可见性/视图级Drop） ----------
    def _rand_feat_mask(self, H: torch.Tensor, keep_row: torch.Tensor, ratio: float):
        # H: [B,D], keep_row: [B] bool，1=该样本保留此视图
        B, D = H.shape
        feat_keep = (torch.rand(B, D, device=H.device) > ratio).float()  # 1=keep
        feat_keep = feat_keep * keep_row.float().unsqueeze(1)
        return H * feat_keep

    def _build_view_sim_matrix(self, view_sims: List[torch.Tensor], device):
        V = self.num_views
        M = torch.zeros((V, V), device=device)
        idx = 0
        for i in range(V):
            for j in range(i + 1, V):
                M[i, j] = view_sims[idx]
                M[j, i] = view_sims[idx]
                idx += 1
        return M

    # ====================== forward ======================
    def forward(self, data_views, return_aug: bool = False):
        dvs: List[torch.Tensor] = []
        features: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []  # 每视图 [B] bool
        view_sims: List[torch.Tensor] = []

        # ===== Step 1: 编码一次，得到各视图 H_v；记录可见掩码 =====
        for v in range(self.num_views):
            x_v = data_views[v]                              # [B, Din_v]
            mask = (x_v.abs().sum(dim=1) > 1e-6) & (~torch.isnan(x_v).any(dim=1))
            all_masks.append(mask)
            H_v = self.encoders[v](x_v)                      # [B, D_high]
            features.append(H_v)
            dvs.append(self.decoders[v](H_v))                # 预重构

        # ===== Step 2: 在 H 上做两次增强 =====
        H1 = [self.aug_H(h) for h in features]
        H2 = [self.aug_H(h) for h in features]

        # ===== 可见性矩阵（B,V）与视图级随机保留（训练态才Drop） =====
        M_vis = torch.stack(all_masks, dim=1).to(torch.bool)   # [B, V]
        if self.training:
            bern = (torch.rand_like(M_vis.float()) > self.p_drop)
            M_keep_view = (bern & M_vis)                       # 仅在可见视图内丢弃
            # 至少保留一个视图
            all_zero = ~M_keep_view.any(dim=1)
            if all_zero.any():
                first_vis = M_vis[all_zero].float().argmax(dim=1)
                M_keep_view[all_zero] = False
                M_keep_view[all_zero, first_vis] = True
        else:
            M_keep_view = M_vis                                # 评估态不丢弃
        key_pad = ~M_keep_view  # [B,V], True=忽略

        # ===== Step 3.1: 融合分支（未掩码） =====
        seq1 = self.concat_modalities(H1)                                  # [V,B,D]
        seq2 = self.concat_modalities(H2)                                  # [V,B,D]
        Zf1_tokens = self.fuse_trm(seq1, src_key_padding_mask=key_pad)     # [V,B,D]
        Zf2_tokens = self.fuse_trm(seq2, src_key_padding_mask=key_pad)     # [V,B,D]
        Z_fuse1 = self.pool_tokens(Zf1_tokens, mask=M_keep_view.t())       # [B,D]
        Z_fuse2 = self.pool_tokens(Zf2_tokens, mask=M_keep_view.t())       # [B,D]

        # ===== Step 3.2: 单视图分支 =====
        Z_uni1, Z_uni2 = [], []
        for v in range(self.num_views):
            Zu1_tok = self.uni_trm(H1[v].unsqueeze(0))       # [1,B,D]
            Zu2_tok = self.uni_trm(H2[v].unsqueeze(0))       # [1,B,D]
            Z_uni1.append(Zu1_tok.squeeze(0))                # [B,D]
            Z_uni2.append(Zu2_tok.squeeze(0))                # [B,D]

        # ===== Step 3.3: 随机掩码融合（K 次采样, 仅训练态） =====
        if self.training and self.num_mask_views > 0:
            Z_mask_1, Z_mask_2 = [], []
            for _ in range(self.num_mask_views):
                H1_masked = [self._rand_feat_mask(H1[v], M_keep_view[:, v], self.mask_ratio) for v in range(self.num_views)]
                H2_masked = [self._rand_feat_mask(H2[v], M_keep_view[:, v], self.mask_ratio) for v in range(self.num_views)]
                seq1_m = self.concat_modalities(H1_masked)                                      # [V,B,D]
                seq2_m = self.concat_modalities(H2_masked)                                      # [V,B,D]
                Zm1_tokens = self.fuse_trm(seq1_m, src_key_padding_mask=key_pad)                # [V,B,D]
                Zm2_tokens = self.fuse_trm(seq2_m, src_key_padding_mask=key_pad)                # [V,B,D]
                Zm1 = self.pool_tokens(Zm1_tokens, mask=M_keep_view.t())                        # [B,D]
                Zm2 = self.pool_tokens(Zm2_tokens, mask=M_keep_view.t())                        # [B,D]
                Z_mask_1.append(Zm1)
                Z_mask_2.append(Zm2)
            Zm1_mean = torch.stack(Z_mask_1, dim=0).mean(dim=0)
            Zm2_mean = torch.stack(Z_mask_2, dim=0).mean(dim=0)
        else:
            Z_mask_1, Z_mask_2 = [], []
            Zm1_mean = Z_fuse1
            Zm2_mean = Z_fuse2

        # ===== 三头输出 + 视图内 logit 重心，得到每视图 Q =====
        Q_fuse_1 = self.F_fuse(Z_fuse1)
        Q_fuse_2 = self.F_fuse(Z_fuse2)
        Q_fuse = self.F_fuse(0.5 * (Z_fuse1 + Z_fuse2))

        Q_mask_1 = self.F_mask(Zm1_mean)
        Q_mask_2 = self.F_mask(Zm2_mean)
        Q_mask = self.F_mask(0.5 * (Zm1_mean + Zm2_mean))

        Q_per_view = []
        Q_uni_1, Q_uni_2 = [], []
        for v in range(self.num_views):
            q_u1 = self.F_uni(Z_uni1[v])                       # [B,K]
            q_u2 = self.F_uni(Z_uni2[v])                       # [B,K]
            Q_uni_1.append(q_u1); Q_uni_2.append(q_u2)
            Q_uni_avg = self.F_uni(0.5 * (Z_uni1[v] + Z_uni2[v]))
            Q_v = self.logits_barycenter(Q_fuse, Q_uni_avg, Q_mask, tau=self.tau_bar)  # [B,K]
            Q_per_view.append(Q_v)

        # ===== 视图相似度（标签空间） =====
        view_sims = []
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                view_sim = self.compute_label_similarity(Q_per_view[i], Q_per_view[j], all_masks[i], all_masks[j],
                                                         temperature_l=0.5, normalized=False)
                view_sims.append(view_sim)

        # ====== 分阶段 + 置信门控的填补 ======
        if self.impute_enabled:
            probs_list_new = self._impute_probs_ot(Q_per_view, features, all_masks,
                                                   self._build_view_sim_matrix(view_sims, Q_per_view[0].device))
            # 第二阶段才开 kNN
            if self.impute_enable_knn:
                features_list_new = self._impute_features_knn(features, probs_list_new, all_masks, k=5)
            else:
                features_list_new = features
        else:
            probs_list_new = Q_per_view
            features_list_new = features

        # 重构
        dvs_new = []
        for v in range(self.num_views):
            dvs_new.append(self.decoders[v](features_list_new[v]))

        if return_aug:
            aug = dict(
                Z_fuse_1=Z_fuse1, Z_fuse_2=Z_fuse2,
                Z_uni_1=Z_uni1,   Z_uni_2=Z_uni2,
                Z_mask_1=Z_mask_1, Z_mask_2=Z_mask_2,
                Q_fuse_1=Q_fuse_1, Q_fuse_2=Q_fuse_2,
                Q_uni_1=Q_uni_1,   Q_uni_2=Q_uni_2,
                Q_mask_1=Q_mask_1, Q_mask_2=Q_mask_2,
                vis_masks=all_masks
            )
            return probs_list_new, dvs_new, features, features_list_new, view_sims, aug
        else:
            return probs_list_new, dvs_new, features, features_list_new, view_sims

