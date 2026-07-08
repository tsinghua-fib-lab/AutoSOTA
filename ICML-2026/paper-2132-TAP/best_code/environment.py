"""
环境模块 - Inpainting 条件生成

支持分类和回归任务：
- 分类：target_class (0~C-1), Label Gate (p(y)+margin), 分类NLL
- 回归：target_bin (0~B-1), Target Gate (残差阈值), 回归NLL
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

from config import Config
from generators.base import BaseGenerator


class DataGenerationEnv:
    """数据生成环境（支持分类/回归）"""
    
    def __init__(self, config: Config, generator: BaseGenerator, train_data: pd.DataFrame):
        self.config = config
        self.generator = generator
        
        # 任务类型
        self.task_type = config.data.task_type
        self.is_regression = (self.task_type == "regression")
        
        # 数据设置
        self.target_col = config.data.target_column
        self.columns = [col for col in train_data.columns if col != self.target_col]
        
        # 编码数据
        self.train_data_raw = train_data.copy()
        self.label_encoder = None
        self.feature_encoders = {}
        self._encode_train_data()
        
        # 类别/bin信息
        if self.is_regression:
            self._init_bins()
            self.num_targets = self.num_bins
        else:
            self.classes = np.unique(self.train_data_raw[self.target_col])
            self.num_classes = len(self.classes)
            self.num_targets = self.num_classes
        
        self.D_real = self.train_data_raw.copy()
        self._original_train_data = train_data.copy()
        
        # TabPFN
        self._init_tabpfn()
        self._fit_tabpfn()
        
        # 统计信息
        self._init_stats()
        
        # 模板和规则
        self.templates = list(config.mask_template.templates.keys())
        self.num_templates = len(self.templates)
        self.anchor_rules = config.inpaint.anchor_rules
        self.num_anchor_rules = len(self.anchor_rules)
        
        # 合成数据缓冲
        self.synthetic_buffer = pd.DataFrame()
        self.proposals = []
        
        # Gate 统计
        self.gate_stats = {t: {"pass": 0, "total": 0} for t in self.templates}
        # C-01: diversity gate periodic recalibration
        self._diversity_stale = True
        self._last_diversity_buffer_size = 0
        
        # 计算列重要性
        self._compute_column_importance()
        
        # Anchor quality scores (A-03)
        self._anchor_quality_scores = None
        self._compute_anchor_quality()
    
    def _init_bins(self):
        """回归：将y按分位数切成bins"""
        y = self.train_data_raw[self.target_col].values.astype(float)
        self.num_bins = self.config.inpaint.num_bins
        self.bin_edges = np.percentile(y, np.linspace(0, 100, self.num_bins + 1))
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        print(f"回归bins: {self.num_bins}个")
    
    def _get_bin(self, y):
        """返回y所属的bin索引"""
        y = np.asarray(y).astype(float)
        return np.clip(np.digitize(y, self.bin_edges) - 1, 0, self.num_bins - 1)
    
    def _encode_train_data(self):
        """编码训练数据"""
        if self.is_regression:
            # 回归：target保持float
            self.train_data_raw[self.target_col] = pd.to_numeric(
                self.train_data_raw[self.target_col], errors='coerce'
            ).astype('float64')
        else:
            # 分类：标签编码
            if self.train_data_raw[self.target_col].dtype == 'object':
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self.train_data_raw[self.target_col])
                self.train_data_raw[self.target_col] = self.label_encoder.transform(
                    self.train_data_raw[self.target_col]
                )
        
        # 特征编码
        for col in self.columns:
            try:
                self.train_data_raw[col] = pd.to_numeric(self.train_data_raw[col], errors='raise').astype('float64')
                if self.train_data_raw[col].isna().any():
                    self.train_data_raw[col] = self.train_data_raw[col].fillna(self.train_data_raw[col].mean())
            except (ValueError, TypeError):
                le = LabelEncoder()
                self.train_data_raw[col] = le.fit_transform(self.train_data_raw[col].astype(str)).astype('float64')
                self.feature_encoders[col] = le
    
    def _init_tabpfn(self):
        """初始化 TabPFN (with sklearn fallback on license/auth failure)"""
        if self.is_regression:
            try:
                from tabpfn import TabPFNRegressor
                self.tabpfn = TabPFNRegressor(ignore_pretraining_limits=True)
                # Quick test fit to catch license/auth errors early
                X_test = np.random.randn(4, 2)
                y_test = np.array([0.0, 1.0, 0.0, 1.0])
                self.tabpfn.fit(X_test, y_test)
                print("TabPFN Regressor 初始化成功")
            except Exception as e:
                from sklearn.ensemble import GradientBoostingRegressor
                self.tabpfn = GradientBoostingRegressor(n_estimators=100, random_state=42)
                print(f"警告: TabPFN 不可用 ({e}), 使用 GradientBoostingRegressor")
        else:
            try:
                from tabpfn import TabPFNClassifier
                self.tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
                # Quick test fit to catch license/auth errors early
                X_test = np.random.randn(4, 2)
                y_test = np.array([0, 1, 0, 1])
                self.tabpfn.fit(X_test, y_test)
                print("TabPFN Classifier 初始化成功")
            except Exception as e:
                from sklearn.ensemble import RandomForestClassifier
                self.tabpfn = RandomForestClassifier(n_estimators=100, random_state=42)
                print(f"警告: TabPFN 不可用 ({e}), 使用 RandomForestClassifier")
    
    def _fit_tabpfn(self):
        """在 D_real 上拟合 TabPFN"""
        X = self.D_real[self.columns].values
        y = self.D_real[self.target_col].values
        self.tabpfn.fit(X, y)
    
    def _init_stats(self):
        """初始化统计信息"""
        y = self.D_real[self.target_col].values
        X = self.D_real[self.columns].values
        
        if self.is_regression:
            # 回归：bin-wise统计
            bins = self._get_bin(y)
            self.target_counts = np.bincount(bins, minlength=self.num_targets)
            
            preds = self.tabpfn.predict(X)
            residuals = np.abs(y - preds)
            
            # bin-wise NLL (用平均残差作为proxy)
            self.target_nll = np.zeros(self.num_targets)
            self.target_error = np.zeros(self.num_targets)
            for b in range(self.num_targets):
                mask = (bins == b)
                if mask.sum() > 0:
                    self.target_nll[b] = residuals[mask].mean()
                    self.target_error[b] = residuals[mask].mean()
            
            self.target_per_target = len(y) // self.num_targets
            
            # 残差阈值（用于gate）
            self._residual_threshold = np.percentile(
                residuals, self.config.inpaint.residual_threshold_percentile
            )
            print(f"残差阈值 (p{self.config.inpaint.residual_threshold_percentile}): {self._residual_threshold:.4f}")
        else:
            # 分类：class-wise统计
            self.target_counts = np.bincount(y.astype(int), minlength=self.num_targets)
            
            probs = self.tabpfn.predict_proba(X)
            preds = np.argmax(probs, axis=1)
            
            self.target_error = np.zeros(self.num_targets)
            self.target_nll = np.zeros(self.num_targets)
            for c in range(self.num_targets):
                mask = (y == c)
                if mask.sum() > 0:
                    self.target_error[c] = (preds[mask] != c).mean()
                    self.target_nll[c] = -np.log(probs[mask, c].clip(1e-10, 1)).mean()
            
            self.target_per_target = len(y) // self.num_targets
        
        self.diversity_score = 0.0
    
    def _compute_column_importance(self):
        """预计算列重要性"""
        if self.is_regression:
            from sklearn.feature_selection import mutual_info_regression
            mi_func = mutual_info_regression
        else:
            from sklearn.feature_selection import mutual_info_classif
            mi_func = mutual_info_classif
        
        X = self.D_real[self.columns].values
        y = self.D_real[self.target_col].values
        
        mi = mi_func(X, y, random_state=42)
        correlated_idx = np.argsort(mi)[-max(3, len(self.columns)//4):]
        
        stabilities = []
        for col_idx in range(len(self.columns)):
            vals = []
            for _ in range(10):
                boot_idx = np.random.choice(len(X), len(X), replace=True)
                vals.append(X[boot_idx, col_idx].std())
            stabilities.append(np.std(vals))
        stable_idx = np.argsort(stabilities)[:max(3, len(self.columns)//4)]
        
        self.correlated_cols = set(correlated_idx.tolist())
        self.stable_cols = set(stable_idx.tolist())
        
        print(f"列重要性: correlated={len(self.correlated_cols)}, stable={len(self.stable_cols)}")
    
    def _compute_anchor_quality(self):
        X = self.D_real[self.columns].values
        y = self.D_real[self.target_col].values
        n_samples = len(X)
        
        if self.is_regression:
            preds = self.tabpfn.predict(X)
            residuals = np.abs(y.astype(float) - preds)
            max_res = np.max(residuals) if np.max(residuals) > 0 else 1.0
            conf = 1.0 - residuals / max_res
        else:
            probs = self.tabpfn.predict_proba(X)
            conf = np.max(probs, axis=1)
        
        centroid_dist = np.zeros(n_samples)
        if self.is_regression:
            bins = self._get_bin(y)
            for b in range(self.num_targets):
                mask = bins == b
                if mask.sum() > 1:
                    centroid = X[mask].mean(axis=0)
                    std = X[mask].std(axis=0) + 1e-8
                    centroid_dist[mask] = np.linalg.norm((X[mask] - centroid) / std, axis=1)
        else:
            for c in range(self.num_targets):
                mask = y.astype(int) == c
                if mask.sum() > 1:
                    centroid = X[mask].mean(axis=0)
                    std = X[mask].std(axis=0) + 1e-8
                    centroid_dist[mask] = np.linalg.norm((X[mask] - centroid) / std, axis=1)
        
        max_cd = np.max(centroid_dist) if np.max(centroid_dist) > 0 else 1.0
        proximity = 1.0 - centroid_dist / max_cd
        
        uniqueness = np.zeros(n_samples)
        if self.is_regression:
            bins = self._get_bin(y)
            for b in range(self.num_targets):
                mask = bins == b
                n_in = mask.sum()
                if n_in > 1:
                    class_X = X[mask]
                    nn = NearestNeighbors(n_neighbors=min(2, n_in))
                    nn.fit(class_X)
                    dist, _ = nn.kneighbors(class_X)
                    uniqueness[mask] = dist[:, 1] if dist.shape[1] > 1 else dist[:, 0]
                else:
                    uniqueness[mask] = 1.0
        else:
            for c in range(self.num_targets):
                mask = y.astype(int) == c
                n_in = mask.sum()
                if n_in > 1:
                    class_X = X[mask]
                    nn = NearestNeighbors(n_neighbors=min(2, n_in))
                    nn.fit(class_X)
                    dist, _ = nn.kneighbors(class_X)
                    uniqueness[mask] = dist[:, 1] if dist.shape[1] > 1 else dist[:, 0]
                else:
                    uniqueness[mask] = 1.0
        
        max_unq = np.max(uniqueness) if np.max(uniqueness) > 0 else 1.0
        uniqueness = uniqueness / max_unq
        
        self._anchor_quality_scores = 0.4 * conf + 0.3 * proximity + 0.3 * uniqueness
        
        if self.is_regression:
            bins = self._get_bin(y)
            for b in range(self.num_targets):
                mask = bins == b
                if mask.sum() > 0:
                    best_idx = np.where(mask)[0][np.argmax(self._anchor_quality_scores[mask])]
                    self._anchor_quality_scores[best_idx] = max(self._anchor_quality_scores[best_idx], 0.5)
        else:
            for c in range(self.num_targets):
                mask = y.astype(int) == c
                if mask.sum() > 0:
                    best_idx = np.where(mask)[0][np.argmax(self._anchor_quality_scores[mask])]
                    self._anchor_quality_scores[best_idx] = max(self._anchor_quality_scores[best_idx], 0.5)
        
        mean_q = float(self._anchor_quality_scores.mean())
        std_q = float(self._anchor_quality_scores.std())
        print(f"Anchor quality scores: mean={mean_q:.3f}, std={std_q:.3f}")
    
    def get_state(self) -> np.ndarray:
        """获取状态向量"""
        # C-04: Defensive TabPFN refresh to prevent state leakage
        self._fit_tabpfn()
        current = self.target_counts.copy()
        if len(self.synthetic_buffer) > 0:
            encoded = self._encode_data(self.synthetic_buffer)
            if len(encoded) > 0:
                syn_y = encoded[self.target_col].values
                if self.is_regression:
                    syn_bins = self._get_bin(syn_y)
                    current = current + np.bincount(syn_bins, minlength=self.num_targets)
                else:
                    current = current + np.bincount(syn_y.astype(int), minlength=self.num_targets)
        
        deficit = np.clip((self.target_per_target - current) / max(self.target_per_target, 1), -1, 1)
        
        gate_rates = [self.gate_stats[t]["pass"] / max(self.gate_stats[t]["total"], 1) 
                      for t in self.templates]
        
        return np.concatenate([
            deficit, self.target_nll,
            np.array(gate_rates), [self.diversity_score]
        ]).astype(np.float32)
    
    def get_state_dim(self) -> int:
        return 2 * self.num_targets + self.num_templates + 1
    
    def _determine_anchor_rule(self, target_idx: int) -> str:
        """基于信息论的确定性anchor选择"""
        nll = self.target_nll[target_idx]
        error = self.target_error[target_idx]
        
        if nll > np.median(self.target_nll):
            return "high_uncertainty"
        if error > np.median(self.target_error):
            return "high_error"
        return "random"
    
    def step(self, action, splits: List = None) -> Tuple[float, Dict]:
        """执行一步"""
        anchor_rule = self._determine_anchor_rule(action.target_class)
        anchor_rule_idx = self.anchor_rules.index(anchor_rule) if anchor_rule in self.anchor_rules else 3
        anchor_indices = self._select_anchor_indices(action.target_class, anchor_rule_idx)
        if len(anchor_indices) == 0:
            return 0.0, {"n_generated": 0, "n_passed": 0, "delta_U": 0.0, "passed_batch": pd.DataFrame()}
        
        template = self.templates[action.template_id]
        mask_strength = action.explore_level
        num_mask, cat_mask = self._build_mask(template, mask_strength)
        
        stochasticity = action.explore_level * 0.5
        try:
            generated = self.generator.sample_inpaint(
                anchor_indices=anchor_indices,
                num_mask=num_mask,
                cat_mask=cat_mask,
                n_samples_per_anchor=self.config.inpaint.samples_per_anchor,
                stochasticity=stochasticity
            )
        except Exception as e:
            print(f"生成失败: {e}")
            return 0.0, {"n_generated": 0, "n_passed": 0, "delta_U": 0.0, "passed_batch": pd.DataFrame()}
        
        if len(generated) == 0:
            return 0.0, {"n_generated": 0, "n_passed": 0, "delta_U": 0.0, "passed_batch": pd.DataFrame()}
        
        passed = self._apply_gates(generated)
        self.gate_stats[template]["total"] += len(generated)
        self.gate_stats[template]["pass"] += len(passed)
        
        if len(passed) == 0:
            return 0.0, {"n_generated": len(generated), "n_passed": 0, "delta_U": 0.0, "passed_batch": pd.DataFrame()}
        
        reward, delta_U = self._compute_reward(passed, splits=splits)
        
        return reward, {
            "n_generated": len(generated), 
            "n_passed": len(passed), 
            "delta_U": delta_U,
            "passed_batch": passed
        }
    
    def _select_anchor_indices(self, target_idx: int, anchor_rule: int) -> np.ndarray:
        """选择 anchor 行的索引"""
        # C-04: Ensure TabPFN is in correct state (full D_real fit)
        self._fit_tabpfn()
        y = self.D_real[self.target_col].values
        
        if self.is_regression:
            bins = self._get_bin(y)
            mask = (bins == target_idx)
        else:
            mask = (y == target_idx)
        
        candidate_indices = np.where(mask)[0]
        
        if len(candidate_indices) == 0:
            return np.array([], dtype=int)
        
        n_anchors = min(
            self.config.inpaint.samples_per_step // self.config.inpaint.samples_per_anchor,
            len(candidate_indices)
        )
        
        rule = self.anchor_rules[anchor_rule]
        
        if rule == "high_uncertainty":
            X = self.D_real.iloc[candidate_indices][self.columns].values
            if self.is_regression:
                # 用bagging方差近似不确定性
                preds = self.tabpfn.predict(X)
                y_cand = y[candidate_indices]
                residuals = np.abs(y_cand - preds)
                selected_local = np.argsort(residuals)[-n_anchors:]
            else:
                probs = self.tabpfn.predict_proba(X)
                entropy = -np.sum(probs * np.log(probs.clip(1e-10, 1)), axis=1)
                selected_local = np.argsort(entropy)[-n_anchors:]
            return candidate_indices[selected_local]
        
        elif rule == "high_error":
            X = self.D_real.iloc[candidate_indices][self.columns].values
            y_cand = y[candidate_indices]
            preds = self.tabpfn.predict(X)
            if self.is_regression:
                residuals = np.abs(y_cand - preds)
                selected_local = np.argsort(residuals)[-n_anchors:]
            else:
                wrong_local = np.where(preds != y_cand)[0]
                if len(wrong_local) >= n_anchors:
                    return candidate_indices[wrong_local[:n_anchors]]
                selected_local = np.arange(min(n_anchors, len(candidate_indices)))
            return candidate_indices[selected_local]
        
        if self._anchor_quality_scores is not None and len(candidate_indices) > n_anchors:
            cand_qualities = self._anchor_quality_scores[candidate_indices]
            temperature = 0.5
            exp_q = np.exp((cand_qualities - cand_qualities.max()) / temperature)
            probs = exp_q / exp_q.sum()
            n_select = min(n_anchors, len(candidate_indices))
            selected_local = np.random.choice(len(candidate_indices), size=n_select, replace=False, p=probs)
            return candidate_indices[selected_local]
        
        return np.random.choice(candidate_indices, size=min(n_anchors, len(candidate_indices)), replace=False)
    
    def _build_mask(self, template_name: str, strength: float) -> Tuple[list, list]:
        """构造mask"""
        try:
            num_mask, cat_mask = self.generator.get_column_masks([])
            d_num, d_cat = len(num_mask), len(cat_mask)
        except:
            return [True], [True]
        
        num_mask = [True] * d_num
        cat_mask = [True] * d_cat
        
        all_fixed = set()
        if template_name == "conservative":
            all_fixed = self.correlated_cols | self.stable_cols
        
        for col_idx in all_fixed:
            if col_idx < d_num:
                num_mask[col_idx] = False
            else:
                cat_mask[col_idx - d_num] = False
        
        if strength < 1.0:
            n_gen = sum(num_mask) + sum(cat_mask)
            n_release = int((1 - strength) * n_gen * 0.3)
            gen_idx = [i for i in range(d_num) if num_mask[i]]
            if len(gen_idx) > n_release and n_release > 0:
                for i in np.random.choice(gen_idx, n_release, replace=False):
                    num_mask[i] = False
        
        return num_mask, cat_mask
    
    def _apply_gates(self, generated: pd.DataFrame) -> pd.DataFrame:
        """Gate 过滤"""
        if len(generated) == 0:
            return generated
        
        encoded = self._encode_data(generated)
        if len(encoded) == 0:
            return generated.iloc[:0]
        
        if self.is_regression:
            passed = self._apply_gates_regression(generated, encoded)
        else:
            passed = self._apply_gates_classification(generated, encoded)
        
        if len(passed) > 0:
            passed = self._diversity_gate(passed)
        
        return passed
    
    def _apply_gates_classification(self, generated: pd.DataFrame, encoded: pd.DataFrame) -> pd.DataFrame:
        """分类Gate：p(y) + margin"""
        X = encoded[self.columns].values
        y = encoded[self.target_col].values.astype(int)
        probs = self.tabpfn.predict_proba(X)
        
        keep = []
        for i in range(len(probs)):
            if y[i] >= probs.shape[1]:
                continue
            p_y = probs[i, y[i]]
            p_others = np.delete(probs[i], y[i])
            margin = p_y - p_others.max() if len(p_others) > 0 else p_y
            if p_y >= self.config.inpaint.label_p_min and margin >= self.config.inpaint.label_margin_threshold:
                keep.append(i)
        
        return generated.iloc[keep].reset_index(drop=True)
    
    def _apply_gates_regression(self, generated: pd.DataFrame, encoded: pd.DataFrame) -> pd.DataFrame:
        """回归Gate：残差阈值"""
        X = encoded[self.columns].values
        y = encoded[self.target_col].values.astype(float)
        
        preds = self.tabpfn.predict(X)
        residuals = np.abs(y - preds)
        
        keep = residuals <= self._residual_threshold
        return generated.iloc[keep].reset_index(drop=True)
    
    def _diversity_gate(self, df: pd.DataFrame) -> pd.DataFrame:
        """多样性过滤"""
        # C-04: Ensure TabPFN is in correct state
        self._fit_tabpfn()
        encoded = self._encode_data(df)
        if len(encoded) == 0:
            return df.iloc[:0]
        
        X_syn = encoded[self.columns].values
        y_syn = encoded[self.target_col].values
        X_real = self.D_real[self.columns].values
        y_real = self.D_real[self.target_col].values
        
        # 计算real-real距离分布 (C-01: periodic recalibration)
        if not hasattr(self, '_diversity_thresholds') or self._diversity_stale:
            if len(X_real) < 2:
                return df.reset_index(drop=True)
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(X_real)
            dist, _ = nn.kneighbors(X_real)
            real_real_dist = dist[:, 1] / np.sqrt(X_real.shape[1])
            
            self._diversity_thresholds = {}
            if self.is_regression:
                bins = self._get_bin(y_real)
                for b in range(self.num_targets):
                    mask = (bins == b)
                    if mask.sum() > 1:
                        self._diversity_thresholds[b] = np.percentile(real_real_dist[mask], 5)
                    else:
                        self._diversity_thresholds[b] = 0.02
            else:
                for c in range(self.num_targets):
                    mask = (y_real == c)
                    if mask.sum() > 1:
                        self._diversity_thresholds[c] = np.percentile(real_real_dist[mask], 5)
                    else:
                        self._diversity_thresholds[c] = 0.02
            
            # C-01: After computing thresholds, clear the stale flag
            self._diversity_stale = False
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_real)
        dist_syn_real, _ = nn.kneighbors(X_syn)
        norm_dist = dist_syn_real.flatten() / np.sqrt(X_syn.shape[1])
        
        n_valid = min(len(df), len(X_syn))
        keep_mask = np.zeros(n_valid, dtype=bool)
        for i in range(n_valid):
            if self.is_regression:
                t = int(self._get_bin(y_syn[i]))
            else:
                t = int(y_syn[i])
            thresh = self._diversity_thresholds.get(t, 0.05)
            if norm_dist[i] > thresh:
                keep_mask[i] = True

        passed = df.iloc[:n_valid].iloc[keep_mask].reset_index(drop=True)
        
        if len(passed) > 1:
            encoded_passed = self._encode_data(passed)
            if len(encoded_passed) > 1:
                X_passed = encoded_passed[self.columns].values
                n_dup = len(X_passed)
                nn_syn = NearestNeighbors(n_neighbors=min(2, n_dup))
                nn_syn.fit(X_passed)
                dist_syn_syn, _ = nn_syn.kneighbors(X_passed)
                if dist_syn_syn.shape[1] >= 2:
                    dup_mask = dist_syn_syn[:, 1] / np.sqrt(X_passed.shape[1]) > 0.01
                    passed = passed.iloc[:n_dup].iloc[dup_mask].reset_index(drop=True)
        
        return passed
    
    def _encode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码 DataFrame"""
        encoded = df.copy()
        
        if not self.is_regression and self.label_encoder is not None and self.target_col in encoded.columns:
            valid = encoded[self.target_col].isin(self.label_encoder.classes_)
            encoded = encoded[valid]
            if len(encoded) > 0:
                encoded[self.target_col] = self.label_encoder.transform(encoded[self.target_col])
        
        if self.is_regression and self.target_col in encoded.columns:
            encoded[self.target_col] = pd.to_numeric(encoded[self.target_col], errors='coerce')
        
        for col in self.columns:
            if col in encoded.columns:
                if col in self.feature_encoders:
                    le = self.feature_encoders[col]
                    valid = encoded[col].astype(str).isin(le.classes_)
                    encoded = encoded[valid]
                    if len(encoded) > 0:
                        encoded[col] = le.transform(encoded[col].astype(str)).astype('float64')
                else:
                    encoded[col] = pd.to_numeric(encoded[col], errors='coerce').astype('float64')
        
        return encoded.dropna().reset_index(drop=True)
    
    def _compute_reward(self, batch: pd.DataFrame, splits: List = None, M: int = 5, top_ratio: float = 0.2) -> Tuple[float, float]:
        """计算reward"""
        if self.is_regression:
            return self._compute_reward_regression(batch, splits, M, top_ratio)
        else:
            return self._compute_reward_classification(batch, splits, M, top_ratio)
    
    def _compute_reward_classification(self, batch: pd.DataFrame, splits: List = None, M: int = 5, top_entropy_ratio: float = 0.2) -> Tuple[float, float]:
        """分类：高熵query子集上的ΔNLL"""
        if splits is None:
            splits = []
            n_sup = int(len(self.D_real) * 0.8)
            for _ in range(M):
                idx = np.random.permutation(len(self.D_real))
                splits.append((idx[:n_sup], idx[n_sup:]))
        
        encoded = self._encode_data(batch)
        if len(encoded) == 0:
            return 0.0, 0.0
        
        X_batch = encoded[self.columns].values
        y_batch = encoded[self.target_col].values
        X_real = self.D_real[self.columns].values
        y_real = self.D_real[self.target_col].values
        
        delta_Us = []
        for sup_idx, qry_idx in splits:
            X_sup, y_sup = X_real[sup_idx], y_real[sup_idx]
            X_qry, y_qry = X_real[qry_idx], y_real[qry_idx]
            
            try:
                self.tabpfn.fit(X_sup, y_sup)
                probs_0 = self.tabpfn.predict_proba(X_qry)
                
                entropy = -np.sum(probs_0 * np.log(probs_0.clip(1e-10, 1)), axis=1)
                n_hi = max(1, int(len(entropy) * top_entropy_ratio))
                hi_idx = np.argsort(entropy)[-n_hi:]
                
                U0 = np.mean(np.log(probs_0[hi_idx, y_qry[hi_idx].astype(int)].clip(1e-10, 1)))
                
                self.tabpfn.fit(np.vstack([X_sup, X_batch]), np.concatenate([y_sup, y_batch]))
                probs_B = self.tabpfn.predict_proba(X_qry)
                U_B = np.mean(np.log(probs_B[hi_idx, y_qry[hi_idx].astype(int)].clip(1e-10, 1)))
                
                delta_Us.append(U_B - U0)
            except:
                delta_Us.append(0.0)
        
        self._fit_tabpfn()
        return float(np.mean(delta_Us)), float(np.mean(delta_Us))
    
    def _compute_reward_regression(self, batch: pd.DataFrame, splits: List = None, M: int = 5, top_var_ratio: float = 0.2) -> Tuple[float, float]:
        """回归：高不确定query上的Δ(-MSE)"""
        if splits is None:
            splits = []
            n_sup = int(len(self.D_real) * 0.8)
            for _ in range(M):
                idx = np.random.permutation(len(self.D_real))
                splits.append((idx[:n_sup], idx[n_sup:]))
        
        encoded = self._encode_data(batch)
        if len(encoded) == 0:
            return 0.0, 0.0
        
        X_batch = encoded[self.columns].values
        y_batch = encoded[self.target_col].values.astype(float)
        X_real = self.D_real[self.columns].values
        y_real = self.D_real[self.target_col].values.astype(float)
        
        delta_Us = []
        for sup_idx, qry_idx in splits:
            X_sup, y_sup = X_real[sup_idx], y_real[sup_idx]
            X_qry, y_qry = X_real[qry_idx], y_real[qry_idx]
            
            try:
                self.tabpfn.fit(X_sup, y_sup)
                preds_0 = self.tabpfn.predict(X_qry)
                residuals_0 = np.abs(y_qry - preds_0)
                
                # 选高残差query（模型不确定/错误的地方）
                n_hi = max(1, int(len(residuals_0) * top_var_ratio))
                hi_idx = np.argsort(residuals_0)[-n_hi:]
                
                # Baseline: 负MSE
                U0 = -np.mean((y_qry[hi_idx] - preds_0[hi_idx]) ** 2)
                
                self.tabpfn.fit(np.vstack([X_sup, X_batch]), np.concatenate([y_sup, y_batch]))
                preds_B = self.tabpfn.predict(X_qry)
                U_B = -np.mean((y_qry[hi_idx] - preds_B[hi_idx]) ** 2)
                
                delta_Us.append(U_B - U0)
            except:
                delta_Us.append(0.0)
        
        self._fit_tabpfn()
        return float(np.mean(delta_Us)), float(np.mean(delta_Us))
    
    def _update_diversity(self):
        """更新多样性指标"""
        if len(self.synthetic_buffer) == 0:
            self.diversity_score = 0.0
            return
        
        encoded = self._encode_data(self.synthetic_buffer)
        if len(encoded) == 0:
            self.diversity_score = 0.0
            return
        
        X_syn = encoded[self.columns].values
        X_real = self.D_real[self.columns].values
        
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(X_real)
        dist, _ = nn.kneighbors(X_syn)
        self.diversity_score = float(dist.mean() / np.sqrt(X_syn.shape[1]))
    
    def commit_top_proposals(self, top_l: int = 20) -> int:
        """按IG排序选top-L proposals，轻量去重后提交"""
        if len(self.proposals) == 0:
            return 0
        
        sorted_props = sorted(self.proposals, key=lambda x: x["ig"], reverse=True)
        top_props = sorted_props[:top_l]
        
        all_batches = [p["batch"] for p in top_props if len(p["batch"]) > 0]
        if len(all_batches) == 0:
            self.proposals = []
            return 0
        
        merged = pd.concat(all_batches, ignore_index=True)
        
        if len(merged) > 1:
            encoded = self._encode_data(merged)
            if len(encoded) > 1:
                X = encoded[self.columns].values
                n_enc = len(X)
                nn = NearestNeighbors(n_neighbors=min(2, n_enc))
                nn.fit(X)
                dist, _ = nn.kneighbors(X)
                if dist.shape[1] > 1:
                    keep = dist[:, 1] / np.sqrt(X.shape[1]) > 0.01
                    merged = merged.iloc[:n_enc].iloc[keep].reset_index(drop=True)
        
        committed = len(merged)
        if committed > 0:
            self.synthetic_buffer = pd.concat([
                self.synthetic_buffer, merged
            ], ignore_index=True)
            self._update_diversity()
        
        self.proposals = []
        
        # C-01: Mark diversity thresholds as stale after committing new samples
        if committed > 0:
            new_buffer_size = len(self.synthetic_buffer)
            if new_buffer_size >= self._last_diversity_buffer_size * 1.5:
                self._diversity_stale = True
                self._last_diversity_buffer_size = new_buffer_size
        
        return committed
    
    def get_synthetic_data(self) -> pd.DataFrame:
        """获取合成数据"""
        return self.synthetic_buffer.copy()
    
    def decode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """解码数据"""
        if df is None or len(df) == 0:
            return df
        
        decoded = df.copy()
        
        if not self.is_regression and self.label_encoder is not None and self.target_col in decoded.columns:
            try:
                labels = decoded[self.target_col].values.astype(int)
                valid = (labels >= 0) & (labels < len(self.label_encoder.classes_))
                decoded = decoded[valid].copy()
                if len(decoded) > 0:
                    decoded[self.target_col] = self.label_encoder.inverse_transform(
                        decoded[self.target_col].values.astype(int)
                    )
            except:
                pass
        
        for col, encoder in self.feature_encoders.items():
            if col in decoded.columns:
                try:
                    values = np.clip(decoded[col].values.astype(int), 0, len(encoder.classes_) - 1)
                    decoded[col] = encoder.inverse_transform(values)
                except:
                    pass
        
        return decoded.reset_index(drop=True)
