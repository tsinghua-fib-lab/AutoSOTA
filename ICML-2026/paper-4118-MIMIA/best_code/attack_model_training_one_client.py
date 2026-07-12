import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import false
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
from attack_models import SimpleMIA,GapGatedMIA,CrossAttnGapMIA,AffineGapMIA
from crient_function import *
import time
# ==========================================
#              1. 配置区域
# ==========================================
def check_nan(df, name):
    """诊断 DataFrame 中的 NaN"""
    print(f"\n{'='*50}")
    print(f"🔍 NaN 诊断: {name}  (shape={df.shape})")
    print(f"{'='*50}")

    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if len(nan_cols) == 0:
        print("✅ 无 NaN")
    else:
        print(f"❌ 发现 {len(nan_cols)} 列含 NaN:")
        for col, cnt in nan_cols.items():
            pct = cnt / len(df) * 100
            print(f"   [{col}]  →  {cnt} 行 ({pct:.1f}%)")

        # 定位 NaN 所在行号
        nan_rows = df[df.isnull().any(axis=1)].index.tolist()
        print(f"\n   含 NaN 的行索引 (前20): {nan_rows[:20]}")

        # 检查是否是标签映射失败导致的
        if 'Sample_Type' in df.columns:
            unique_labels = df['Sample_Type'].unique()
            print(f"\n   Sample_Type 原始值: {unique_labels}")

    return nan_cols
# 基础路径配置
SAVE_DIR = r"E:\machang\code\FedMIA-main\FedMIA-main\saved_mia_models\cremad_K6_N1100_AVClassifer_defnone_iid$1_$1.0_$sgd_local3_s42_mod$multimodal_random\train_excel"
EPCH = 60
MODE = "mix"

# [关键 1] 合谋用户列表
# 逻辑：代码会自动轮询，例如 [3, 4, 5] 会生成文件名：
# 3_[4, 5], 4_[3, 5], 5_[3, 4] 并将它们合并作为训练集
COLLUDING_USERS = [3,4,5,0,1,2]

# 测试目标 (攻击对象)
TEST_RUN = {'target': 0, 'val': [1,2,3,4]}
# COLLUDING_USERS = [0,1,2]
#
# # 测试目标 (攻击对象)
# TEST_RUN = {'target': 5, 'val': [3,4]}
#
# COLLUDING_USERS = [5,6,7,8,9]
#
# # 测试目标 (攻击对象)
# TEST_RUN = {'target': 0, 'val': [1,2,3,4]}


# COLLUDING_USERS = [10,11,12,13,14,15,16,17,18,19]
#
# # 测试目标 (攻击对象)
# TEST_RUN = {'target': 0, 'val': [1,2,3,4,5,6,7,8,9]}


FEATURE_COLUMNS = [
    'LOSS_full_Score',
    'LOSS_audio_Score',
    'LOSS_visual_Score',
    'COS_full_Score',
    'COS_audio_Score',
    'COS_visual_Score',
    'COSINE_ATTACK_full_Score',
    'COSINE_ATTACK_audio_Score',
    'COSINE_ATTACK_visual_Score',
    'GRAD_DIFF_full_Score',
    'GRAD_DIFF_audio_Score',
    'GRAD_DIFF_visual_Score',
    'LOSS_BASED_full_Score',
    'LOSS_BASED_audio_Score',
    'LOSS_BASED_visual_Score',
    'GRAD_NORM_full_Score',
    'GRAD_NORM_audio_Score',
    'GRAD_NORM_visual_Score'
]

NEW_FEATURE_NAME = 'Diff_Audio_Visual_Score'

# 标签配置
LABEL_COLUMN = 'Sample_Type'
LABEL_MAPPING = {'Member': 1, 'Non_Member': 0}

# 模型超参数
BATCH_SIZE = 1024
LEARNING_RATE = 1e-6
EPOCHS = 20
SEED = 534563

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==========================================
#              2. PyTorch 模型定义
# ==========================================

def load_and_merge_data():
    print(">>> [1/3] 读取与合并合谋用户数据 (One-vs-Rest 策略)...")

    train_dfs = []

    # --- 遍历每一个用户做 Target，其余做 Val ---
    for target_id in COLLUDING_USERS:
        # 1. 获取"剩下"的所有用户 ID
        val_ids = [u for u in COLLUDING_USERS if u != target_id]

        # 2. 转换成字符串格式 (例如 "[3, 4]")
        val_str = str(val_ids)

        # 3. 构建文件名
        file_name = f"training_construct_{target_id}_{val_str}_{EPCH}_{MODE}_lira_opti_one_attack_client.xlsx"
        file_path = os.path.join(SAVE_DIR, file_name)

        # 4. 读取逻辑
        if os.path.exists(file_path):
            try:
                print(f"   -> 正在读取: {file_name}")
                df_temp = pd.read_excel(file_path)
                check_nan(df_temp, "df_train (合并后, 原始)")
                train_dfs.append(df_temp)
            except Exception as e:
                print(f"❌ 读取错误 {file_path}: {e}")
        else:
            print(f"⚠️ 警告: 文件不存在 (跳过): {file_name}")
            # 备用尝试：防止文件名没有空格 (例如 [3,4])
            val_str_nospace = str(val_ids).replace(" ", "")
            file_name_nospace = f"training_construct_{target_id}_{val_str_nospace}_{EPCH}_{MODE}.xlsx"
            file_path_nospace = os.path.join(SAVE_DIR, file_name_nospace)
            if os.path.exists(file_path_nospace):
                print(f"   -> (无空格重试) 正在读取: {file_name_nospace}")
                df_temp = pd.read_excel(file_path_nospace)
                train_dfs.append(df_temp)

    if not train_dfs:
        print("❌ 错误: 没有成功加载任何训练数据！")
        return None, None, None, None, None

    # 合并
    print("\n🔍 检查各文件列名一致性:")
    all_columns = [set(df.columns) for df in train_dfs]
    base_cols = all_columns[0]

    for i, cols in enumerate(all_columns[1:], 1):
        diff_a = base_cols - cols  # 第0个文件有、第i个没有
        diff_b = cols - base_cols  # 第i个文件有、第0个没有
        if diff_a or diff_b:
            print(f"❌ 第 {i} 个文件列名不一致!")
            if diff_a:
                print(f"   缺少列: {diff_a}")
            if diff_b:
                print(f"   多余列: {diff_b}")
        else:
            print(f"✅ 第 {i} 个文件列名一致")
    df_train = pd.concat(train_dfs, ignore_index=True)

    df_train = shuffle(df_train, random_state=SEED).reset_index(drop=True)
    print(f"   >>> 训练集构建完成，共 {len(df_train)} 条样本")

    # --- 读取测试集 ---
    test_val_str = str(TEST_RUN['val'])
    f_test = f"{SAVE_DIR}/training_construct_{TEST_RUN['target']}_{test_val_str}_{EPCH}_{MODE}_lira_opti_one_attack_client_test.xlsx"

    try:
        print(f"   -> 正在读取测试集: {os.path.basename(f_test)}")
        if not os.path.exists(f_test):
            raise FileNotFoundError(f"找不到测试文件: {f_test}")
        df_test = pd.read_excel(f_test)
    except Exception as e:
        print(f"❌ 测试集读取错误: {e}")
        return None, None, None, None, None

    # --- 特征工程 ---
    print(">>> [2/3] 计算 Diff 特征...")
    if 'LOSS_BASED_audio_Score' in df_train.columns and 'LOSS_BASED_visual_Score' in df_train.columns:
        df_train[NEW_FEATURE_NAME] = df_train['LOSS_BASED_audio_Score'] - df_train['LOSS_BASED_visual_Score']
        df_test[NEW_FEATURE_NAME] = df_test['LOSS_BASED_audio_Score'] - df_test['LOSS_BASED_visual_Score']

    FINAL_FEATURE_LIST = FEATURE_COLUMNS + [NEW_FEATURE_NAME]

    # --- 标签处理 ---
    print(">>> [3/3] 转换标签...")

    def process_labels(df):
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().map(LABEL_MAPPING)
        return df.dropna(subset=[LABEL_COLUMN])

    df_train = process_labels(df_train)
    df_test = process_labels(df_test)

    # 提取数据
    X_train = df_train[FINAL_FEATURE_LIST].values.astype(np.float32)
    y_train = df_train[LABEL_COLUMN].values.astype(np.float32)
    X_test = df_test[FINAL_FEATURE_LIST].values.astype(np.float32)
    y_test = df_test[LABEL_COLUMN].values.astype(np.float32)

    return X_train, y_train, X_test, y_test, FINAL_FEATURE_LIST


# ==========================================
#              4. 主程序
# ==========================================

if __name__ == "__main__":
    # ==========================================
    # [新增配置] 训练模式开关
    # ==========================================
    IS_TRAIN_MODE = False  # <--- True: 训练并保存; False: 直接读取模型评估

    # 定义模型保存路径


    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"当前模式: {'🏋️ 训练模式 (Training)' if IS_TRAIN_MODE else '⚡ 推理模式 (Inference Only)'}")

    # 1. 获取数据
    X_train, y_train, X_test, y_test, feature_names = load_and_merge_data()

    if X_train is not None:
        print("\n" + "=" * 50)
        print(" 数据预处理 ".center(50, "="))

        # 2. 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train).to(device)
        y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(device)
        X_test_tensor = torch.tensor(X_test).to(device)
        y_test_tensor = torch.tensor(y_test).unsqueeze(1).to(device)

        # 3. 模型初始化
        input_dim = X_train.shape[1]
        model = CrossAttnGapMIA(input_dim, temperature=0.5).to(device)
        # model = GapGatedMIA(input_dim).to(device)
        # model = AffineGapMIA(input_dim, temperature=0.5).to(device)
        model_name = model.__class__.__name__  # 自动获取类名，如 "AffineGapMIA"
        loss_name = "Asymmetric"
        # loss_name = "BCE"
        MODEL_FILENAME = f"best_model_{model_name}_{loss_name}_{TEST_RUN['target']}_{MODE}_dim{input_dim}_lira_opti_one_attack_client.pth"
        BEST_MODEL_PATH = os.path.join(SAVE_DIR, MODEL_FILENAME)
        # =========================================================
        #    分支 A: 训练模式 (Training Mode)
        # =========================================================
        if IS_TRAIN_MODE:
            print("\n" + "=" * 50)
            print(" 开始训练流程 ".center(50, "="))
            print("=" * 50)

            # 准备训练组件
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # 计算类别权重
            num_neg = np.sum(y_train == 0)
            num_pos = np.sum(y_train == 1)
            penalty_factor = num_pos/num_neg # 建议尝试 0.1, 0.2, 0.5
            basic_weight = num_neg / num_pos if num_pos > 0 else 1.0
            pos_weight_value = basic_weight * penalty_factor
            pos_weight = torch.tensor([pos_weight_value]).to(device)
            print(f"⚖️  正样本权重: {pos_weight_value:.4f}")
            # criterion = FocalLoss(alpha=0.1, gamma=3.0).to(device)
            criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05,reduction='sum').to(device)

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.01, epochs=EPOCHS, steps_per_epoch=len(train_loader)
            )

            # 训练循环
            best_val_auc = 0.0
            print(f"开始训练 (Epochs: {EPOCHS})...")
            model.train()

            for epoch in range(EPOCHS):
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()

                    # [关键修正] 训练时默认不返回 attention，所以只接收一个值
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()

                # 每 10 轮验证
                if (epoch + 1) % 1 == 0:
                    model.eval()
                    with torch.no_grad():
                        # [关键修正] 验证时也只接收一个值
                        test_logits = model(X_test_tensor)

                        y_prob = torch.sigmoid(test_logits).cpu().numpy().ravel()
                        y_true_np = y_test_tensor.cpu().numpy().ravel()
                        try:
                            fpr, tpr, _ = roc_curve(y_true_np, y_prob)
                            roc_auc = auc(fpr, tpr)
                        except:
                            roc_auc = 0.0

                    print(
                        f"Ep [{epoch + 1}/{EPOCHS}] | Loss: {epoch_loss / len(train_loader):.4f} | Test AUC: {roc_auc:.4f}")

                    # 保存最佳模型
                    if roc_auc > best_val_auc:
                        best_val_auc = roc_auc
                        torch.save(model.state_dict(), BEST_MODEL_PATH)
                        print(f"   💾 发现新高 AUC，模型已保存至: {os.path.basename(BEST_MODEL_PATH)}")

                    model.train()

            print(f"\n✅ 训练结束。历史最佳 AUC: {best_val_auc:.4f}")

        # =========================================================
        #    分支 B: 推理模式 (Inference Mode)
        # =========================================================
        else:
            print("\n" + "=" * 50)
            print(" 加载模型进行推理 ".center(50, "="))
            print("=" * 50)

            if os.path.exists(BEST_MODEL_PATH):
                model.load_state_dict(torch.load(BEST_MODEL_PATH))
                print(f"📂 成功加载已保存的最佳模型: {BEST_MODEL_PATH}")
            else:
                raise FileNotFoundError(f"❌ 未找到模型文件: {BEST_MODEL_PATH}。请先设置 IS_TRAIN_MODE=True 进行训练！")

        # =========================================================
        #    通用评估流程 (最终评估)
        # =========================================================
        model.eval()
        train_start_time = time.time()  # 加这一行

        with torch.no_grad():
            # [关键修正] 最终评估时也只接收一个值
            test_logits = model(X_test_tensor)

            y_prob = torch.sigmoid(test_logits).cpu().numpy().ravel()
            y_true = y_test_tensor.cpu().numpy().ravel()
        train_elapsed = time.time() - train_start_time  # 加这一行
        print(f"⏱️ 训练总耗时: {train_elapsed:.2f} 秒 ({train_elapsed / 60:.2f} 分钟)")  # 加这一行
        # 1. 基础 ROC 计算
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # 2. 寻找最佳阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_prob > optimal_threshold).astype(int)

        print("\n" + "-" * 30)
        print("        最终评估结果        ")
        print("-" * 30)
        print(f"AUC 值      : {roc_auc:.4f}")
        print(f"最佳阈值    : {optimal_threshold:.4f}")
        print(f"准确率 (Acc): {accuracy_score(y_true, y_pred):.4f}")

        print("\n详细报告:\n", classification_report(y_true, y_pred, target_names=['Non-Member', 'Member']))

        # 3. 详细混淆矩阵指标
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        mem_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        non_mem_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"✅ Member 识别准确率    : {mem_acc:.2%}")
        print(f"🛡️ Non-Member 排除准确率: {non_mem_acc:.2%}")


        # 4. Low FPR 下的 TPR
        def get_tpr_at_fpr(target_fpr, fpr_arr, tpr_arr):
            idx = np.where(fpr_arr <= target_fpr)[0]
            return tpr_arr[idx[-1]] if len(idx) > 0 else 0.0


        tpr_1_percent = get_tpr_at_fpr(0.01, fpr, tpr)
        tpr_01_percent = get_tpr_at_fpr(0.001, fpr, tpr)
        print(f"\n🌟 高置信度指标:")
        print(f"TPR @ FPR=1%   : {tpr_1_percent:.4f}")
        print(f"TPR @ FPR=0.1% : {tpr_01_percent:.4f}")

        # 5. 模态偏好细分分析

        print("\n" + "=" * 40)
        print(" 🏋️ 训练集性能回测 (Training Check) ".center(40, "="))

        model.eval()
        with torch.no_grad():
            # 1. 获取训练集预测 logits
            # 注意：需处理模型可能返回 (logits, attention) 的情况
            train_out = model(X_train_tensor)
            if isinstance(train_out, tuple):
                train_logits = train_out[0]
            else:
                train_logits = train_out

            # 2. 转为概率
            train_probs = torch.sigmoid(train_logits).cpu().numpy().ravel()

        # 3. 使用测试集找到的最佳阈值进行二分类 (这样对比才公平)
        # (确保 optimal_threshold 变量在之前的代码中已定义)
        train_preds = (train_probs > optimal_threshold).astype(int)

        # 4. 计算指标
        # 总体准确率
        train_acc = accuracy_score(y_train, train_preds)

        # Member (y=1) 识别率
        train_mask_mem = (y_train == 1)
        train_acc_mem = accuracy_score(y_train[train_mask_mem], train_preds[train_mask_mem]) if np.sum(
            train_mask_mem) > 0 else 0.0

        # Non-Member (y=0) 识别率
        train_mask_non = (y_train == 0)
        train_acc_non = accuracy_score(y_train[train_mask_non], train_preds[train_mask_non]) if np.sum(
            train_mask_non) > 0 else 0.0

        # 均衡准确率
        train_balanced_acc = (train_acc_mem + train_acc_non) / 2

        print(f"📊 [训练集表现]")
        print(f"   - 🏆 训练集准确率 (Train Acc)      : {train_acc:.2%}")
        print(f"   - ⚖️  训练集均衡准确率 (Balanced)    : {train_balanced_acc:.2%}")
        print(f"   - Member 识别率 (Recall)           : {train_acc_mem:.2%}")
        print(f"   - Non-Mem 识别率 (TN Rate)         : {train_acc_non:.2%}")
        print(f"   (注: 使用与测试集相同的最佳阈值: {optimal_threshold:.4f})")
        print("-" * 40)



        # 5. 模态偏好细分分析
        print("\n" + "=" * 40)
        print(" 🔍 模态偏好细分分析 (基于原始 Loss) ".center(40, "="))

        # ================== [新增部分开始] ==================
        # 1. 计算全局总准确率
        total_acc = accuracy_score(y_true, y_pred)

        # 2. 计算全局 Member 和 Non-Member 的各自准确率
        mask_all_mem = (y_true == 1)
        mask_all_non = (y_true == 0)

        total_acc_mem = accuracy_score(y_true[mask_all_mem], y_pred[mask_all_mem]) if np.sum(mask_all_mem) > 0 else 0.0
        total_acc_non = accuracy_score(y_true[mask_all_non], y_pred[mask_all_non]) if np.sum(mask_all_non) > 0 else 0.0

        # 3. [新增] 计算全局均衡准确率
        total_balanced_acc = (total_acc_mem + total_acc_non) / 2

        print(f"\n📊 [总体性能汇总 Total Performance]")
        print(f"   - 🏆 全局准确率 (Overall Acc) : {total_acc:.2%}")
        print(f"   - ⚖️  均衡准确率 (Balanced Acc): {total_balanced_acc:.2%}")  # <--- 新增
        print(f"   - Member 识别率 (Recall)      : {total_acc_mem:.2%}")
        print(f"   - Non-Mem 识别率 (TN Rate)    : {total_acc_non:.2%}")
        print("-" * 40)
        print("\n" + "#" * 60)
        print(" " * 15 + "📌 关键实验结果汇总 (SUMMARY) 📌")
        print("#" * 60)

        print(f"📂 模型文件名称: {MODEL_FILENAME}")
        print("-" * 60)

        print(f"🌟 高置信度指标 (High Confidence Metrics):")
        print(f"   ► TPR @ FPR=1%   : {tpr_1_percent:.4f}")
        print(f"   ► TPR @ FPR=0.1% : {tpr_01_percent:.4f}")
        print("-" * 60)

        print(f"📈 核心性能指标 (Core Performance):")
        print(f"   ► 测试集 AUC     : {roc_auc:.4f}")
        print(f"   ► 训练集均衡准确率 : {train_balanced_acc:.2%}")
        print(f"   ► 全局均衡准确率   : {total_balanced_acc:.2%}")
        # =========================================================
        # [新增] 低 FPR 下的模态偏好样本识别统计
        # =========================================================
        print("\n" + "-" * 30)
        print("   低 FPR 下各模态识别统计    ")
        print("-" * 30)

        # 1. 准备模态偏好掩码 (基于原始数据)
        X_test_raw = scaler.inverse_transform(X_test)
        idx_a_score = feature_names.index('LOSS_BASED_audio_Score')
        idx_v_score = feature_names.index('LOSS_BASED_visual_Score')

        # 定义测试集中所有 Member 的偏好属性
        m_mask_a = (X_test_raw[:, idx_a_score] > X_test_raw[:, idx_v_score])
        m_mask_v = ~m_mask_a

        # 统计测试集总共有的各偏好 Member 数量
        total_m_a = np.sum((y_true == 1) & m_mask_a)
        total_m_v = np.sum((y_true == 1) & m_mask_v)


        def print_detections_at_fpr(target_fpr, name):
            # 找到最接近目标 FPR 的阈值索引
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) == 0: return

            # 获取该 FPR 下的判定阈值
            current_thresh = thresholds[idx[-1]]

            # 执行预测：概率 > 阈值 则判定为 Member
            y_pred_at_fpr = (y_prob >= current_thresh).astype(int)

            # 筛选出真正识别出的 Member (True Positives)
            tp_indices = (y_true == 1) & (y_pred_at_fpr == 1)

            # 统计不同偏好的识别数
            det_a = np.sum(tp_indices & m_mask_a)
            det_v = np.sum(tp_indices & m_mask_v)

            print(f"🌟 FPR @ {name}:")
            print(f"   - 总识别 Member: {np.sum(tp_indices)}")
            print(f"   - Audio-Pref 识别: {det_a} / {total_m_a} (召回率: {det_a / total_m_a:.2%})")
            print(f"   - Visual-Pref 识别: {det_v} / {total_m_v} (召回率: {det_v / total_m_v:.2%})")


        # 执行统计
        print_detections_at_fpr(0.01, "1%")
        print_detections_at_fpr(0.001, "0.1%")
        print("-" * 30)