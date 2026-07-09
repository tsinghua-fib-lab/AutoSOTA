import os
import sys
print("PYTHON:", sys.executable)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Subset
import random
import torch.nn.functional as F
from dataset.dataset import CompositionDataset
from model.model_mine import BandModelSE, combined_vae_evidential_loss_SE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from scipy.stats import kendalltau
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(1023)


def train_one_epoch(model, loader, optimizer):
    model.train()
    epoch_loss = 0
    epoch_mae = 0
    epoch_loss_rank = 0
    for batch in loader:
        x_comp = batch["x_comp"].to(device)
        x_total = batch["x_total_feats"].to(device)
        x_state = batch["state"].to(device)
        y = batch["y_bandgap"].to(device)
        optimizer.zero_grad()
        loss, loss_dict = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_loss_rank += loss_dict["rank_loss"]
        epoch_mae  += loss_dict["mae"]

    avg_loss = epoch_loss / len(loader)
    avg_mae  = epoch_mae  / len(loader)
    avg_rank_loss = epoch_loss_rank/ len(loader)
    return avg_loss, avg_mae,avg_rank_loss



def eval_one_epoch(model, loader):
    model.eval()
    epoch_loss = 0
    epoch_mae = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in loader:
            x_comp = batch["x_comp"].to(device)
            x_total = batch["x_total_feats"].to(device)
            x_state = batch["state"].to(device)
            y = batch["y_bandgap"].to(device)
            loss, loss_dict = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y)
            epoch_loss += loss.item()
            epoch_mae  += loss_dict["mae"]
            pred = loss_dict["pred"]
            all_preds.append(pred.detach().cpu().ravel())  
            all_trues.append(y.detach().cpu().ravel()) 
    avg_loss = epoch_loss / len(loader)
    avg_mae  = epoch_mae  / len(loader)
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    return avg_loss, avg_mae,all_preds,all_trues


def train_model( model=None, epochs=30, train_loader=None, val_loader=None,lr=1e-3,test_loader=None,  save_path="./pt/best_model.pt", patience=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\n========= Epoch {epoch} =========")
        train_loss, train_mae,avg_rank_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_mae, all_preds, all_trues = eval_one_epoch(model, val_loader)
        tau, _ = kendalltau(all_trues, all_preds)

        print(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f}| Train avg_rank_loss: {avg_rank_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   MAE: {val_mae:.4f} | Kendall Tau = {tau:.4f}", flush=True)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model (val MAE={val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered (patience={patience})")
                break
        if test_loader != None:
            test_model(model,test_loader)
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model




def test_model(model, test_loader):
    model.eval()
    test_loss, test_mae, all_preds, all_trues = eval_one_epoch(model, test_loader)
    tau, _ = kendalltau(all_trues, all_preds)
    print(f"\n===== Test Loss = {test_loss:.4f} | Test MAE = {test_mae:.4f} | Kendall Tau = {tau:.4f}", flush=True)
    
    return test_loss, test_mae, tau

def main():
    files = ["exp.csv", "gllb-sc.csv", "hse.csv", "scan.csv", "pbe.csv"]
    states = [0, 1, 2, 3, 4]  # 从高到低

    # 2️⃣ 读取数据并添加 state
    datasets = []
    for f, s in zip(files, states):
        df = pd.read_csv(f"./data/{f}")
        df = df.dropna(subset=['formula'])
        df["state"] = s
        print(f"数据集 {f} | state={s} | 样本量={len(df)}", flush=True)  # 打印每个数据集信息
        dataset = CompositionDataset(df, "formula", "band_gap", "state")
        datasets.append(dataset)

    # 3️⃣ 按8:1:1划分每个数据集
    split_datasets = {}
    for state, ds in zip(states, datasets):
        total_len = len(ds)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len
        train_ds, val_ds, test_ds = random_split(ds, [train_len, val_len, test_len])
        split_datasets[state] = {"train": train_ds, "val": val_ds, "test": test_ds}
        print(f"state={state} 划分: train={train_len}, val={val_len}, test={test_len}", flush=True)  # 划分信息

    # 4️⃣ 实验组合定义 (多保真度训练集)
    train_combinations = [
        [0],               # 1-fi
        [0, 4],            # 2-fi
        [0, 2, 4],         # 3-fi
        [0, 2, 3, 4],     # 4-fi: exp, scan, hse, pbe
        [0, 1, 2, 3, 4]    # 5-fi
    ]

    # 5️⃣ 测试集顺序固定
    test_order = [0, 1, 2, 3, 4]
    batch_size = 1024

    results = {}  # 保存每次实验的测试结果

    from torch.utils.data import DataLoader
    from torch.utils.data import ConcatDataset
    for i, combo in enumerate(train_combinations, 1):
        print(f"\n===== 开始第 {i}-fi 实验 =====", flush=True)
        combo_names = [files[s].split(".")[0] for s in combo]
        print(f"训练集组合: {combo_names}")

        # 合并训练集和验证集
        train_datasets = [split_datasets[s]["train"] for s in combo]
        val_datasets = [split_datasets[s]["val"] for s in combo]
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)

        print(f"总训练样本量: {len(train_dataset)}, 总验证样本量: {len(val_dataset)}", flush=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = BandModelSE().to(device)

        print("开始训练模型...")
        best_model = train_model(
            model,
            epochs=200,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-3,
            test_loader=None
        )
        print("训练完成！")

        # 在5个保真度测试集上测试
        results[i] = {}
        for test_state in test_order:
            test_loader = DataLoader(split_datasets[test_state]["test"], batch_size=batch_size, shuffle=False)
            test_score = test_model(best_model, test_loader)  # 返回 (loss, mae, tau)
            
            # 格式化打印
            score_str = ", ".join([f"{x:.4f}" for x in test_score])
            results[i][test_state] = test_score
            
            print(f"测试集 {files[test_state]} | 样本量={len(split_datasets[test_state]['test'])} | 得分=({score_str})", flush=True)


    # 打印5x5矩阵结果
    results_df = pd.DataFrame(results).T  # 行: 实验行, 列: 测试集
    results_df.columns = ["exp", "gllb-sc", "hse", "scan", "pbe"]
    print("\n===== 最终5x5测试结果矩阵 =====")
    pd.set_option("display.max_columns", None)      # 显示所有列
    pd.set_option("display.max_colwidth", None)     # 单元格不省略
    pd.set_option("display.width", 200)    
    print(results_df)
    results_df.to_csv("./results/fi_5x5_results.csv", index=True)
    print("已保存结果到 ./results/fi_5x5_results.csv", flush=True)



if __name__ == "__main__":
    main()
