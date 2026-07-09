import os, sys, numpy as np, torch, torch.nn.functional as F, time
from sklearn.ensemble import RandomForestClassifier
from mantis.architecture import MantisV1
from mantis.trainer import MantisTrainer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

dataset_path = "/datasets/ucr/Adiac"
train_file = dataset_path + "/Adiac_TRAIN.ts"
test_file = dataset_path + "/Adiac_TEST.ts"

def parse_ts(filepath):
    data, labels = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.rsplit(":", 1)
            values = np.array([float(v) for v in parts[0].split(",")], dtype=np.float32)
            data.append(values)
            labels.append(float(parts[1]))
    return np.array(data), np.array(labels, dtype=np.int64)

X_train, y_train = parse_ts(train_file)
X_test, y_test = parse_ts(test_file)
all_labels = np.unique(np.concatenate([y_train, y_test]))
label_map = {lbl: i for i, lbl in enumerate(all_labels)}
y_train = np.array([label_map[l] for l in y_train], dtype=np.int64)
y_test = np.array([label_map[l] for l in y_test], dtype=np.int64)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, n_classes: {len(all_labels)}")

X_train_rs = F.interpolate(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), size=512, mode="linear", align_corners=False).numpy()
X_test_rs = F.interpolate(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1), size=512, mode="linear", align_corners=False).numpy()
print(f"X_train_rs: {X_train_rs.shape}")

print("Loading model...")
network = MantisV1(device="cuda", output_token="combined", return_transf_layer=2)
network = network.from_pretrained("paris-noah/Mantis-8M")
print(f"hidden_dim: {network.hidden_dim}")
model = MantisTrainer(device="cuda", network=network)

print("Extracting features...")
t0 = time.time()
Z_train = model.transform(X_train_rs, batch_size=256)
Z_test = model.transform(X_test_rs, batch_size=256)
print(f"Z_train: {Z_train.shape}, Z_test: {Z_test.shape} (took {time.time()-t0:.1f}s)")

print("Training RF...")
clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=0)
clf.fit(Z_train, y_train)
y_pred = clf.predict(Z_test)
acc = np.mean(y_test == y_pred)
print(f"Adiac Accuracy: {acc:.4f}")
