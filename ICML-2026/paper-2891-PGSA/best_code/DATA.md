# Dataset setup for PSAHS

The code expects all data under `dataset/` at the **repository root** (`PSAHS-release/dataset/`).

Scripts and training entry points use absolute paths via `psahs/paths.py`, so you can run commands from the repo root.

## Quick setup

From the **repository root** (`PSAHS-release/`):

```bash
# Twitch (SNAP download)
python scripts/download_twitch_musae.py

# Synthetic Noncircle graphs
python scripts/generate_synthetic_graphs.py --num_nodes 4000
```

Place Blog, DBLP–ACM, and Airport files manually (see sections below).

Train from the repo root:

```bash
python main/train_mlp.py -d Twitch --src_name EN --tgt_name DE
python main/train_mlp.py -d Blog --src_name Blog1 --tgt_name Blog2
python main/main.py -d Twitch --src_name EN --tgt_name DE
```

Below are the **exact layouts and file formats** for each dataset and where to get or how to build them.

---

## 1. Blog (source/target: Blog1, Blog2)

**Used by:** `load_data_from_mat("../dataset/Blog", name)` with `name in ("Blog1", "Blog2")`.

### Expected layout

```
dataset/
  Blog/
    Blog1.mat
    Blog2.mat
```

### .mat file format

Each `.mat` file must contain (MATLAB/NumPy `scipy.io.loadmat`):

| Variable | Type | Description |
|----------|------|-------------|
| `network` | sparse matrix (N×N) | Adjacency matrix (binary or weighted). Symmetrized in code. |
| `attrb`   | array (N×D)         | Node features (float). |
| `group`   | array (N×C)         | One-hot or multi-label; code uses `labels = argmax(group, 1)` for class indices. |

### Where to get it

- This format (Blog1/Blog2 as two domains) is common in **graph domain adaptation** papers. If you have a copy from a paper (e.g. StruRW, DANN on graphs), place `Blog1.mat` and `Blog2.mat` in `dataset/Blog/`.
- If you only have a single Blog graph (e.g. BlogCatalog) with adjacency, features, and labels, you can split it into two .mat files (e.g. by time, subgraph, or random split) and save each with `network`, `attrb`, `group` as above.

### Example (creating from NumPy/Scipy)

```python
import scipy.io as sio
import numpy as np
from scipy import sparse

# adj: scipy sparse NxN, feats: (N,D), labels: (N,) class indices 0..C-1
N, C = feats.shape[0], int(labels.max()) + 1
group = np.zeros((N, C))
group[np.arange(N), labels] = 1
sio.savemat('dataset/Blog/Blog1.mat', {'network': adj, 'attrb': feats, 'group': group})
```

---

## 2. Twitch (source/target: DE, EN, FR, etc.)

**Used by:** `prepare_Twitch("../dataset/Twitch/", lang)` with `lang` e.g. `"DE"`, `"EN"`, `"FR"`.

### Expected layout

```
dataset/
  Twitch/
    DE/
      raw/
        musae_DE_target.csv
        musae_DE_features.json
        musae_DE_edges.csv
    EN/
      raw/
        musae_EN_target.csv
        musae_EN_features.json
        musae_EN_edges.csv
    FR/
      raw/
        musae_FR_target.csv
        musae_FR_features.json
        musae_FR_edges.csv
```

### File formats

- **musae_{lang}_target.csv**  
  CSV with header. Code uses: column index **5** as node id, column index **2** as label (`"True"` / `"False"` → 1/0). Duplicate node ids are skipped (first occurrence kept).

- **musae_{lang}_features.json**  
  JSON object: keys = node id (string), values = list of integer feature indices (one-hot style). Code builds an (N×3170) matrix and sets `features[node_id, feat_idx] = 1`.

- **musae_{lang}_edges.csv**  
  CSV with header. First two columns are source and target node ids (integers).

### Where to get it

- **SNAP Twitch (language subgraphs):**  
  https://snap.stanford.edu/data/twitch-social-networks.html  
  Provides edges and node features/labels per language. You may need to convert to the MUSAE filenames and CSV/JSON format above (same column semantics as in `Utils/pre_data/datasets.py`).

- **PyTorch Geometric:**  
  `torch_geometric.datasets.Twitch` downloads a different layout (e.g. `.npz`). To use it here, write a small script that loads each language’s data and saves the three files (`*_target.csv`, `*_features.json`, `*_edges.csv`) under `dataset/Twitch/{DE,EN,FR}/raw/` with the formats above.

- **MUSAE-style repos:**  
  Some MUSAE-related repos (e.g. benedekrozemberczki/MUSAE, or datasets derived from it) provide Twitch in this CSV/JSON form; check their `raw` or `input` folders and copy into `dataset/Twitch/<lang>/raw/` with the `musae_*` names.

### Run example

```bash
python main/train_mlp.py -d Twitch --src_name DE --tgt_name EN
```

---

## 3. DBLP–ACM (source/target: e.g. ACMv9, DBLPv7)

**Used by:** `prepare_dblp_acm("../dataset", name)` with `name` e.g. `"ACMv9"`, `"DBLPv7"`.

### Expected layout

```
dataset/
  ACMv9/
    raw/
      ACMv9_docs.txt
      ACMv9_edgelist.txt
      ACMv9_labels.txt
  DBLPv7/
    raw/
      DBLPv7_docs.txt
      DBLPv7_edgelist.txt
      DBLPv7_labels.txt
```

### File formats

- **{name}_docs.txt**  
  One line per node. Each line is a comma-separated list of floats (node feature vector). Number of lines = number of nodes; line index = node index (0-based).

- **{name}_edgelist.txt**  
  One edge per line: `src,tgt` (comma-separated, 0-based node indices). Symmetrized in code.

- **{name}_labels.txt**  
  One line per node (same order as docs). Each line is a single integer class label (e.g. 0, 1, 2, …). No header.

### Where to get it

- **DBLP–ACM benchmark:**  
  - Open ICPSR: https://www.openicpsr.org/openicpsr/project/100843/version/V2/view  
  - Texas Data Repository (DLREP): doi:10.18738/T8/G5GZ51  

  These typically provide ACM/DBLP CSV (e.g. title, authors, venue, year) and possibly entity resolution mappings, not the preprocessed `_docs.txt` / `_edgelist.txt` / `_labels.txt`. You will need to:

  1. Build a graph (e.g. papers as nodes, edges from citations or similarity).
  2. Extract or learn node features (e.g. bag-of-words, embeddings) and write one line per node to `{name}_docs.txt`.
  3. Write edges to `{name}_edgelist.txt` and class labels (e.g. venue or topic) to `{name}_labels.txt`.

- **Preprocessed DBLP/ACM from GDA papers:**  
  Some graph domain adaptation or transfer learning papers release preprocessed DBLP/ACM with exactly this format (docs, edgelist, labels). If you obtain such a release, place the `ACMv9` and `DBLPv7` folders (or your variant names) under `dataset/` and pass the same names to `--src_name` and `--tgt_name`.

### Run example

```bash
python main/train_mlp.py -d dblp_acm --src_name ACMv9 --tgt_name DBLPv7
```

---

## 4. Airport (source/target: region names, e.g. USA, Europe, Brazil)

**Used by:** `prepare_airport("../dataset/Airport", name)` with `name` e.g. `"USA"`, `"Europe"`, `"Brazil"`.

**Important (label alignment):** The provided `scripts/prepare_airport.py` assigns labels per region as `id % n_classes` (USA: 5 classes, Europe: 10, Brazil: 3). Class index 0 in one region does **not** correspond to the same concept in another, so unsupervised domain adaptation will show poor target accuracy unless you use a **shared label scheme** (e.g. same taxonomy or same number of classes with a consistent rule across regions). For best results with Airport, either use labels that are semantically aligned across regions or treat results as structure-only transfer.

### Expected layout

```
dataset/
  Airport/
    USA/
      raw/
        USA_labels.txt
        USA_edgelist.txt
    Europe/
      raw/
        Europe_labels.txt
        Europe_edgelist.txt
    Brazil/
      raw/
        Brazil_labels.txt
        Brazil_edgelist.txt
```

### File formats

- **{name}_labels.txt**  
  One line per node. Each line is a single integer class label (e.g. 0..K−1). Number of lines = number of nodes; line index = node index (0-based). No header.

- **{name}_edgelist.txt**  
  One edge per line: `src,tgt` (comma-separated, 0-based node indices). Symmetrized in code.

Node features are not read from disk: the code uses degree-only and applies `OneHotDegree(241)` to get a one-hot degree feature vector (max degree 241).

### Where to get it

- **OpenFlights / airport networks:**  
  OpenFlights (https://openflights.org/data.html) or network repositories (e.g. “openflights” or “airport” on Network Repository) provide nodes (airports) and edges (routes). You need to:

  1. Restrict to a region (e.g. USA, Europe, Brazil) and build one graph per region.
  2. Assign a class label to each airport (e.g. by country, hub size, or region id) and write one label per line to `{name}_labels.txt`.
  3. Write edges as 0-based `src,tgt` to `{name}_edgelist.txt`. Node index must match the line index in `_labels.txt`.

- **Preprocessed Airport (USA/Europe/Brazil):**  
  Cross-region airport node classification (USA↔Europe, USA↔Brazil, etc.) is used in transfer/cross-network learning papers. If a paper or benchmark provides preprocessed graphs with the above layout, place them under `dataset/Airport/{USA,Europe,Brazil}/raw/` and use those names as `--src_name` / `--tgt_name`.

### Run example

```bash
python main/train_mlp.py -d Airport --src_name USA --tgt_name Europe
```

---

## Summary table

| Dataset   | Path passed to code           | Key files / format |
|-----------|-------------------------------|---------------------|
| Blog      | `../dataset/Blog`, `Blog1`/`Blog2` | `Blog1.mat`, `Blog2.mat`: `network`, `attrb`, `group` |
| Twitch    | `../dataset/Twitch/`, `DE`/`EN`/`FR` | `raw/musae_{lang}_target.csv`, `_features.json`, `_edges.csv` |
| DBLP–ACM  | `../dataset`, `ACMv9`/`DBLPv7` | `{name}/raw/{name}_docs.txt`, `_edgelist.txt`, `_labels.txt` |
| Airport   | `../dataset/Airport`, `USA`/`Europe`/`Brazil` | `{name}/raw/{name}_labels.txt`, `_edgelist.txt` |

Ensure the `dataset` directory (or your chosen root) exists and matches these layouts so that `main.py`, `train_mlp.py`, and `MLP_GNN_one_layer.py` can load the data without changing the loader logic.
