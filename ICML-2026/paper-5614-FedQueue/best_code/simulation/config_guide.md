# Simulation Config Guide

```yaml
data_configs:
  dataset_name: str
  dataset_backend: str  # custom, hf, leaf, torchaudio, torchvision
  data_dir: str
  download: bool
  partition: str  # iid, dirichlet, pathological, unbalanced, pre
  partition_kwargs:
    data_seed: int = 42
    test_size: float = 0.2
    dirichlet_alpha: float = 0.3
    dirichlet_min_size: int = 2
    pathological_min_classes: int = 2
    unbalanced_keep_min: float = 0.5
    pre_infer_num_clients: bool = false
    pre_min_samples_per_client: int = 0
    pre_raw_data_fraction: float = 1.0
    pre_source: str = ""
    pre_index: int = -1
    ext_train_split: str = "train"
    ext_test_split: str = "test"
    ext_feature_key: str = ""
    ext_label_key: str = ""
    ext_config_name: str = ""
    seq_len: int = dataset default
    num_embeddings: int = dataset default
```

Notes:
- LEAF datasets require `partition: "pre"`.
- (TODO) `custom` dataset loading would be touched later
