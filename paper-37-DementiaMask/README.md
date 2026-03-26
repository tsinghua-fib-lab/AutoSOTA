## DualFilter
This is the code repo for paper _Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking_ accepted to ACL 2025 Main.
### Data Preprocessing
```python
from src.preprocessing.data_generator import cf_train_test_split

train_df, test_df = cf_train_test_split(
    data_name="pitts",
    train_pos_z0=0.2,
    train_pos_z1=0.8,
    alpha_test=0.25,
    test_size=150,
    pz=0.5,
    random_state=2025,
    min_samples=10,
    verbose=True
)

print(train_df.head())

# INFO:src.preprocessing.utils:Generated 1 valid configurations 

# Empirical Probabilities of Y=1 given Z
# -------------------------
# | P(Y=1|Z) | Z=0 |  Z=1  |
# -------------------------
# | Train    | 0.20 | 0.80 |
# | Test     | 0.80 | 0.20 |
# -------------------------
```

| id  | file  | source  | gender | text                                                 | label |
|-----|-------|--------|--------|------------------------------------------------------|-------|
| 289 | 289-2 | address | 1      | Wahoo ho ho. Well it's kind of a calamity isn't... | 1     |
| 178 | 178-1 | address | 1      | taking some cookies. And falling over. And mot... | 1     |
| 465 | 465-0 | address | 1      | she's doing the dishes. He's on the cookie try... | 1     |
| 094 | 094-3 | address | 1      | mother son and daughter. The water's spilling ... | 1     |
| 612 | 612-0 | address | 0      | Well for one thing the boy is stealing cookies... | 0     |

### Running Bias Mitigation

```python
model = DualFilter("bert-base-uncased")
train_df, eval_df, test_df = cf_train_test_split(
    data_name="pitts",
    train_pos_z0=0.2,
    train_pos_z1=0.8,
    alpha_test=0.25,
    test_size=150,
    validation_size=120,
    random_state=42,
    verbose=True
)
model.train_and_debias(train_df, eval_df, test_df, num_epochs=1,
                mask_ratio=0.2, mask_type = 'I')
results = model.predict(test_df)
print(model.metrics)

# {'base': {'full': {'accuracy': 0.7933333333333333, 'aps': 0.8842163554888419, 'roc': 0.8686222222222223, 'f1': 0.7973856209150326}, 'confounder_0': {'accuracy': 0.7733333333333333, 'aps': 0.7076584509401846, 'roc': 0.9155555555555556, 'f1': 0.6382978723404256, 'precision_pos': 0.46875, 'precision_neg': 1.0, 'recall_pos': 1.0, 'recall_neg': 0.7166666666666667, 'pos_rate': 0.4721135}, 'confounder_1': {'accuracy': 0.8133333333333334, 'aps': 0.9545751138330678, 'roc': 0.7977777777777778, 'f1': 0.8679245283018869, 'precision_pos': 1.0, 'precision_neg': 0.5172413793103449, 'recall_pos': 0.7666666666666667, 'recall_neg': 1.0, 'pos_rate': 0.57852304}}, 'I_0.2_0.0': {'full': {'accuracy': 0.7933333333333333, 'aps': 0.8690633406598616, 'roc': 0.8609777777777777, 'f1': 0.8098159509202455}, 'confounder_0': {'accuracy': 0.8266666666666667, 'aps': 0.7242109742109741, 'roc': 0.9222222222222222, 'f1': 0.6976744186046512, 'precision_pos': 0.5357142857142857, 'precision_neg': 1.0, 'recall_pos': 1.0, 'recall_neg': 0.7833333333333333, 'pos_rate': 0.500205}, 'confounder_1': {'accuracy': 0.76, 'aps': 0.961406924605666, 'roc': 0.8366666666666667, 'f1': 0.85, 'precision_pos': 0.85, 'precision_neg': 0.4, 'recall_pos': 0.85, 'recall_neg': 0.4, 'pos_rate': 0.51275617}}}
```
