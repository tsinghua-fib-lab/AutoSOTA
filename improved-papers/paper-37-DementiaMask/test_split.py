from src.preprocessing.data_generator import cf_train_test_split

train_df, eval_df, test_df = cf_train_test_split(
    data_name="pitts",
    train_pos_z0=0.2,
    train_pos_z1=0.8,
    alpha_test=0.25,
    test_size=150,
    pz=0.5,
    random_state=2025,
    validation_size=100,
    min_samples=10,
    verbose=True,
)

print(eval_df.head())
