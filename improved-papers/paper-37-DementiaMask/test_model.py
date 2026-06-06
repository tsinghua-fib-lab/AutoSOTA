
from src.preprocessing.data_generator import cf_train_test_split
from src.model.weights_filter import DualFilter, ECFilter
import argparse

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default='ecf')

    args = argparser.parse_args()

    if args.model == "ecf":
    # model = DualFilter("bert-base-uncased")
        model = ECFilter("bert-base-uncased")
        train_df, eval_df, test_df = cf_train_test_split(
            data_name="pitts",
            train_pos_z0=0.8333,
            train_pos_z1=0.1667,
            alpha_test=5.0,
            test_size=150,
            validation_size=120,
            random_state=24,
            verbose=True
        )
        model.train(train_df, eval_df, test_df, num_epochs=1,
                    mask_ratio=0.15, n_layers=5, emb=False)
        model.predict(test_df)
        print(model.metrics)
    
    elif args.model == "df":
        model = DualFilter("bert-base-uncased")
        train_df, eval_df, test_df = cf_train_test_split(
            data_name="pitts",
            train_pos_z0=0.8333,
            train_pos_z1=0.1667,
            alpha_test=5.0,
            test_size=150,
            validation_size=120,
            random_state=24,
            verbose=True
        )
        model.train(train_df, eval_df, test_df, num_epochs=1,
                    mask_ratio=0.15, mask_type = 'A')
        model.predict(test_df)
        print(model.metrics)