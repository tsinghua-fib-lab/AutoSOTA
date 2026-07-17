import torch
import numpy as np
import time

from activation_gathering import activation_gathering_data_ordering
from data_processing import cached_function2, split_data_accoring_to_sentence_id2
from ml_model import SimpleLDA, SimpleMLP, SimpleTransformer


def detection_data_ordering(params):
    # Implement the data ordering function for detection
    return activation_gathering_data_ordering(params)


def detect_watermark(df, params):
    # Curate the usefull parameters for the caching system
    curated_params = {
        "parameters_type": "detection",
        "data_ordering_function": detection_data_ordering(params),
        "model_arguments": params["model_arguments"],
        "data_arguments": params["data_arguments"],
        "generation_arguments": params["generation_arguments"],
        "steering_arguments": params["steering_arguments"],
        "gathering_arguments": params["gathering_arguments"],
        "detection_arguments": params["detection_arguments"],
        "verbose": params.get("verbose", True),
        "hf_token": params["hf_token"],
        "split_labels": params["split_labels"] if "split_labels" in params else None,
    }
    # Optionally add paraphrasing parameters
    if params["robustness_arguments"]["paraphrasing"]["enabled"]:
        curated_params["robustness_arguments"] = {"paraphrasing": params["robustness_arguments"]["paraphrasing"]}
    return detect_watermark_cached(df, curated_params)


@cached_function2()
def detect_watermark_cached(data, params):
    verbose = params.get("verbose", True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(params["detection_arguments"].get("seed", 1))
    np.random.seed(params["detection_arguments"].get("seed", 1))

    # Truncation if prarameter is set
    if params["detection_arguments"].get("number_prompts_truncation", None) is not None:
        number_prompts_truncation = params["detection_arguments"]["number_prompts_truncation"]
        data = data[data["input_text_id"] < number_prompts_truncation]
        print("'#################################################################################")
        print("/!\ ATTENTION. Training data has be truncated down to ", number_prompts_truncation, "prompts for detection.")
        print("'#################################################################################")

    # print(">>>>>>>>> In Detection:")
    # for i in range(len(data)):
    #     print("Data index:", i)
    #     print("Params:", data.iloc[i]["output_text"][:200])

    print("Mesure time before splitting data...")
    time1 = time.time()

    # print("data columns:", data.columns)
    # print("activation data shape example:", data.iloc[0]["activations"][15].shape)
    # print("1 activation len :", data.iloc[0]["activations"][15][0])
    # print("Above should be an array of length 4096 or so")

    df_train, df_val, df_test, split_list = split_data_accoring_to_sentence_id2(
        data,
        val_size=0.1, 
        test_size=0.2, 
        seed=0, 
        token_aggregation=params["detection_arguments"]["token_aggregation"],
        sentence_array=params["detection_arguments"]["sentence_array"],
        max_token_seq=params["detection_arguments"].get("max_seq_length", None),
        split_labels=params["split_labels"] if "split_labels" in params else None,
    )

    time2 = time.time()
    print(f"Data splitting took {time2 - time1:.2f} seconds.")

    # Data
    X_train, Y_train = df_train["fwd_data"].values, df_train["classification_label"].values.astype(np.int64)
    X_val, Y_val = df_val["fwd_data"].values, df_val["classification_label"].values.astype(np.int64)
    X_test, Y_test = df_test["fwd_data"].values, df_test["classification_label"].values.astype(np.int64)

    # Robust data if available
    if params.get("robustness_arguments", {}).get("paraphrasing", {}).get("enabled", False):
        # X_test, Y_test = df_test["fwd_data_robust"].values, df_test["classification_label"].values.astype(np.int64)
        # TODO: Check if dropno dont create a problem by forgetting where the data is located - No token id etc..
        # Drop rows with NaN in fwd_data_robust
        df_val = df_val.dropna(subset=["fwd_data_robust"])
        df_test = df_test.dropna(subset=["fwd_data_robust"])

        X_val, Y_val = df_val["fwd_data_robust"].values, df_val["classification_label"].values.astype(np.int64)
        X_test, Y_test = df_test["fwd_data_robust"].values, df_test["classification_label"].values.astype(np.int64)



    print("Training label content shapes:")
    print("Label min and max", Y_train.min(), Y_train.max())
    print("Count each value of Y_train:", np.bincount(Y_train))
    print("Count each value of Y_val:", np.bincount(Y_val))
    print("Count each value of Y_test:", np.bincount(Y_test))


    # Padding of the inputs to have the same length, In case of sentence_array representation
    if params["detection_arguments"]["sentence_array"]:
        print("Train size:", len(X_train), "Val size:", len(X_val), "Test size:", len(X_test))
        input_dim = np.max([x.shape[0] for x in list(X_train) + list(X_val) + list(X_test)])
        if input_dim % params["detection_arguments"]["transformer_parameters"]["num_heads"] != 0:
            additional_padding = params["detection_arguments"]["transformer_parameters"]["num_heads"] - (input_dim % params["detection_arguments"]["transformer_parameters"]["num_heads"])
            input_dim += additional_padding
        #     print("Rounding up by ", additional_padding, "to be divisible by num_heads:", params["detection_arguments"]["transformer_parameters"]["num_heads"])
        # print("Input dim:", input_dim)
        print("Just a test print")
        for i in range(len(X_train)):
            if X_train[i].shape[0] < input_dim:
                padding = np.zeros((input_dim - X_train[i].shape[0], X_train[i].shape[1]), dtype=np.float16)
                X_train[i] = np.vstack((X_train[i], padding))
        for i in range(len(X_val)):
            if X_val[i].shape[0] < input_dim:
                padding = np.zeros((input_dim - X_val[i].shape[0], X_val[i].shape[1]), dtype=np.float16)
                X_val[i] = np.vstack((X_val[i], padding))
        for i in range(len(X_test)):
            if X_test[i].shape[0] < input_dim:
                padding = np.zeros((input_dim - X_test[i].shape[0], X_test[i].shape[1]), dtype=np.float16)
                X_test[i] = np.vstack((X_test[i], padding))
        print("The other test print")

        X_train = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(device) for x in X_train]).to(device)
        X_val = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(device) for x in X_val]).to(device)
        X_test = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(device) for x in X_test]).to(device)

        print("After stacking:")

    else:
        # set as array
        # X_train = np.array([np.array(x, dtype=np.float16) for x in X_train])
        # X_val = np.array([np.array(x, dtype=np.float16) for x in X_val])
        # X_test = np.array([np.array(x, dtype=np.float16) for x in X_test])
        X_train = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(device) for x in X_train]).to(device)
        X_val = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(device) for x in X_val]).to(device)
        X_test = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(device) for x in X_test]).to(device)


    # X_train = [np.asarray(x, dtype=np.float16) for x in X_train]  # Assuming raw list
    # X_val = [np.asarray(x, dtype=np.float16) for x in X_val]  # Assuming raw list
    # X_test = [np.asarray(x, dtype=np.float16) for x in X_test]  # Assuming raw list


    output_dim = len(np.unique(Y_train))

    print("Initializing model...")

    # Simple MLP model
    if params["detection_arguments"]["model_type"] == "mlp":
        print("Y_train", Y_train)
        print("X_train", X_train)
        print("len X_train", len(X_train))
        token_dim = X_train[0].shape[0]
        print("Token dim:", token_dim)

        model_parameters = params["detection_arguments"]["mlp_parameters"]
        model = SimpleMLP(
            input_dim=token_dim,
            hidden_dims=model_parameters["hidden_dims"],
            output_dim=output_dim,
            device=device,
        ).to(device)

    # Transformer model for sequence data
    elif params["detection_arguments"]["model_type"] == "transformer":
        model_parameters = params["detection_arguments"]["transformer_parameters"]
        if not params["detection_arguments"]["sentence_array"]:
            raise ValueError("Transformer model requires sentence_array to be True.")

        input_dim = X_train[0].shape[0]  # Sequence length
        token_dim = X_train[0].shape[1]  # Number of features in the input data

        model = SimpleTransformer(
            input_dim=input_dim,
            token_dim=token_dim,
            num_heads=model_parameters["num_heads"],
            num_layers=model_parameters["num_layers"],
            dim_feedforward=model_parameters["dim_feedforward"],
            dropout=model_parameters["dropout"],
            output_dim=output_dim,
            device=device,
        ).to(device)

    elif params["detection_arguments"]["model_type"] == "LDA":
        model_parameters = params["detection_arguments"]["lda_parameters"]
        model = SimpleLDA(model_parameters)


    print("Model fitting...")
    # Training
    loss_memorry, train_accuracy, validation_accuracy, validation_loss = model.fit(
        train_data=X_train, 
        train_labels=Y_train, 
        val_data=X_val, 
        val_labels=Y_val,
        epochs=model_parameters.get("num_epochs", None),
        batch_size=model_parameters.get("batch_size", None),
        learning_rate=model_parameters.get("learning_rate", None),
        verbose=True,
    )

    # Evaluation
    test_accuracy, test_predictions, test_probabilities = model.evaluate(X_test, Y_test, batch_size=model_parameters.get("batch_size", None))
    
    result_dict = {
        "train_loss": loss_memorry,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "validation_loss": validation_loss,
        "test_accuracy": test_accuracy,
        "test_ground_truth": Y_test,
        "test_sentence_ids": df_test["input_text_id"].values,
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
        "test_token_ids": df_test["token_id"].values,
        "split_list": split_list,
        "model": model,
    }

    return result_dict




