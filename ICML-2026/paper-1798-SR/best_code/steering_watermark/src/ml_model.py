# Define simple MLP for classification with pytorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class SimpleLDA(object):
    def __init__(self, model_parameters):
        self.model = LinearDiscriminantAnalysis(
            shrinkage=model_parameters.get("shrinkage", None),
            solver=model_parameters.get("solver", "lsqr"),
        )

    def fit(self, train_data, train_labels, val_data, val_labels, epochs=None, batch_size=None, learning_rate=None, verbose=True):

        self.model.fit(train_data.cpu().numpy(), train_labels)

        # Get the training accuracy
        train_predictions = self.model.predict(train_data.cpu().numpy())
        train_accuracy = accuracy_score(train_labels, train_predictions)

        # Get the validation accuracy
        val_predictions = self.model.predict(val_data.cpu().numpy())
        val_accuracy = accuracy_score(val_labels, val_predictions)

        if verbose:
            print("-> LDA Training accuracy:", train_accuracy)
            print("-> LDA Validation accuracy:", val_accuracy)

        fake_loss = [0.0]

        return fake_loss, [train_accuracy], [val_accuracy], fake_loss


    def evaluate(self, test_data, test_labels, batch_size=128):
        test_predictions = self.model.predict(test_data.cpu().numpy())
        test_accuracy = accuracy_score(test_labels, test_predictions)

        # Get probabilities
        test_probabilities = self.model.predict_proba(test_data.cpu().numpy())
        print("-> LDA Test accuracy:", test_accuracy)
        print("-> LDA Test Classification Report:\n", classification_report(test_labels, test_predictions))

        return test_accuracy, test_predictions, test_probabilities        



class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, device="CUDA"):
        super(SimpleMLP, self).__init__()
        layer_size_list = [input_dim] + hidden_dims + [output_dim]
        self.device = device

        self.layers = []
        for i in range(len(layer_size_list) - 1):
            self.layers.append(nn.Linear(layer_size_list[i], layer_size_list[i + 1]))
            if i < len(layer_size_list) - 2:
                self.layers.append(nn.ReLU())
        self.layers = nn.ModuleList(self.layers)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.softmax(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(self.layers[0].weight.device)
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def fit(self, train_data, train_labels, val_data, val_labels, epochs=10, batch_size=128, learning_rate=0.001, verbose=True):
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # dataloaders
        # x_tensor = torch.from_numpy(train_data)
        y_tensor = torch.from_numpy(train_labels)
        train_dataset = TensorDataset(train_data, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # x_val_tensor = torch.from_numpy(val_data)
        y_val_tensor = torch.from_numpy(val_labels)
        val_dataset = TensorDataset(val_data, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


        loss_memorry = []
        train_accuracy = []
        validation_accuracy = []
        validation_loss = []

        # Train the model
        for epoch in range(epochs):
            loss_accumulator = 0.0
            accuracy_accumulator = 0.0
            counter = 0
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.device).float()
                Y_batch = Y_batch.to(self.device)

                self.train()
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()

                loss_accumulator += loss.item()
                accuracy_accumulator += (outputs.argmax(dim=1) == Y_batch).sum().item()
                counter += Y_batch.size(0)

                if verbose :
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

                if (i+1) % 1 == 0:
                    loss_memorry.append(loss_accumulator / counter)
                    loss_accumulator = 0.0
                    train_accuracy.append(accuracy_accumulator / counter)
                    accuracy_accumulator = 0.0
                    counter = 0

            loss_accumulator = 0.0
            accuracy_accumulator = 0.0
            counter = 0

            # Calculating Validation accuracy
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(self.device).float()
                Y_batch = Y_batch.to(self.device)

                self.eval()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, Y_batch)

                loss_accumulator += loss.item()
                accuracy_accumulator += (outputs.argmax(dim=1) == Y_batch).sum().item()
                counter += Y_batch.size(0)

                validation_loss.append(loss_accumulator / counter)
                validation_accuracy.append(accuracy_accumulator / counter)


        return loss_memorry, train_accuracy, validation_accuracy, validation_loss
    

    def evaluate(self, test_data, test_labels, batch_size=128):
        # Evaluate the model
        self.eval()
        with torch.no_grad():
            # batch wise test
            # X_test_tensor = torch.tensor(test_data, dtype=torch.float32)
            Y_test_tensor = torch.tensor(test_labels, dtype=torch.long)
            test_dataset = TensorDataset(test_data, Y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            accuracy = 0.0
            counter = 0
            total_predictions = []
            total_probabilities = []

            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(self.device).float()
                Y_batch = Y_batch.to(self.device)

                self.eval()
                outputs = self.forward(X_batch)
                _, predicted = torch.max(outputs, 1)
                accuracy += (predicted == Y_batch).sum().item()
                counter += Y_batch.size(0)
                total_predictions.extend(predicted.cpu().numpy())
                total_probabilities.extend(outputs.cpu().numpy())

            accuracy = accuracy / counter
            print("PyTorch ST test Accuracy:", accuracy)
            print("PyTorch ST test Classification Report:\n", classification_report(test_labels, total_predictions))
        return accuracy, total_predictions, total_probabilities




##########################
#   Transformer model    #
##########################

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, token_dim, num_heads, num_layers, dim_feedforward, dropout, output_dim, device="CUDA"):
        super(SimpleTransformer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.pos_embedding = nn.Parameter(torch.randn(1, self.input_dim, self.token_dim)).to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim, 
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='relu',
            batch_first=True,
        ).to(self.device)

        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
        ).to(self.device)

        self.fc = nn.Linear(self.token_dim, output_dim).to(self.device)
        # self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1).to(self.device)

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)   # Average pooling over the sequence length
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(self.fc.weight.device)
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()


    def fit(self, train_data, train_labels, val_data, val_labels, epochs=10, batch_size=128, learning_rate=0.001, verbose=True):
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # dataloaders
        # x_tensor = torch.from_numpy(train_data)
        y_tensor = torch.from_numpy(train_labels)
        # train_dataset = TensorDataset(x_tensor, y_tensor)
        train_dataset = TensorDataset(train_data, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # x_val_tensor = torch.from_numpy(val_data)
        y_val_tensor = torch.from_numpy(val_labels)
        # val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_dataset = TensorDataset(val_data, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


        loss_memorry = []
        train_accuracy = []
        validation_accuracy = []
        validation_loss = []
        for epoch in range(epochs):
            loss_accumulator, accuracy_accumulator, counter = 0.0, 0.0, 0
            self.train()
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()

                loss_accumulator += loss.item()
                accuracy_accumulator += (outputs.argmax(dim=1) == Y_batch).sum().item()
                counter += Y_batch.size(0)

                if (i+1) % 1 == 0:
                    loss_memorry.append(loss_accumulator)
                    loss_accumulator = 0.0
                    train_accuracy.append(accuracy_accumulator / counter)
                    accuracy_accumulator = 0.0
                    counter = 0

                if verbose:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f} Train accuracy: {train_accuracy[-1]:.4f}")

            loss_accumulator = 0.0
            accuracy_accumulator = 0.0
            counter = 0

            self.eval()
            # Calculating Validation accuracy
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                outputs = self.forward(X_batch)
                loss = criterion(outputs, Y_batch)

                loss_accumulator += loss.item()
                accuracy_accumulator += (outputs.argmax(dim=1) == Y_batch).sum().item()
                counter += Y_batch.size(0)

                validation_loss.append(loss_accumulator / counter)
                validation_accuracy.append(accuracy_accumulator / counter)


        return loss_memorry, train_accuracy, validation_accuracy, validation_loss
    
    def evaluate(self, test_data, test_labels, batch_size=128):
        # Evaluate the model
        self.eval()
        with torch.no_grad():
            # batch wise test
            # X_test_tensor = torch.tensor(test_data, dtype=torch.float32)
            Y_test_tensor = torch.tensor(test_labels, dtype=torch.long)
            # test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
            test_dataset = TensorDataset(test_data, Y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            accuracy = 0.0
            counter = 0
            total_predictions = []
            total_probabilities = []

            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                outputs = self.forward(X_batch)
                _, predicted = torch.max(outputs, 1)
                accuracy += (predicted == Y_batch).sum().item()
                counter += Y_batch.size(0)
                total_predictions.extend(predicted.cpu().numpy())
                total_probabilities.extend(outputs.cpu().numpy())


            accuracy = accuracy / counter
            print("PyTorch ST test Accuracy:", accuracy)
            print("PyTorch ST test Classification Report:\n", classification_report(test_labels, total_predictions))
        return accuracy, total_predictions, total_probabilities




if __name__ == "__main__":
    import pickle
    from data_processing import split_data_accoring_to_sentence_id
    import numpy as np
    import random
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import matplotlib.pyplot as plt

    global_seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlp_parameters = {
        # Transformer parameters
        "num_heads": 2,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0,
        "output_dim": 2,  # Binary classification

        # MLP parameters
        "hidden_dims": [1024, 64],

        # Training parameters
        "batch_size": 512,           #16
        "num_epochs": 1,            #6
        "learning_rate": 0.0001    #0.00001
    }

    # Load some data:
    with open("data/home_made_v1.210_Meta-Llama-3-8B_21a608ad/gathering_detection_steered_677d4d77d359413736ad692409e9fce4.pkl", "rb") as f:
        detection_data = pickle.load(f)

    with open("data/home_made_v1.210_Meta-Llama-3-8B_21a608ad/gathering_detection_vanilla_ca5d7c6f7ce21dd373ec2154cc1a943d.pkl", "rb") as f:
        vanilla_data = pickle.load(f)

    print("Parameter of the data inputs:")
    print("Detection data:", detection_data[0]["input"]["steering"])
    print("Detection data:", detection_data[0].keys())
    for key in detection_data[0]["input"].keys():
        print(f" - {key}: {detection_data[0]['input'][key]}")
    print("Vanilla data:", vanilla_data[0]["input"]["steering"])

    all_detection_data = detection_data + vanilla_data

    # Transformer model test
    if False:

        df_train, df_val, df_test = split_data_accoring_to_sentence_id(all_detection_data, val_size=0.1, test_size=0.2, seed=global_seed, token_aggregation=False, sentence_array=True)

        # Data
        X_train, Y_train = df_train["fwd_data"].values, df_train["label"].values.astype(np.int64)
        X_val, Y_val = df_val["fwd_data"].values, df_val["label"].values.astype(np.int64)
        X_test, Y_test = df_test["fwd_data"].values, df_test["label"].values.astype(np.int64)

        print("Train size:", len(X_train), "Val size:", len(X_val), "Test size:", len(X_test))
        # Padding of the inputs to have the same length
        input_dim = np.max([x.shape[0] for x in list(X_train) + list(X_val) + list(X_test)])
        if input_dim % mlp_parameters["num_heads"] != 0:
            additional_padding = mlp_parameters["num_heads"] - (input_dim % mlp_parameters["num_heads"])
            input_dim += additional_padding
            print("Rounding up by ", additional_padding, "to be divisible by num_heads:", mlp_parameters["num_heads"])
        print("Input dim:", input_dim)
        for i in range(len(X_train)):
            if X_train[i].shape[0] < input_dim:
                padding = np.zeros((input_dim - X_train[i].shape[0], X_train[i].shape[1]), dtype=np.float32)
                X_train[i] = np.vstack((X_train[i], padding))
        for i in range(len(X_val)):
            if X_val[i].shape[0] < input_dim:
                padding = np.zeros((input_dim - X_val[i].shape[0], X_val[i].shape[1]), dtype=np.float32)
                X_val[i] = np.vstack((X_val[i], padding))
        for i in range(len(X_test)):
            if X_test[i].shape[0] < input_dim:
                padding = np.zeros((input_dim - X_test[i].shape[0], X_test[i].shape[1]), dtype=np.float32)
                X_test[i] = np.vstack((X_test[i], padding))

        X_train = np.array([np.array(x, dtype=np.float32) for x in X_train])
        X_val = np.array([np.array(x, dtype=np.float32) for x in X_val])
        X_test = np.array([np.array(x, dtype=np.float32) for x in X_test])

        print("After padding:")
        print("Train shape:", X_train.shape)
        print("Val shape:", X_val.shape)
        print("Test shape:", X_test.shape)

        token_dim = X_train[0].shape[1]  # Number of features in the input data

        model = SimpleTransformer(
            input_dim, 
            token_dim,
            mlp_parameters["num_heads"], 
            mlp_parameters["num_layers"], 
            mlp_parameters["dim_feedforward"],
            mlp_parameters["dropout"],
            mlp_parameters["output_dim"], 
            device=device,
        ).to(device)


    # MLP model test
    else:

        df_train, df_val, df_test = split_data_accoring_to_sentence_id(all_detection_data, val_size=0.1, test_size=0.2, seed=global_seed, token_aggregation=False, sentence_array=False)

        X_train, Y_train = df_train["fwd_data"].values, df_train["label"].values.astype(np.int64)
        X_val, Y_val = df_val["fwd_data"].values, df_val["label"].values.astype(np.int64)
        X_test, Y_test = df_test["fwd_data"].values, df_test["label"].values.astype(np.int64)

        X_train = np.array([np.array(x, dtype=np.float32) for x in X_train])
        X_val = np.array([np.array(x, dtype=np.float32) for x in X_val])
        X_test = np.array([np.array(x, dtype=np.float32) for x in X_test])

        token_dim = X_train[0].shape[0]
        print("Train size:", len(X_train), "Val size:", len(X_val), "Test size:", len(X_test))
        print("Token dimention (Input):", token_dim)

        model = SimpleMLP(
            input_dim=token_dim,
            hidden_dims=mlp_parameters["hidden_dims"],
            output_dim=mlp_parameters["output_dim"],
            device=device,
        ).to(device)

    # Train the model
    loss_memorry, train_accuracy, validation_accuracy, validation_loss = model.fit(
        train_data=X_train, 
        train_labels=Y_train, 
        val_data=X_val, 
        val_labels=Y_val,
        epochs=mlp_parameters["num_epochs"],
        batch_size=mlp_parameters["batch_size"],
        learning_rate=mlp_parameters["learning_rate"],
        verbose=True,
    )

    # plot the loss and train accuracy for each key
    color_palette = px.colors.qualitative.Plotly
    curve_counter = 0
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Loss Memory", "Train Accuracy", "Validation Loss", "Validation Accuracy"))
    
    legend_name = f"This test training"

    fig.add_trace(
        go.Scatter(y=loss_memorry, name=legend_name, 
                    line=dict(color=color_palette[curve_counter % len(color_palette)]),
                    showlegend=True,
                    legendgroup=f"{legend_name}",
                    text=f"",
                    hoverinfo="text"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=train_accuracy, name=legend_name,
                    line=dict(color=color_palette[curve_counter % len(color_palette)]),
                    showlegend=False,
                    legendgroup=f"{legend_name}",
                    text=f"",
                    hoverinfo="y"),
        row=1, col=2
    ) 
    fig.add_trace(
        go.Scatter(y=validation_loss, name=legend_name,
                    line=dict(color=color_palette[curve_counter % len(color_palette)]),
                    showlegend=False,
                    legendgroup=f"{legend_name}",
                    text=f"",
                    hoverinfo="text"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=validation_accuracy, name=legend_name,
                    line=dict(color=color_palette[curve_counter % len(color_palette)]),
                    showlegend=False,
                    legendgroup=f"{legend_name}",
                    text=f"",
                    hoverinfo="y"),
        row=2, col=2
    )

    curve_counter += 1
    fig.update_layout(title_text="Loss and Train Accuracy for each Key")
    fig.write_html("Trainain_test_transformer.html")
    fig.show()


    # Evaluate the model
    model.evaluate(X_test, Y_test, batch_size=mlp_parameters["batch_size"])