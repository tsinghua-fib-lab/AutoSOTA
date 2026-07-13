import torch
import numpy as np
from dataloader import DataLoader
from NAVAR import NAVAR, NAVARLSTM

def train_NAVAR(data, maxlags=5, hidden_nodes=256, dropout=0, epochs=200, learning_rate=1e-4,
                          batch_size=300, lambda1=0, val_proportion=0.0,  weight_decay=0,
                          check_every=1000, hidden_layers=1, normalize=True, split_timeseries=False, lstm=False):
    """
    Trains a Neural Additive Vector Autoregression (NAVAR) model on time series data and scores the
    potential causal links between variables.

    Args:
        data:  ndarray
            T (time points) x N (variables) input data
        maxlags: int
            Maximum number of time lags
        hidden_nodes: int
            Number of hidden nodes in each layers
        dropout: float
            Dropout probability in the hidden layers
        epochs: int
            Number of training epochs
        learning_rate: float
            Learning rate for Adam optimizer
        batch_size: int
            The size of the training batches
        lambda1: float
            Parameter for penalty to the contributions
        val_proportion: float
            Proportion of the dataset used for validation
        weight_decay: float
            Weight decay used in neural networks
        check_every: int
            Every 'check_every'th epoch we print training progress
        hidden_layers: int
            Number of hidden layers in the neural networks
        normalize: bool
            Indicates whether we should should normalize every variable
        split_timeseries: int
            If the original time series consists of multiple shorter time series, this argument should indicate the
            original time series length. Otherwise should be zero.
        lstm: bool
            Indicates whether we should use the LSTM model (instead of MLP).

    Returns:
        causal_matrix: ndarray
            N (variables) x N (variables) array containing the scores for every causal link.
            causal_matrix[i, j] indicates the score for potential link i -> j

        contributions: ndarray
            N^2 x training_examples array containing the contributions from and to every variable
            for every sample in the training_set

        loss_val: float
            Validation loss of the model after training
    """
    #Modification to add a device - was not there in the original file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # T is the number of time steps, N the number of variables
    T, N = data.shape

    # initialize the NAVAR model
    if lstm:
        model = NAVARLSTM(N, hidden_nodes, maxlags, dropout=dropout, hidden_layers=hidden_layers)
    else:
        model = NAVAR(N, hidden_nodes, maxlags, dropout=dropout, hidden_layers=hidden_layers)

    # use Mean Squared Error and the Adam optimzer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # obtain the training and validation data
    dataset = DataLoader(data, maxlags, normalize=normalize, val_proportion=val_proportion, split_timeseries=split_timeseries, lstm=lstm)
    X_train, Y_train = dataset.train_Xs, dataset.train_Ys
    X_val, Y_val = dataset.val_Xs, dataset.val_Ys
    # push model and data to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        if X_val is not None:
            X_val = X_val.cuda()
            Y_val = Y_val.cuda()

    num_training_samples = X_train.shape[0]
    total_loss = 0
    loss_val = 0

    # start of training loop
    batch_counter = 0
    for t in range(1, epochs +1):
        #obtain batches
        batch_indeces_list = []
        if batch_size < num_training_samples:
            batch_perm = np.random.choice(num_training_samples, size=num_training_samples, replace=False)
            for i in range(int(num_training_samples/batch_size) + 1):
                start = i*batch_size
                batch_i = batch_perm[start:start+batch_size]
                if len(batch_i) > 0:
                    batch_indeces_list.append(batch_perm[start:start+batch_size])
        else:
            batch_indeces_list = [np.arange(num_training_samples)]

        for batch_indeces in batch_indeces_list:
            batch_counter += 1
            X_batch = X_train[batch_indeces]
            Y_batch = Y_train[batch_indeces]
            
            # forward pass to calculate predictions and contributions
            predictions, contributions = model(X_batch)

            # calculate the loss
            if not lstm and not split_timeseries:
                loss_pred = criterion(predictions, Y_batch)
            else:
              #Replaced the line below with different code (below)
                #loss_pred = criterion(predictions[:,:,-1], Y_batch[:,:,-1])
                # If predictions are (B, N, T) use last time step; if (B, N) use as-is.
                if predictions.dim() == 3 and Y_batch.dim() == 3:
                  loss_pred = criterion(predictions[:, :, -1], Y_batch[:, :, -1])
                else:
                  loss_pred = criterion(predictions, Y_batch)
            loss_l1 = (lambda1/N) * torch.mean(torch.sum(torch.abs(contributions), dim=1))
            loss = loss_pred + loss_l1
            total_loss += loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # every 'check_every' epochs we calculate and print the validation loss
        if t % check_every == 0:
            model.eval()
            if val_proportion > 0.0:
                val_pred, val_contributions = model(X_val)
                loss_val = criterion(val_pred, Y_val)
            model.train()

            print(f'iteration {t}. Loss: {total_loss/batch_counter}  Val loss: {loss_val}')
            total_loss = 0
            batch_counter = 0

    # use the trained model to calculate the causal scores
    model.eval()
    y_pred, contributions = model(X_train)
    causal_matrix = torch.std(contributions, dim=0).view(N, N).detach().cpu().numpy()

    # This line of code is original, commented out here for use with SHAP
    #return causal_matrix, contributions, loss_val

    #For a modified version compute val_loss:
    if val_proportion and val_proportion > 0:
      model.eval()
      with torch.no_grad():
        Xv = X_val.detach().clone().to(device)
        Yv = Y_val.detach().clone().to(device)
        preds_v, _ = model(Xv)
        val_loss = criterion(preds_v, Yv).item()
    else:
      val_loss = None

    #This is a modified version
    return causal_matrix, contributions, val_loss, model
