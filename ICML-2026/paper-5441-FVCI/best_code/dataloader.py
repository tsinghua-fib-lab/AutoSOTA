import torch
import numpy as np

class DataLoader(object):

    def __init__(self, data, maxlags, normalize=True,  val_proportion=0.1, split_timeseries=False, lstm=False):
        self.all_Xs, self.all_Ys = self.prepare_data(data, maxlags, normalize, split_timeseries, lstm=lstm)
        self.train_Xs, self.train_Ys, self.val_Xs, self.val_Ys = self.split_train_val(val_proportion)

    def prepare_data(self, data, maxlags, normalize, split_timeseries=False, lstm=False):
        """
        Prepares multivariate time series data such that it can be used by a NAVAR model

        Args:
            data: ndarray
                T (time points) x N (variables) input data
            maxlags: int
                Maximum number of time lags
            normalize: bool
                Indicates whether we should should normalize every variable
            split_timeseries: int
                If the original time series consists of multiple shorter time series, this argument should indicate the
                original time series length. Otherwise should be zero.
            lstm: bool
                Indicates whether we should prepare the data for a LSTM model (or MLP).
        Returns:
            X: Tensor (T - maxlags - 1) x maxlags x N
                Input for the NAVAR model
            Y: Tensor (T - maxlags - 1) x N
                Target variables for the NAVAR model
        """
        # T is the total number of time steps, N is the number of variables
        T, N = data.shape
        data = torch.from_numpy(data)

        #This is new, was not in the original data file. Needed for unbalanced data.
        #it_timeseries is a vector (unbalanced panel), it is only supported for lstm=False.
        if lstm and isinstance(split_timeseries, (list, tuple, np.ndarray)):
            raise ValueError("Variable-length split_timeseries is not supported when lstm=True (requires padding/masks).")
        

        # normalize every variable to have 0 mean and standard deviation 1
        if normalize:
            data = data / torch.std(data, dim=0)
            data = data - data.mean(dim=0)
        
        if not lstm:
            # initialize our input and target variables
            X = torch.zeros((T - maxlags, maxlags, N))
            Y = torch.zeros((T - maxlags, N))

            # X consists of the past K values of Y
            for i in range(T - maxlags - 1):
                X[i, :, :] = data[i:i + maxlags, :]
                Y[i, :] = data[i + maxlags, :]

            # if the data originated from multiple smaller time series, we make sure not to predict over the boundaries.
                        # if the data originated from multiple smaller time series, we make sure not to predict over the boundaries.
            if split_timeseries:
                # ---------------------------------------------------------
                # NEW: split_timeseries can be:
                #   - int (balanced panel, same length each segment)
                #   - list/tuple/np.ndarray of ints (unbalanced panel)
                # ---------------------------------------------------------
                if isinstance(split_timeseries, (list, tuple, np.ndarray)):
                    lengths = np.asarray(split_timeseries, dtype=int)

                    if lengths.ndim != 1:
                        raise ValueError("split_timeseries vector must be 1D")
                    if int(lengths.sum()) != int(T):
                        raise ValueError("split_timeseries vector must sum to total number of rows T")

                    # segment starts: 0, L1, L1+L2, ...
                    starts = np.cumsum(np.r_[0, lengths[:-1]]).astype(int)
                    starts_set = set(starts.tolist())

                    rows_to_be_kept = []
                    for i in range(0, X.shape[0]):
                        # window uses raw indices: inputs [i .. i+maxlags-1], target [i+maxlags]
                        # invalid if any segment start s lies in (i, i+maxlags]
                        crosses_boundary = any((i < s <= i + maxlags) for s in starts_set)
                        if not crosses_boundary:
                            rows_to_be_kept.append(i)

                    rows_to_be_kept = np.asarray(rows_to_be_kept, dtype=int)
                    X = X[rows_to_be_kept]
                    Y = Y[rows_to_be_kept]

                else:
                    # existing balanced behavior (int split_timeseries)
                    split_timeseries = int(split_timeseries)

                    rows_to_be_kept = []
                    for x in range(0, X.shape[0]):
                        to_be_deleted = sum([(x + maxlags - y) % split_timeseries == 0 for y in range(maxlags)]) > 0
                        if not to_be_deleted:
                            rows_to_be_kept.append(x)

                    rows_to_be_kept = np.asarray(rows_to_be_kept, dtype=int)
                    X = X[rows_to_be_kept]
                    Y = Y[rows_to_be_kept]


            X = X.permute(0, 2, 1)
            
        else:
            if split_timeseries:
                # initialize our input and target variables
                X = torch.zeros((int(T/split_timeseries), split_timeseries, N))

                # X and Y consist of timeseries of length K
                for i in range(int(T/split_timeseries) -1):
                    X[i, :, :] = data[i*split_timeseries:(i+1)*split_timeseries, :]
                X = X.permute(0, 2, 1)
                X.view(-1, N, split_timeseries)
                Y = X[:, :, 1:]
                X = X[:, :, :-1]
            else:
                # initialize our input and target variables
                X = torch.zeros((T, maxlags + 1, N))

                # X and Y consist of timeseries of length K
                for i in range(int(T)):
                    for counter, j in enumerate(range(maxlags + 1, 0, -1)):
                        if i - j >= 0:
                            X[i, counter, :] = data[i - j, :]
                X = X.permute(0, 2, 1)
                X.view(-1, N, maxlags+1)
                Y = X[:, :, 1:]
                X = X[:, :, :-1]
            
        return X, Y

    def split_train_val(self, val_proportion):
        """
        Splits the data in a training and validation set. The validation set is the final 'val_proportion' of the
        data.
        Args:
            val_proportion: float
                Proportion of the data set that should be used for validation
        Returns:
            List of Tensors:
                [training_Xs, training_Ys, validation_Xs, validation_Ys]

        """
        #Replace np.int with int or int.64 for precision
        #Then update the environment with a reload:
        number_of_val_indices = int(np.floor(val_proportion * self.all_Ys.shape[0]))
        train_indices = np.arange(self.all_Ys.shape[0] - number_of_val_indices)
        val_indices = np.arange(self.all_Ys.shape[0] - number_of_val_indices, self.all_Ys.shape[0])
        if val_proportion == 0:
            return self.all_Xs, self.all_Ys, None, None
        else:
            return self.all_Xs[train_indices], self.all_Ys[train_indices], \
                   self.all_Xs[val_indices], self.all_Ys[val_indices]