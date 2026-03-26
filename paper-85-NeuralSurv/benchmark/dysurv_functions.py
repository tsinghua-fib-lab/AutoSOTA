import torch  # For building the networks
from torch import nn
import torch.nn.functional as F
import torchtuples as tt  # Some useful functions

## Code from: https://github.com/munibmesinovic/DySurv/blob/main/Models/Results/DySurv_Final.ipynb


class extract_tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        if tensor.dim() == 2:
            return tensor[:, :]
        return tensor[:, -1, :]


class Decoder(nn.Module):
    def __init__(self, no_features, output_size):
        super().__init__()

        self.no_features = no_features
        self.hidden_size = no_features
        self.output_size = output_size

        self.fc1 = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.fc2 = nn.Linear(3 * self.hidden_size, 5 * self.hidden_size)
        self.fc3 = nn.Linear(5 * self.hidden_size, 3 * self.hidden_size)
        self.fc4 = nn.Linear(3 * self.hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.fc4(x)
        return out


class DySurv(nn.Module):
    def __init__(self, in_features, encoded_features, out_features):
        super().__init__()
        self.fc11 = nn.Linear(in_features, 3 * in_features)
        self.fc12 = nn.Linear(3 * in_features, 5 * in_features)
        self.fc13 = nn.Linear(5 * in_features, 3 * in_features)
        self.fc14 = nn.Linear(3 * in_features, encoded_features)

        self.fc24 = nn.Linear(3 * in_features, encoded_features)

        self.relu = nn.ReLU()

        self.surv_net = nn.Sequential(
            nn.Linear(encoded_features, 3 * in_features),
            nn.ReLU(),
            nn.Linear(3 * in_features, 5 * in_features),
            nn.ReLU(),
            nn.Linear(5 * in_features, 3 * in_features),
            nn.ReLU(),
            nn.Linear(3 * in_features, out_features),
        )

        self.decoder2 = Decoder(encoded_features, in_features)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        sample_z = eps.mul(std).add_(mu)

        return sample_z

    def encoder(self, x):
        x = self.relu(self.fc11(x))
        x = self.relu(self.fc12(x))
        x = self.relu(self.fc13(x))
        mu_z = self.fc14(x)
        logvar_z = self.fc24(x)

        return mu_z, logvar_z

    def forward(self, input):

        mu, logvar = self.encoder(input.float())
        z = self.reparameterize(mu, logvar)
        return self.decoder2(z), self.surv_net(z), mu, logvar

    def predict(self, input):
        # Will be used by model.predict later.
        # As this only has the survival output,
        # we don't have to change LogisticHazard.
        mu, logvar = self.encoder(input)
        encoded = self.reparameterize(mu, logvar)
        return self.surv_net(encoded)


class _Loss(torch.nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction


def _reduction(loss, reduction: str = "mean"):
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    raise ValueError(
        f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'."
    )


def nll_logistic_hazard(
    phi,
    idx_durations,
    events,
    reduction: str = "mean",
    training: bool = True,
):
    """
    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf

    Inputs:
        phi - network output
        idx_durations - tensor, recording the time at which an event or censoring occured. [time_1, time_2, etc]
        events - tensor, indicating whether the event occured (1) or censoring happened (0). [1, 1, 0, 1, 0, etc]
    Output:
        loss - scalar, reducted tensor, of the BCE loss along the time-axis for each patient
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(
            f"Network output `phi` is too small for `idx_durations`."
            + f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"
            + f" but got `phi.shape[1] = {phi.shape[1]}`"
        )

    # Change type of events if necessary
    if events.dtype is torch.bool:
        events = events.float()

    # Change views
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)

    # Creates a target for bce: initialise everything with 0, and setting events at idx_duration
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)

    # Add weighting
    pos_weight = torch.tensor([50.0])

    # Compute BCE
    if training:
        bce = F.binary_cross_entropy_with_logits(
            phi, y_bce, pos_weight=pos_weight, reduction="none"
        )
    else:
        bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction="none")

    # Compute the loss, along the time axis, ie for each patient separately
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)

    # Take the mean or something of the loss in the end
    return _reduction(loss, reduction)


class NLLLogistiHazardLoss(_Loss):
    def forward(self, phi, idx_durations, events):
        return nll_logistic_hazard(
            phi, idx_durations, events, self.reduction, self.training
        )


class Loss(nn.Module):
    def __init__(self, alpha: list):
        super().__init__()
        self.alpha = alpha
        self.loss_surv = NLLLogistiHazardLoss()
        self.loss_ae = nn.MSELoss()

    def forward(self, decoded, phi, mu, logvar, target_loghaz, target_ae):
        """
        Forward call of the Loss Module. Computes the DySurv model loss by combining three weighted losses.
            1. Survival loss: negative log likelihood logistic hazard or BCE loss over the predictions.
            2. AE loss: reconstruction or MSE loss
            3. KL-divergence: KL-divergence or pushing the model to have a latent space close to a normal distribution.
        """
        # Unpack data
        idx_durations, events = target_loghaz

        # Survival Loss
        loss_surv = self.loss_surv(phi, idx_durations, events)

        # AutoEncoder Loss
        loss_ae = self.loss_ae(decoded, target_ae)

        # KL-divergence Loss
        loss_kd = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (
            self.alpha[0] * loss_surv
            + self.alpha[1] * loss_ae
            + self.alpha[2] * loss_kd
        )
