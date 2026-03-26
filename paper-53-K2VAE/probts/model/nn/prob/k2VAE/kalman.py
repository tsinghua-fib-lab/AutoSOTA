'''
* @author: EmpyreanMoon
*
* @create: 2025-01-02 14:28
*
* @description: The main structure of kalman filter
'''
from torch.distributions import MultivariateNormal

import torch
import torch.nn as nn
import torch.nn.functional as F

class KalmanFilter(nn.Module):
    def __init__(self, state_dim, init='identity'):
        """
        Batch-parallel Kalman filter with learnable kernel for Q and R
        Args:
            state_dim (int): Dimension of the state vector
            init_sigma (float): Initial bandwidth of the kernel function
        """
        super(KalmanFilter, self).__init__()
        self.state_dim = state_dim

        # Initialize Kalman filter parameters
        if init == 'identity':
            self.B = nn.Parameter(torch.eye(state_dim, state_dim))  # Control input matrix B
            self.F = nn.Parameter(torch.eye(state_dim, state_dim))  # State transition matrix F
        else:
            self.B = nn.Parameter(torch.randn(state_dim, state_dim))  # Control input matrix B
            self.F = nn.Parameter(torch.randn(state_dim, state_dim))  # State transition matrix F

        self.H = nn.Parameter(torch.eye(state_dim, state_dim))  # Observation matrix H
        # Learnable covariance matrices Q and R
        self.LQ = nn.Parameter(torch.tril(torch.eye(state_dim, state_dim)))
        self.LR = nn.Parameter(torch.tril(torch.eye(state_dim, state_dim)))

    def compute_Q_R(self, batch_size):
        """
        Compute process noise covariance Q and measurement noise covariance R
        Args:
            batch_size (int): Number of parallel batches
        Returns:
            Q (torch.Tensor): Process noise covariance matrix [B, state_dim, state_dim]
            R (torch.Tensor): Measurement noise covariance matrix [B, state_dim, state_dim]
        """
        Q = self.LQ @ self.LQ.T  # Ensure Q is positive semi-definite
        R = self.LR @ self.LR.T  # Ensure R is positive semi-definite
        Q = Q.unsqueeze(0).repeat(batch_size, 1, 1)  # Expand Q for batch size
        R = R.unsqueeze(0).repeat(batch_size, 1, 1)  # Expand R for batch size
        return Q, R

    def one_step(self, x_t_prev, u_t, y_t, P_t):
        """
        Perform one step of Kalman filtering
        Args:
            x_t_prev (torch.Tensor): State vector at the previous time step [B, state_dim]
            u_t (torch.Tensor): Control input at the current time step [B, state_dim]
            y_t (torch.Tensor): Measurement at the current time step [B, state_dim]
            P_t (torch.Tensor): Covariance matrix at the previous time step [B, state_dim, state_dim]
        Returns:
            x_t (torch.Tensor): Updated state estimate [B, state_dim]
            P_t (torch.Tensor): Updated covariance matrix [B, state_dim, state_dim]
        """
        batch_size = x_t_prev.shape[0]

        # Compute process noise Q and measurement noise R
        Q, R = self.compute_Q_R(batch_size)

        # 1. Prediction step
        x_t_pred = torch.matmul(x_t_prev, self.F.T) + torch.matmul(u_t, self.B.T)  # [B, state_dim]
        P_t_pred = torch.matmul(self.F, torch.matmul(P_t, self.F.T)) + Q  # [B, state_dim, state_dim]

        # 2. Update step
        H_P_Ht = torch.matmul(self.H, torch.matmul(P_t_pred, self.H.T))  # [B, state_dim, state_dim]
        S = H_P_Ht + R  # [B, state_dim, state_dim]
        S_inv = torch.linalg.inv(S)  # Inverse of S, [B, state_dim, state_dim]
        K_t = torch.matmul(P_t_pred, torch.matmul(self.H.T, S_inv))  # Kalman gain, [B, state_dim, state_dim]

        # Measurement update
        y_pred = torch.matmul(x_t_pred, self.H.T)  # Predicted measurement, [B, state_dim]
        x_t = x_t_pred + torch.matmul((y_t - y_pred).unsqueeze(1), K_t).squeeze(1)  # Updated state [B, state_dim]

        # Joseph stabilized form for P_t
        I = torch.eye(self.state_dim, device=x_t.device).unsqueeze(0).repeat(batch_size, 1, 1)  # Identity matrix
        K_H = torch.matmul(K_t, self.H)  # [B, state_dim, state_dim]
        I_minus_K_H = I - K_H  # [B, state_dim, state_dim]
        P_t = torch.matmul(I_minus_K_H, torch.matmul(P_t_pred, I_minus_K_H.transpose(1, 2))) + \
              torch.matmul(K_t, torch.matmul(R, K_t.transpose(1, 2)))  # Joseph stabilized covariance update
        return x_t, P_t

    def keep_positive_definite(self, tensor):
        """
           Perform positive definite correction on a tensor with shape [B, pred_len, state_dim, state_dim].
           :param tensor: Input tensor with shape [B, pred_len, state_dim, state_dim]
           :return: Corrected positive definite tensor with the same shape as the input
           """
        tensor = tensor + tensor.transpose(-1, -2)

        eigvals, eigvecs = torch.linalg.eigh(tensor)

        eigvals_clamped = torch.clamp(eigvals, min=1e-6)  # [B, pred_len, state_dim]

        fixed_tensor = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-1, -2)

        fixed_tensor = fixed_tensor + fixed_tensor.transpose(-1, -2)
        return fixed_tensor

    def forward(self, x, U_t, Y_t):
        """
        Perform Kalman filtering over a sequence of time steps
        Args:
            x (torch.Tensor): Initial state estimate [B, state_dim]
            U_t (torch.Tensor): Control inputs over time [B, seq_len, state_dim]
            Y_t (torch.Tensor): Measurements over time [B, seq_len, state_dim]
        Returns:
            prediction (torch.Tensor): Predicted states over time [B, seq_len, state_dim]
            dist (MultivariateNormal): Multivariate normal distribution for predicted states
        """
        batch_size, seq_len, _ = U_t.shape
        P_t = torch.eye(self.state_dim, device=x.device).unsqueeze(0).repeat(batch_size, 1,
                                                                             1)  # Initial covariance [B, state_dim, state_dim]

        prediction = []
        covariance = []

        # Iterate over the sequence
        for t in range(seq_len):
            x, P_t = self.one_step(x, U_t[:, t, :], Y_t[:, t, :], P_t)
            P_t = 0.5 * (P_t + P_t.transpose(-2, -1))
            prediction.append(x)
            covariance.append(P_t)

        # Stack predictions and covariances
        prediction = torch.stack(prediction, dim=1)  # [B, pred_len, state_dim]
        covariance = torch.stack(covariance, dim=1)  # [B, pred_len, state_dim, state_dim]

        # Create a multivariate normal distribution for the predictions
        try:
            dist = MultivariateNormal(loc=torch.zeros_like(prediction).to(x.device), covariance_matrix=covariance)
        except:
            covariance = self.keep_positive_definite(covariance)
            dist = MultivariateNormal(loc=torch.zeros_like(prediction).to(x.device), covariance_matrix=covariance)

        return prediction, dist

