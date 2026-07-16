"""
Nonlinear MPC Controller for 2D Kuramoto-Sivashinsky Equation using CasADi + IPOPT
Dual-Grid Architecture: Compresses the N_sim grid into N_mpc to solve natively in IPOPT.
"""

import casadi as ca
import numpy as np
import scipy.sparse as sp
import time
from scipy.ndimage import zoom

class KSMPC2D:
    """
    Nonlinear Model Predictive Controller for the 2D KS Equation using Sparse FD on a Coarse Grid.
    """
    
    def __init__(self, N_sim, N_mpc, L, dt, centers, sigma, horizon,
                 Q=1.0, R=0.01, u_min=-50, u_max=50, terminal_weight=10.0):
        self.N_sim = N_sim
        self.N_mpc = N_mpc
        
        self.N = N_mpc
        self.N2 = N_mpc * N_mpc
        self.L = L
        self.dt = dt
        self.dx = L / self.N_mpc
        
        self.centers = np.array(centers)
        self.sigma = sigma
        self.n_controls = len(centers)
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.terminal_weight = terminal_weight
        self.u_min = u_min
        self.u_max = u_max
        
        self.x = np.linspace(0, L, self.N_mpc, endpoint=False)
        self.y = np.linspace(0, L, self.N_mpc, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        print(f"Building forcing matrix G for N_mpc={N_mpc}...")
        self.G = np.zeros((self.N2, self.n_controls))
        for j, c in enumerate(self.centers):
            dx_dist = np.abs(self.X - c[0])
            dx_dist = np.minimum(dx_dist, L - dx_dist)
            dy_dist = np.abs(self.Y - c[1])
            dy_dist = np.minimum(dy_dist, L - dy_dist)
            dist_sq = dx_dist**2 + dy_dist**2
            self.G[:, j] = np.exp(-0.5 * dist_sq / (sigma**2)).flatten()
            
        self.G_ca = ca.DM(self.G)
        
        print(f"Building sparse dynamics matrices for N_mpc={N_mpc}...")
        self._build_dynamics_matrix()
        
        print("Building CasADi Opti MPC NLP...")
        self._build_mpc()
        print("MPC setup complete.")
    
    def _build_dynamics_matrix(self):
        N = self.N
        dx = self.dx
        dt = self.dt
        N2 = self.N2
        
        # 1D operators
        diagonals = [np.ones(N), -np.ones(N)]
        D1 = sp.diags(diagonals, offsets=[1, -1], shape=(N, N)).tolil()
        D1[0, -1] = -1
        D1[-1, 0] =  1
        D1 = D1.tocsr() / (2 * dx)
        
        diagonals_lap = [np.ones(N), -2 * np.ones(N), np.ones(N)]
        Lap1 = sp.diags(diagonals_lap, offsets=[1, 0, -1], shape=(N, N)).tolil()
        Lap1[0, -1] = 1
        Lap1[-1, 0] = 1
        Lap1 = Lap1.tocsr() / (dx**2)
        
        I1 = sp.eye(N)
        
        # 2D operators using kronecker products
        Dx_sparse = sp.kron(D1, I1, format='csr')
        Dy_sparse = sp.kron(I1, D1, format='csr')
        
        Lap_x = sp.kron(Lap1, I1, format='csr')
        Lap_y = sp.kron(I1, Lap1, format='csr')
        Lap_sparse = Lap_x + Lap_y
        
        Bih_sparse = Lap_sparse.dot(Lap_sparse)
        
        # L = -Laplace - Biharmonic
        L_linear_sparse = -Lap_sparse - Bih_sparse
        
        I_sp = sp.eye(N2, format='csr')
        L_lhs_sp = I_sp - (dt / 2.0) * L_linear_sparse
        L_rhs_sp = I_sp + (dt / 2.0) * L_linear_sparse
        
        # Convert to casadi DM
        self.Dx_ca = ca.DM(Dx_sparse.tocsc())
        self.Dy_ca = ca.DM(Dy_sparse.tocsc())
        self.L_lhs_ca = ca.DM(L_lhs_sp.tocsc())
        self.L_rhs_ca = ca.DM(L_rhs_sp.tocsc())
        
    def _build_mpc(self):
        self.opti = ca.Opti()
        
        N2 = self.N2
        H = self.horizon
        n_ctrl = self.n_controls
        dt = self.dt
        
        self.U = self.opti.variable(N2, H + 1)
        self.A = self.opti.variable(n_ctrl, H)
        
        self.u0_param = self.opti.parameter(N2)
        self.u_ref_param = self.opti.parameter(N2)
        
        cost = 0
        for k in range(H):
            state_err = self.U[:, k+1] - self.u_ref_param
            cost += self.Q * ca.sumsqr(state_err)
            cost += self.R * ca.sumsqr(self.A[:, k])
        
        terminal_err = self.U[:, H] - self.u_ref_param
        cost += self.terminal_weight * self.Q * ca.sumsqr(terminal_err)
        
        self.opti.minimize(cost)
        
        self.opti.subject_to(self.U[:, 0] == self.u0_param)
        
        for k in range(H):
            u_k   = self.U[:, k]
            u_kp1 = self.U[:, k+1]
            a_k   = self.A[:, k]
            
            # Predictor implicit equation:
            # (I - dt/2 * L)*u_kp1 = (I + dt/2 * L)*u_k + dt * f_nonlin
            u_x = ca.mtimes(self.Dx_ca, u_k)
            u_y = ca.mtimes(self.Dy_ca, u_k)
            
            nonlinear_term = -0.5 * (u_x * u_x + u_y * u_y)
            forcing_term = ca.mtimes(self.G_ca, a_k)
            
            lhs = ca.mtimes(self.L_lhs_ca, u_kp1)
            rhs = ca.mtimes(self.L_rhs_ca, u_k) + dt * (nonlinear_term + forcing_term)
            
            self.opti.subject_to(lhs == rhs)
        
        self.opti.subject_to(self.opti.bounded(self.u_min, self.A, self.u_max))
        
        from config import ipopt_options
        self.opti.solver('ipopt', ipopt_options)
        
        self.U_init = None
        self.A_init = None
        
    def solve(self, u0_2d, u_ref_2d, warm_start=True):
        # Downsample the initial condition and reference grid to the internal MPC resolution
        scale_factor = self.N_mpc / self.N_sim
        u0_2d_coarse = zoom(u0_2d, scale_factor, order=3, mode='wrap')
        u_ref_2d_coarse = zoom(u_ref_2d, scale_factor, order=3, mode='wrap')
        
        u0 = u0_2d_coarse.flatten()
        u_ref = u_ref_2d_coarse.flatten()
        
        self.opti.set_value(self.u0_param, u0)
        self.opti.set_value(self.u_ref_param, u_ref)
        
        if warm_start and self.U_init is not None:
            self.opti.set_initial(self.U, self.U_init)
            self.opti.set_initial(self.A, self.A_init)
        else:
            U_guess = np.zeros((self.N2, self.horizon + 1))
            for k in range(self.horizon + 1):
                alpha = k / self.horizon
                U_guess[:, k] = (1 - alpha) * u0 + alpha * u_ref
            
            A_guess = np.zeros((self.n_controls, self.horizon))
            self.opti.set_initial(self.U, U_guess)
            self.opti.set_initial(self.A, A_guess)
        
        try:
            sol = self.opti.solve()
            
            U_opt = sol.value(self.U)
            A_opt = sol.value(self.A)
            
            self.U_init = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])
            temp_A = A_opt[:, 1:] if self.horizon > 1 else np.zeros((self.n_controls, 0))
            self.A_init = np.hstack([temp_A, A_opt[:, -1:]]) if self.horizon > 0 else A_opt
            
            u_next_opt_mpc = U_opt[:, 1].reshape((self.N_mpc, self.N_mpc))
            
            # Upsample back to full scale for external physics plotting
            u_next_opt_2d = zoom(u_next_opt_mpc, self.N_sim / self.N_mpc, order=3, mode='wrap')
            
            return A_opt[:, 0], u_next_opt_2d, A_opt
            
        except Exception as e:
            print(f"MPC solve failed: {e}. Opti status: {self.opti.debug.show_infeasibilities()}")
            try:
                a_opt = self.opti.debug.value(self.A)[:, 0]
                a_opt = np.clip(a_opt, self.u_min, self.u_max)
                return a_opt, None, None
            except:
                print("Falling back to zero control.")
                return np.zeros(self.n_controls), None, None