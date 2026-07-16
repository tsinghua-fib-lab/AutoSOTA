"""
Nonlinear MPC Controller for 1D Kuramoto-Sivashinsky Equation using CasADi + IPOPT

The controller uses exact dense spatial matrices derived from the 
spectral (Crank-Nicolson / Forward Euler) scheme to model KS dynamics natively in CasADi.
"""

import casadi as ca
import numpy as np

class KSMPC:
    """
    Nonlinear Model Predictive Controller for the 1D KS Equation.
    
    Uses CasADi Opti interface with IPOPT solver.
    """
    
    def __init__(self, N, L, dt, centers, sigma, horizon,
                 Q=1.0, R=0.01, u_min=-50, u_max=50, terminal_weight=10.0):
        """
        Initialize the MPC controller.
        """
        self.N = N
        self.L = L
        self.dt = dt
        self.dx = L / N
        self.centers = np.array(centers)
        self.sigma = sigma
        self.n_controls = len(centers)
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.terminal_weight = terminal_weight
        self.u_min = u_min
        self.u_max = u_max
        
        # Spatial grid
        self.x = np.linspace(0, L, N, endpoint=False)
        
        # Build Gaussian forcing matrix G (N x n_controls)
        self.G = np.zeros((N, self.n_controls))
        for j, c in enumerate(self.centers):
            dist = np.abs(self.x - c)
            dist = np.minimum(dist, L - dist) # Periodic boundaries
            self.G[:, j] = np.exp(-0.5 * (dist / sigma) ** 2)
        self.G_ca = ca.DM(self.G)
        
        # Build dynamics matrices
        self._build_dynamics_matrix()
        
        # Build the MPC optimization problem
        self._build_mpc()
    
    def _build_dynamics_matrix(self):
        """
        Extract exact dense spatial matrices corresponding to the spectral step operations.
        """
        N = self.N
        dx = self.dx
        dt = self.dt
        L = self.L
        
        k = 2 * np.pi * np.fft.rfftfreq(N, d=dx)
        L_linear = k**2 - k**4

        denom = 1.0 - (dt / 2.0) * L_linear
        num_A = 1.0 + (dt / 2.0) * L_linear
        
        diag_A_hat = num_A / denom
        diag_B_hat = dt / denom
        diag_D_hat = -0.5 * (1j * k)

        self.A_dyn = np.zeros((N, N))
        self.B_dyn = np.zeros((N, N))
        self.D_x = np.zeros((N, N))

        for i in range(N):
            e_i = np.zeros(N)
            e_i[i] = 1.0
            
            e_i_hat = np.fft.rfft(e_i)
            
            self.A_dyn[:, i] = np.fft.irfft(diag_A_hat * e_i_hat, n=N)
            self.B_dyn[:, i] = np.fft.irfft(diag_B_hat * e_i_hat, n=N)
            self.D_x[:, i] = np.fft.irfft(diag_D_hat * e_i_hat, n=N)

        # Convert to CasADi
        self.A_ca = ca.DM(self.A_dyn)
        self.B_ca = ca.DM(self.B_dyn)
        self.Dx_ca = ca.DM(self.D_x)
        
    def _dynamics_step(self, u, ctrl):
        """
        One step of the KS dynamics using exact matrix formulation.
        u^{n+1} = A_dyn * u^n + B_dyn * (D_x * (u^n * u^n) + G * ctrl)
        """
        u_sq = u * u
        nonlinear_part = ca.mtimes(self.Dx_ca, u_sq)
        forcing_part = ca.mtimes(self.G_ca, ctrl)
        
        return ca.mtimes(self.A_ca, u) + ca.mtimes(self.B_ca, nonlinear_part + forcing_part)
        
    def _build_mpc(self):
        """Build the MPC optimization problem using CasADi Opti."""
        self.opti = ca.Opti()
        
        N = self.N
        H = self.horizon
        n_ctrl = self.n_controls
        
        # Decision variables
        self.U = self.opti.variable(N, H + 1)  # State trajectory
        self.A = self.opti.variable(n_ctrl, H)  # Control inputs
        
        # Parameters (set at each MPC solve)
        self.u0_param = self.opti.parameter(N)     # Initial state
        self.u_ref_param = self.opti.parameter(N)  # Reference/target state
        
        # Objective function
        cost = 0
        
        for k in range(H):
            # State tracking cost
            state_err = self.U[:, k+1] - self.u_ref_param
            cost += self.Q * ca.sumsqr(state_err)
            
            # Control effort cost
            cost += self.R * ca.sumsqr(self.A[:, k])
        
        # Terminal cost
        terminal_err = self.U[:, H] - self.u_ref_param
        cost += self.terminal_weight * self.Q * ca.sumsqr(terminal_err)
        
        self.opti.minimize(cost)
        
        # Constraints
        # Initial condition
        self.opti.subject_to(self.U[:, 0] == self.u0_param)
        
        # Dynamics constraints
        for k in range(H):
            u_k = self.U[:, k]
            a_k = self.A[:, k]
            u_kp1_pred = self._dynamics_step(u_k, a_k)
            self.opti.subject_to(self.U[:, k+1] == u_kp1_pred)
        
        # Control bounds
        self.opti.subject_to(self.opti.bounded(self.u_min, self.A, self.u_max))
        
        from config import ipopt_options
        self.opti.solver('ipopt', ipopt_options)
        
        # Store initial guess for warm starting
        self.U_init = None
        self.A_init = None
        
    def solve(self, u0, u_ref, warm_start=True):
        """
        Solve the MPC problem for current state and reference.
        
        Args:
            u0: Current state (N,)
            u_ref: Reference/target state (N,)
            warm_start: Whether to use warm starting from previous solution
            
        Returns:
            a_opt: Optimal control for first step (n_controls,)
            U_opt: Optimal state trajectory (N, H+1)
            A_opt: Optimal control trajectory (n_controls, H)
        """
        # Set parameters
        self.opti.set_value(self.u0_param, u0)
        self.opti.set_value(self.u_ref_param, u_ref)
        
        # Initialize with better guess
        if warm_start and self.U_init is not None:
            self.opti.set_initial(self.U, self.U_init)
            self.opti.set_initial(self.A, self.A_init)
        else:
            # Cold start: linearly interpolate from u0 to u_ref
            U_guess = np.zeros((self.N, self.horizon + 1))
            for k in range(self.horizon + 1):
                alpha = k / self.horizon
                U_guess[:, k] = (1 - alpha) * u0 + alpha * u_ref
            
            # Use pseudo-inverse of G for naive initial control guess
            A_guess = np.zeros((self.n_controls, self.horizon))
            self.opti.set_initial(self.U, U_guess)
            self.opti.set_initial(self.A, A_guess)
        
        try:
            sol = self.opti.solve()
            
            U_opt = sol.value(self.U)
            A_opt = sol.value(self.A)
            
            # Store for warm start (shift by one step to future)
            self.U_init = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])
            temp_A = A_opt[:, 1:] if self.horizon > 1 else np.zeros((self.n_controls, 0))
            self.A_init = np.hstack([temp_A, A_opt[:, -1:]]) if self.horizon > 0 else A_opt
            
            return A_opt[:, 0], U_opt, A_opt
            
        except Exception as e:
            print(f"MPC solve failed: {e}. Opti status: {self.opti.debug.show_infeasibilities()}")
            try:
                # Still try to extract current best solution even if infeasible (e.g. max iter reached)
                a_opt = self.opti.debug.value(self.A)[:, 0]
                a_opt = np.clip(a_opt, self.u_min, self.u_max)
                return a_opt, None, None
            except:
                print("Falling back to zero control.")
                return np.zeros(self.n_controls), None, None
