import numpy as np

class CovarianceEstimator: 
    def __init__(self): 
        self.N  = None
        self.df = None  
        self.use_witness_function = False
    
    def _compute_term2_helper(self, x1, x2, average=True):
        cross_products = x1[:,None] * x2[None,:]
        mask = ~np.eye(len(x1), dtype=bool)
        if average:
            return cross_products[mask].mean()
        else:
            return cross_products[mask]

    def _compute_covariance_SE_outcome(self):
        N          = self.N 
        trt_df     = self.df.query("A==1")
        control_df = self.df.query("A==0")
        
        if self.use_witness_function:
            trt_w = trt_df["w(X)"].values
            control_w = control_df["w(X)"].values
        else:
            trt_w = np.abs(trt_df["psi"].values)
            control_w = np.abs(control_df["psi"].values)

        term1 = ((control_df["SE_Y0"]*control_w).sum() + 
                (trt_df["SE_Y1"]*trt_w).sum())
        term1 = term1 / N 
        
        # term 2
        c1 = control_df["SE_Y0"].values; c2 = control_w
        t1 = trt_df["SE_Y1"].values; t2 = trt_w

        # Compute cross products for control group
        c_cross = self._compute_term2_helper(c1, c2, average=False)
        # Compute cross products for treatment group
        t_cross = self._compute_term2_helper(t1, t2, average=False)
        # Combine and compute mean
        term2 = np.mean(np.concatenate([c_cross, t_cross]))
        
        return (N / (N-1)) * (term1 - term2)

    def _compute_covariance_SE_treatment(self):
        N = self.N 

        if self.use_witness_function:
            w = self.df["w(X)"].values
        else: 
            w = np.abs(self.df["psi"].values)

        term1 = (self.df["SE_A"]*w).mean()
        # term 2 
        x1 = self.df["SE_A"].values; x2 = w
        term2 = self._compute_term2_helper(x1, x2, average=True)
        # covariance 
        return (N / (N-1)) * (term1 - term2) 

    def _compute_covariance_SE_selection(self):
        N = self.N 
        if self.use_witness_function:
            w = self.df["w(X)"].values
        else: 
            w = np.abs(self.df["psi"].values)

        term1 = (self.df["SE_S"]*w).mean()
        x1 = self.df["SE_S"].values; x2 = w
        term2 = self._compute_term2_helper(x1, x2, average=True)
        return (N / (N-1)) * (term1 - term2)

    def compute_covariance_SE(self, df, use_witness_function=False):
        # set shape and dataframe
        self.N  = df.shape[0]
        self.df = df
        self.use_witness_function = use_witness_function

        # compute covariance signals 
        covariance_signals = {}
        covariance_signals["outcome"] = self._compute_covariance_SE_outcome()
        covariance_signals["treatment"] = self._compute_covariance_SE_treatment()
        covariance_signals["selection"] = self._compute_covariance_SE_selection()
        return covariance_signals

class PsiEstimator: 
    def __init__(self):
        pass  
    
    def psi0(self, row):
        if row["R"] == 1:
            return 0
        else:
            t_ind = row["A"]
            p_r1, p_t1 = row["hat_P(R=1)"], row["hat_P(A=1)"]
            y, mu0, mu1 = row["Y"], row["hat_Y0"], row["hat_Y1"]

            term_1 = mu1 - mu0
            term_2 = t_ind * (y - mu1) / p_t1
            term_3 = (1 - t_ind) * (y - mu0) / (1 - p_t1)

            return (term_1 + term_2 - term_3)/ (1 - p_r1)


    def psi1(self, row):
        if row["R"] == 0:
            return 0
        else:
            t_ind = row["A"]
            p_r1, p_t1 = row["hat_P(R=1)"], row["hat_P(A=1)"]
            y, mu0, mu1 = row["Y"], row["hat_Y0"], row["hat_Y1"]

            term_1 = mu1 - mu0
            term_2 = t_ind * (y - mu1) / p_t1
            term_3 = (1 - t_ind) * (y - mu0) / (1 - p_t1)

            return (term_1 + term_2 - term_3) / p_r1
    

    def calc_contrasts(self, df):
        df["psi0"] = df.apply(self.psi0, axis=1)
        df["psi1"] = df.apply(self.psi1, axis=1)
        df["psi"] = df["psi1"] - df["psi0"]