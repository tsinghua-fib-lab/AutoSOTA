import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class BaseUnbiasedSetup:
    def __init__(self, n_rct=1000, n_obs=10000, n_covs=1, n_unmeasured_covs=1, random_seed=42):
        self.n_rct = n_rct
        self.n_obs = n_obs
        self.n_covs = n_covs
        self.n_unmeasured_covs = n_unmeasured_covs
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.covs = [f'X{i+1}' for i in range(n_covs)]
        self.unmeasured_covs = [f'U{i+1}' for i in range(n_unmeasured_covs)]
        

    def generate_data(self):
        # Generate RCT data
        X_rct = np.random.choice([-1, 1], size=(self.n_rct, self.n_covs), p=[0.5, 0.5])
        U_rct = np.random.choice([-1, 1], size=(self.n_rct, self.n_unmeasured_covs), p=[0.5, 0.5])
        
        df_rct = pd.DataFrame({
            **{cov: X_rct[:,i] for i, cov in enumerate(self.covs)},
            **{u_cov: U_rct[:,i] for i, u_cov in enumerate(self.unmeasured_covs)},
            **{'R': 1}
        })
        
        # Generate observational data
        X_obs = np.random.choice([-1, 1], size=(self.n_obs, self.n_covs), p=[0.5, 0.5])
        U_obs = np.random.choice([-1, 1], size=(self.n_obs, self.n_unmeasured_covs), p=[0.5, 0.5])
        
        df_obs = pd.DataFrame({
            **{cov: X_obs[:,i] for i, cov in enumerate(self.covs)},
            **{u_cov: U_obs[:,i] for i, u_cov in enumerate(self.unmeasured_covs)},
            **{'R': 0}
        })
        
        return df_rct, df_obs
    

    def generate_treatment_outcome_selection(self, df, study="RCT"): 
        if study == "RCT":
            fn_tr = self._generate_rct_treatment
            fn_sel = self._generate_rct_selection
        else: 
            fn_tr = self._generate_obs_treatment 
            fn_sel = self._generate_obs_selection

        fn_out = self._generate_outcome

        df["S"] = df.apply(fn_sel, axis=1)
        df["A"] = df.apply(fn_tr, axis=1)
        df["Y"] = df.apply(fn_out, axis=1)

        return df
    

    def _generate_rct_treatment(self, row):
        return np.random.binomial(1, 0.5)


    def _generate_obs_treatment(self, row):
        if row["X1"] == -1:
            return np.random.binomial(1, 0.1)
        else:
            return np.random.binomial(1, 0.9)
    

    def _generate_rct_selection(self, row):
        return 1
    

    def _generate_obs_selection(self, row): 
        if row["X1"] == -1:
            return np.random.binomial(1, 0.1)
        else: 
            return np.random.binomial(1, 0.9)
    

    def _generate_outcome(self, row):
        if row["A"] == 0:
            return row["X1"] + np.random.normal(0, 0.1)
        else:
            return 2 + row["X1"] + np.random.normal(0, 0.1)
            
            
    def fit_models(self, df_rct, df_obs, models={"outcome": "DTR", 
                                                 "selection": "DTC", 
                                                 "treatment": "DTC", 
                                                 "study": "DTC"}):
        # Fit models on RCT data
        self._fit_outcome_model(df_rct, models["outcome"])
        self._fit_ps_model(df_rct, models["treatment"])
        
        # Fit models on obs data
        self._fit_selection_model(df_obs, models["selection"])
        df_obs = df_obs[df_obs["S"]==1].reset_index(drop=True)
        self._fit_outcome_model(df_obs, models["outcome"])
        self._fit_ps_model(df_obs, models["treatment"])
        
        # Merge and fit remaining model
        df_merged = pd.concat([df_rct, df_obs])
        self._fit_rm_model(df_merged, models["study"])
        
        return df_merged
    
    def _fit_outcome_model(self, df, model="DTR"):
        if model == "DTR":
            om_A0 = DecisionTreeRegressor().fit(df.query("A==0")[self.covs], df.query("A==0")["Y"])
            om_A1 = DecisionTreeRegressor().fit(df.query("A==1")[self.covs], df.query("A==1")["Y"])
        
        df["hat_Y0"] = om_A0.predict(df[self.covs])
        df["hat_Y1"] = om_A1.predict(df[self.covs])
        
        df.loc[df["A"]==0, "SE_Y0"] = (df.loc[df["A"]==0, "Y"] - df.loc[df["A"]==0, "hat_Y0"]) ** 2
        df.loc[df["A"]==1, "SE_Y1"] = (df.loc[df["A"]==1, "Y"] - df.loc[df["A"]==1, "hat_Y1"]) ** 2
    
    def _fit_ps_model(self, df, model="DTC"):
        if model == "DTC":
            psm = DecisionTreeClassifier().fit(df[self.covs], df["A"])

        df["hat_P(A=1)"] = psm.predict_proba(df[self.covs])[:,1]
        df["BCELoss_A"] = -df["A"] * np.log(df["hat_P(A=1)"]) - (1 - df["A"]) * np.log(1 - df["hat_P(A=1)"])
        df["SE_A"] = (df["A"] - df["hat_P(A=1)"]) ** 2
    
    def _fit_selection_model(self, df, model="DTC"):
        if model == "DTC":
            sm = DecisionTreeClassifier().fit(df[self.covs], df["S"])
            
        df["hat_P(S=1)"] = sm.predict_proba(df[self.covs])[:,1]
        df["BCELoss_S"] = -df["S"] * np.log(df["hat_P(S=1)"]) - (1 - df["S"]) * np.log(1 - df["hat_P(S=1)"])
        df["SE_S"] = (df["S"] - df["hat_P(S=1)"]) ** 2
    
    def _fit_rm_model(self, df, model="DTC"):
        if model == "DTC":
            rm = DecisionTreeClassifier().fit(df[self.covs], df["R"])
            
        df["hat_P(R=1)"] = rm.predict_proba(df[self.covs])[:,1]


class SelectionBiasType1or2(BaseUnbiasedSetup):
            
    def _generate_obs_selection(self, row):
        # need to change this to depend on A and Y
        if row["A"] == 0: 
            return np.random.binomial(1, 0.1)
        else: 
            pXY = 1 / (1 + np.exp(-row["Y"]))
            if row["X1"] == 1: 
                return np.random.binomial(1, pXY)
            else: 
                return np.random.binomial(1, 1 - pXY)


class SelectionBiasType3(BaseUnbiasedSetup):
            
    def _generate_obs_selection(self, row):
        if row["X1"] == -1:
            return np.random.binomial(1, 0.1)
        else: 
            if row["U1"] == -1:
                return np.random.binomial(1, 0.95)
            else: 
                return np.random.binomial(1, 0.05)
            
    def _generate_outcome(self, row):
        if row["A"] == 0:
            return row["X1"] + np.random.normal(0, 0.1)
        else:
            if row["X1"] == -1:
                return 2 + row["X1"] + np.random.normal(0, 0.1)
            else:
                return 2 + row["X1"] + row["U1"] + np.random.normal(0, 1)