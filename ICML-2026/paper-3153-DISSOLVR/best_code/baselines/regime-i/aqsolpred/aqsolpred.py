# -*- coding: utf-8 -*-
"""
Created on 04-03-2020 23:39:16 2020

@author:  Murat Cihan Sorkun

This code trains and test only one case for given configuration (removes test data from training)
"""

import pandas as pd
import numpy as np 
import utilities
import datetime
import time
import xgboost
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import preprocess

start = time.time()

#Load Train/Test data 
train_set_df = pd.read_csv("../data/dataset-not-FA.csv")
test_set_df = pd.read_csv("../data/dataset-E.csv")


#Preprocess test and train sets
all_descriptors_train_df_all,element_list=preprocess.preprocess_train(train_set_df,test_set_df)
all_descriptors_test_df_all,test_smiles_list=preprocess.preprocess_test(test_set_df,element_list)
train_logS_list=all_descriptors_train_df_all["LogS"]
test_logS_list=all_descriptors_test_df_all["LogS"]

#Select descriptors by LASSO
selected_data_train, selected_data_test = preprocess.select_features_lasso(all_descriptors_train_df_all,all_descriptors_test_df_all)

#Train and Test with Neural Nets
mlp_model = MLPRegressor(activation='tanh', hidden_layer_sizes=(500), max_iter=500, random_state=0, solver='adam')
mlp_model.fit(selected_data_train, train_logS_list)   
pred_mlp = mlp_model.predict(selected_data_test)
utilities.get_errors(test_logS_list,pred_mlp,"Neural Nets")

#Train and Test with XGBoost
xgboost_model = xgboost.XGBRegressor(n_estimators=1000) 
xgboost_model.fit(selected_data_train, train_logS_list)    
pred_xgb = xgboost_model.predict(selected_data_test)
utilities.get_errors(test_logS_list,pred_xgb,"XGBoost")

#Train and Test with Random Forest
rf_model = RandomForestRegressor(random_state=0, n_estimators=1000) 
rf_model.fit(selected_data_train, train_logS_list)   
pred_rf = rf_model.predict(selected_data_test)   
utilities.get_errors(test_logS_list,pred_rf,"Random Forest")

#Calculate Consensus results
pred_consensus=(pred_mlp+pred_xgb+pred_rf)/3
utilities.get_errors(test_logS_list,pred_consensus,"Consensus")

time_counsumed = time.time()-start
print("Total time used:", time_counsumed)  

#Create DF for results 
results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus,test_logS_list])
df_results = pd.DataFrame(results, columns=['Neural Nets','Xgboost','Random Forest','Consensus','Target'])

#Calculate STD of 3 predictions
preds_all=df_results[['Neural Nets','Xgboost','Random Forest']].values
std=np.std(preds_all, axis = 1)
df_results["STD"]=std    
df_results["SMILES"]=test_smiles_list

#Write results into a file
df_results=df_results.round(3)
df_results.to_csv("../results/results.csv",index=False)



 
    

    


