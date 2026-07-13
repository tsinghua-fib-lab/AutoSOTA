# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:34:16 2019

@author:  Murat Cihan Sorkun

This file contains usefull functions that generate and plot the descriptors
"""

import pandas as pd
import re
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# Create list for all elements dataset includes
def create_element_list(formula_list):
    all_elements_list = []
    for formula in formula_list:
        element_list = re.findall('[A-Z][^A-Z]*', formula)
        for elem in element_list:
            element, number = split_number(elem)
            if element not in all_elements_list:
                all_elements_list.append(element)

    return all_elements_list

# parses elements and number of occurences from element-number couple
def split_number(text):

    element = text.rstrip('0123456789')
    number = text[len(element):]
    if number == "":
        number = 1
    return element, int(number)


#counts number of element occurences from the formula list     
def count_elements(formula_list,all_elements_list):

    element_counts=[0]*len(all_elements_list)
    for formula in formula_list:
        element_list=re.findall('[A-Z][^A-Z]*', formula)
        for elem in element_list:
            element, number = split_number(elem)
            element_index=all_elements_list.index(element)
            element_counts[element_index]=element_counts[element_index]+1
    
    return element_counts

#parses element symbols and number of occurences from the formula list and create one hot vector    
def create_oneHotVector(formula_list,all_elements_list):

    vectors = []   
    for formula in formula_list:
        element_vector=[0]*len(all_elements_list)
        element_list=re.findall('[A-Z][^A-Z]*', formula)
        for elem in element_list:
            element, number = split_number(elem)
            try:
                element_index=all_elements_list.index(element)
                element_vector[element_index]=number
            except:
                print(element+ " not found in:"+ formula)
        vectors.append(element_vector)    
    oneHotVector_df=pd.DataFrame(index = formula_list, data=vectors, columns=all_elements_list)
    
    return oneHotVector_df


#parses element symbols and number of occurences from the formula list and create one hot vector    
def filter_by_elements(formula,all_elements_list):

    current_element_list=re.findall('[A-Z][^A-Z]*', formula)
    for elem in current_element_list:
        element, number = split_number(elem)
        if(element not in all_elements_list):
            return False
    
    return True

def has_element(formula,element):

    element_list=re.findall('[A-Z][^A-Z]*', formula)
    for elem in element_list:
        current_element, number = split_number(elem)
        if(current_element == element):
            return True
    
    return False

def get_errors(y_true,y_pred,model_name="Model"):   

    err_mae=mae(y_true,y_pred)
    err_rmse=np.sqrt(mse(y_true,y_pred))
    err_r2=r2(y_true,y_pred)
        
    print(model_name+" MAE:"+str(err_mae)+" RMSE:"+str(err_rmse)+" R2:"+str(err_r2))
  
    return err_mae,err_rmse,err_r2


def plot_corr(data,size=(40,18)):
 
    corr = data.corr()
    plt.figure(figsize=size)
    sns.heatmap(np.round(corr,2), annot=True,center=0,cmap="RdBu_r")
