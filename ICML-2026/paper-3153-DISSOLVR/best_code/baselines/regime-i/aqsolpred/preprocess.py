# -*- coding: utf-8 -*-
"""
Created on 04-03-2020 23:39:16 2020

@author:  Murat Cihan Sorkun

This file contains preprocess functions for solubility data
"""

import pandas as pd
import numpy as np 
from rdkit import Chem
import utilities
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import mordred
from mordred import Calculator, descriptors


def preprocess_train(train_set_df,test_set_df):

    #Get test data to remove from the train set
    test_inchiKey_list = test_set_df["InChIKey"].values    
    
    train_formula_list=[]
    train_logS_list=[]
    train_mordred_descriptors=[]

    
    # Train data filter
    for index, row in train_set_df.iterrows():
        smiles=row['SMILES']
        logS=row['Solubility']
        inchiKey=row['InChIKey']
        mol=Chem.MolFromSmiles(smiles)
        mol=Chem.AddHs(mol)
        formula=Chem.rdMolDescriptors.CalcMolFormula(mol)
        formula=formula.replace("+","")
        formula=formula.replace("-","")
        
        if(inchiKey not in test_inchiKey_list):
            if("." not in smiles):
                if(utilities.has_element(formula,"C")):
                    if(("+" not in smiles) and ("-" not in smiles)):
                        train_formula_list.append(formula)
                        train_logS_list.append(logS)
                        train_mordred_descriptors.append(get_mordred(mol))

            
    print("Training set size after preprocessing:",len(train_logS_list))
      
    
    #Get Mordred descriptor names
    column_names=get_mordred(Chem.MolFromSmiles("CC"),True)+["LogS"]
    
    unfiltered_train_np=np.column_stack([train_mordred_descriptors,train_logS_list])
    unfiltered_train_df=pd.DataFrame(index=train_formula_list, data=unfiltered_train_np,columns=column_names)
    # Drop columns with nan (only MIN and MAX values of estates has nan values)
    unfiltered_train_df = unfiltered_train_df[unfiltered_train_df.columns.drop(list(unfiltered_train_df.filter(regex='MIN')))]
    unfiltered_train_df = unfiltered_train_df[unfiltered_train_df.columns.drop(list(unfiltered_train_df.filter(regex='MAX')))]
    # Remove columns having lower occurrences then the threshold
    remove_threshold=20
    drop_cols = unfiltered_train_df.columns[(unfiltered_train_df == 0).sum() > (unfiltered_train_df.shape[0]-remove_threshold) ]
    unfiltered_train_df.drop(drop_cols, axis = 1, inplace = True) 
    
    # Convert values to numeric
    train_df = unfiltered_train_df.apply(pd.to_numeric)
    
    #element count vector
    all_elements_list = utilities.create_element_list(train_formula_list)    
    element_vector_df=utilities.create_oneHotVector(train_formula_list,all_elements_list)
    element_vector_df["LogS"]=train_logS_list
    
    # Combine all descriptors for all and best descriptors
    train_descriptors = pd.concat([train_df.drop(columns=['LogS']),element_vector_df], axis=1)
    
    return train_descriptors, all_elements_list



def preprocess_test(test_set_df,element_list):
    
    test_smiles_list=[]
    test_formula_list=[]
    test_logS_list=[]
    test_mordred_descriptors=[]
    
    # Test data filter
    for index, row in test_set_df.iterrows():
        smiles=row['SMILES']
        logS=row['Solubility']
        mol=Chem.MolFromSmiles(smiles)
        mol=Chem.AddHs(mol)
        formula=Chem.rdMolDescriptors.CalcMolFormula(mol)
        formula=formula.replace("+","")
        formula=formula.replace("-","")
        
        if("." not in smiles):
            if(utilities.has_element(formula,"C")):
                if(("+" not in smiles) and ("-" not in smiles)):
                    test_smiles_list.append(smiles)
                    test_formula_list.append(formula)
                    test_logS_list.append(logS)
                    test_mordred_descriptors.append(get_mordred(mol))
    
    print("Test size after preprocessing:",len(test_logS_list))
    
    #Get Mordred descriptor names
    column_names=get_mordred(Chem.MolFromSmiles("CC"),True)+["LogS"]
    
    unfiltered_test_np=np.column_stack([test_mordred_descriptors,test_logS_list])
    unfiltered_test_df=pd.DataFrame(index=test_formula_list, data=unfiltered_test_np,columns=column_names)
    
    # Create element count vector 
    test_element_vector_df=utilities.create_oneHotVector(test_formula_list,element_list)
    test_element_vector_df["LogS"]=test_logS_list
    
    # Combine all descriptors
    test_descriptors = pd.concat([unfiltered_test_df.drop(columns=['LogS']),test_element_vector_df], axis=1)
    
    return test_descriptors, test_smiles_list


def select_features_lasso(train_data,test_data):
    """
    Selects descriptors from training data then applies on both training and test data 
    """
    
    lasso = Lasso(alpha=0.01,max_iter=10000,random_state=1).fit(train_data.drop(columns=['LogS']), train_data['LogS'])
    model = SelectFromModel(lasso, prefit=True)
    X_new_lasso = model.transform(train_data.drop(columns=['LogS']))
    # Get back the kept features as a DataFrame with dropped columns as all 0s
    selected_features = pd.DataFrame(model.inverse_transform(X_new_lasso), index=train_data.drop(columns=['LogS']).index, columns=train_data.drop(columns=['LogS']).columns)
    # Dropped columns have values of all 0s, keep other columns 
    selected_columns_lasso = selected_features.columns[selected_features.var() != 0]
    
    selected_data_train = train_data[selected_columns_lasso]             
    selected_data_test = test_data[selected_columns_lasso]
    selected_data_test = selected_data_test.apply(pd.to_numeric)
    
    print(selected_data_train.columns)
    print("Total selected descriptors by LASSO:",len(selected_data_train.columns))
      
    return selected_data_train, selected_data_test


#returns mordred descriptor vector
def get_mordred(mol, desc_names=False):   
    """
    Generates predefined descriptors for given mol object(Rdkit)
    If desc_names is True then returns only the list of the descriptor name
    """
    
    calc1 = mordred.Calculator()    

    calc1.register(mordred.AtomCount.AtomCount("X"))
    calc1.register(mordred.AtomCount.AtomCount("HeavyAtom"))
    calc1.register(mordred.Aromatic.AromaticAtomsCount)
    
    calc1.register(mordred.HydrogenBond.HBondAcceptor)
    calc1.register(mordred.HydrogenBond.HBondDonor)
    calc1.register(mordred.RotatableBond.RotatableBondsCount)  
    calc1.register(mordred.BondCount.BondCount("any", False))
    calc1.register(mordred.Aromatic.AromaticBondsCount)  
    calc1.register(mordred.BondCount.BondCount("heavy", False))       
    calc1.register(mordred.BondCount.BondCount("single", False))
    calc1.register(mordred.BondCount.BondCount("double", False))
    calc1.register(mordred.BondCount.BondCount("triple", False))
      
    calc1.register(mordred.McGowanVolume.McGowanVolume)
    calc1.register(mordred.TopoPSA.TopoPSA(True))
    calc1.register(mordred.TopoPSA.TopoPSA(False))
    calc1.register(mordred.MoeType.LabuteASA)
    calc1.register(mordred.Polarizability.APol)
    calc1.register(mordred.Polarizability.BPol)
    calc1.register(mordred.AcidBase.AcidicGroupCount)
    calc1.register(mordred.AcidBase.BasicGroupCount)
    calc1.register(mordred.EccentricConnectivityIndex.EccentricConnectivityIndex)        
    calc1.register(mordred.TopologicalCharge.TopologicalCharge("raw",1))
    calc1.register(mordred.TopologicalCharge.TopologicalCharge("mean",1))
    
    calc1.register(mordred.SLogP)
    calc1.register(mordred.BertzCT.BertzCT)
    calc1.register(mordred.BalabanJ.BalabanJ)
    calc1.register(mordred.WienerIndex.WienerIndex(True))
    calc1.register(mordred.ZagrebIndex.ZagrebIndex(1,1))
    calc1.register(mordred.ABCIndex)
    
    calc1.register(mordred.RingCount.RingCount(None, False, False, None, None))
    calc1.register(mordred.RingCount.RingCount(None, False, False, None, True))
    calc1.register(mordred.RingCount.RingCount(None, False, False, True, None))
    calc1.register(mordred.RingCount.RingCount(None, False, False, True, True))
    calc1.register(mordred.RingCount.RingCount(None, False, False, False, None))
    calc1.register(mordred.RingCount.RingCount(None, False, True, None, None))

    calc1.register(mordred.EState)

        
# if desc_names is "True" returns only name list
    if(desc_names):
        name_list=[]
        for desc in calc1.descriptors:
            name_list.append(str(desc))
        return name_list
#        return list(calc1._name_dict.keys())
    else: 
        result = calc1(mol)
        return result._values


    


