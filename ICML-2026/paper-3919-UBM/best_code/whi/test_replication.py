import pytest
import pandas as pd
import numpy as np
from main import run_experiment
from argparse import Namespace
from utils_whi_data import process_ct_df, process_os_df, combine_ct_os, add_target_variables
from utils_models_v2 import get_hr

# Create fixture for processed data
@pytest.fixture
def get_processed_data():
    def _get_processed_data(args):
        ct_df = process_ct_df(args)
        os_df = process_os_df(args)
        ctos = combine_ct_os(ct_df, os_df, args)
        predictors = ['AGE', 'ETHNIC_White', 'EDUC_Some post-graduate or professional', 
          'EDUC_Some college or Associate Degree', 'BMI', 'SMOKING_Past Smoker', 
          'SMOKING_Current Smoker', 'MENO', 'PHYSFUN']
        return ctos, predictors
    return _get_processed_data

def test_HR_chd_rct(get_processed_data):
    """Test replication of CHD HR from trial data"""

    args = Namespace(
        selection_flag='biased',
        drop_mechanism='drop_some_excluded',
        censored=True,
        outcome_name='CHD',
        model_type='RF'
    )
    ctos_df, _ = get_processed_data(args)
    print(ctos_df.columns)
    ct_df = ctos_df.query('OS == 0 & S_GLBL == 1')
    ct_df_chd = ct_df[['HRTARM', 'CHD_E', 'CHD_DY']]
    ct_df_chd = ct_df_chd[ct_df_chd['CHD_DY'].notna()]
    HR, lower, upper = get_hr(ct_df_chd, 'CHD_DY', 'CHD_E', 'CHD')
    output = {
        'HR': HR,
        'lower': lower,
        'upper': upper 
    }

    # Check expected values
    expected_values = {
        'HR': 1.28, 
        'lower': 1.01,
        'upper': 1.61
    }
    
    for metric, value in expected_values.items():
        assert pytest.approx(output[metric], rel=1e-2) == value, \
            f"Incorrect {metric} value for {args.outcome_name}"

def test_HR_chd_os_biased(get_processed_data):
    """Test replication of CHD HR from OS data pre-correction"""

    args = Namespace(
        selection_flag='biased',
        drop_mechanism='drop_all_excluded',
        censored=True,
        outcome_name='CHD',
        model_type='RF'
    )
    ctos_df, predictors = get_processed_data(args)

    os_df = ctos_df.query('OS == 1 & S_CHD == 1')
    treatment = ['HRTARM']
    events = ['CHD_E', 'CHD_DY']
    event_name = 'CHD'
    os_df_sub = os_df[predictors + treatment + events]
    os_df_sub = os_df_sub[os_df_sub[events[1]].notna()]

    HR, lower, upper = get_hr(os_df_sub, 
                              events[1], 
                              events[0], 
                              event_name, 
                              HR_cov='HRTARM', 
                              study_type='Observational Study')
    output = {
        'HR': HR,
        'lower': lower,
        'upper': upper 
    }
    
    # Check expected values
    expected_values = {
        'HR': 0.87, 
        'lower': 0.73,
        'upper': 1.03
    }
    
    for metric, value in expected_values.items():
        assert pytest.approx(output[metric], rel=1e-2) == value, \
            f"Incorrect {metric} value for {args.outcome_name}"
        
def test_HR_chd_os_unbiased(get_processed_data):
    """Test replication of CHD HR from OS data post-correction"""

    args = Namespace(
        selection_flag='unbiased',
        drop_mechanism='drop_all_excluded',
        censored=True,
        outcome_name='CHD',
        model_type='RF'
    )
    ctos_df, predictors = get_processed_data(args)

    os_df = ctos_df.query('OS == 1 & S_CHD == 1')
    treatment = ['HRTARM']
    events = ['CHD_E', 'CHD_DY']
    event_name = 'CHD'
    os_df_sub = os_df[predictors + treatment + events]
    os_df_sub = os_df_sub[os_df_sub[events[1]].notna()]

    HR, lower, upper = get_hr(os_df_sub, 
                              events[1], 
                              events[0], 
                              event_name, 
                              HR_cov='HRTARM', 
                              study_type='Observational Study')
    output = {
        'HR': HR,
        'lower': lower,
        'upper': upper 
    }
    
    # Check expected values
    expected_values = {
        'HR': 1.09, 
        'lower': 0.81,
        'upper': 1.45
    }
    
    for metric, value in expected_values.items():
        assert pytest.approx(output[metric], rel=1e-2) == value, \
            f"Incorrect {metric} value for {args.outcome_name}"

def test_HR_stroke_rct(get_processed_data):
    """Test replication of stroke HR from trial data"""

    args = Namespace(
        selection_flag='biased',
        drop_mechanism='drop_some_excluded',
        censored=True,
        outcome_name='STROKE',
        model_type='RF'
    )
    ctos_df, _ = get_processed_data(args)
    
    ct_df = ctos_df.query('OS == 0 & S_STROKE == 1')
    ct_df_stroke = ct_df[['HRTARM', 'STROKE_E', 'STROKE_DY']]
    ct_df_stroke = ct_df_stroke[ct_df_stroke['STROKE_DY'].notna()]
    HR, lower, upper = get_hr(ct_df_stroke, 'STROKE_DY', 'STROKE_E', 'STROKE')
    output = {
        'HR': HR,
        'lower': lower,
        'upper': upper 
    }
    
    # Check expected values
    expected_values = {
        'HR': 1.37, 
        'lower': 1.04,
        'upper': 1.8
    }
    
    for metric, value in expected_values.items():
        assert pytest.approx(output[metric], rel=1e-2) == value, \
            f"Incorrect {metric} value for {args.outcome_name}"
    
def test_HR_stroke_os_biased(get_processed_data):
    """Test replication of STROKE HR from OS data pre-correction"""

    args = Namespace(
        selection_flag='biased',
        drop_mechanism='drop_all_excluded',
        censored=True,
        outcome_name='STROKE',
        model_type='RF'
    )
    ctos_df, predictors = get_processed_data(args)

    os_df = ctos_df.query('OS == 1 & S_STROKE == 1')
    treatment = ['HRTARM']
    events = ['STROKE_E', 'STROKE_DY']
    event_name = 'STROKE'
    os_df_sub = os_df[predictors + treatment + events]
    os_df_sub = os_df_sub[os_df_sub[events[1]].notna()]

    HR, lower, upper = get_hr(os_df_sub, 
                              events[1], 
                              events[0], 
                              event_name, 
                              HR_cov='HRTARM', 
                              study_type='Observational Study')
    output = {
        'HR': HR,
        'lower': lower,
        'upper': upper 
    }
    
    # Check expected values
    expected_values = {
        'HR': 0.86, 
        'lower': 0.71,
        'upper': 1.04
    }
    
    for metric, value in expected_values.items():
        assert pytest.approx(output[metric], rel=1e-2) == value, \
            f"Incorrect {metric} value for {args.outcome_name}"

def test_HR_stroke_os_unbiased(get_processed_data):
    """Test replication of STROKE HR from OS data post-correction"""

    args = Namespace(
        selection_flag='unbiased',
        drop_mechanism='drop_all_excluded',
        censored=True,
        outcome_name='STROKE',
        model_type='RF'
    )
    ctos_df, predictors = get_processed_data(args)

    os_df = ctos_df.query('OS == 1 & S_STROKE == 1')
    treatment = ['HRTARM']
    events = ['STROKE_E', 'STROKE_DY']
    event_name = 'STROKE'
    os_df_sub = os_df[predictors + treatment + events]
    os_df_sub = os_df_sub[os_df_sub[events[1]].notna()]

    HR, lower, upper = get_hr(os_df_sub, 
                              events[1], 
                              events[0], 
                              event_name, 
                              HR_cov='HRTARM', 
                              study_type='Observational Study')
    output = {
        'HR': HR,
        'lower': lower,
        'upper': upper 
    }
    
    # Check expected values
    expected_values = {
        'HR': 1.29, 
        'lower': 0.95,
        'upper': 1.75
    }
    
    for metric, value in expected_values.items():
        assert pytest.approx(output[metric], rel=1e-2) == value, \
            f"Incorrect {metric} value for {args.outcome_name}"

