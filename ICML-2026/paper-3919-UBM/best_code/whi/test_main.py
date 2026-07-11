import pytest
import pandas as pd
import numpy as np
from main import run_experiment
from argparse import Namespace
from utils_whi_data import process_ct_df, process_os_df, combine_ct_os, add_target_variables

# Create fixture for processed data
@pytest.fixture
def get_processed_data():
    def _get_processed_data(args):
        ct_df = process_ct_df(args)
        os_df = process_os_df(args)
        ctos = combine_ct_os(ct_df, os_df, args)
        ctos_df, predictors = add_target_variables(ctos, args)
        return ctos_df, predictors
    return _get_processed_data

def test_biased_chd_rf(get_processed_data):
    """Test biased selection with CHD outcome and RF model"""
    args = Namespace(
        selection_flag='biased',
        drop_mechanism='drop_some_excluded',
        censored=True,
        outcome_name='CHD',
        model_type='RF'
    )

    ctos_df, predictors = get_processed_data(args)
    
    # Run experiment
    df = run_experiment(ctos_df, predictors, args, save_file=False)
    print(df)
    # Check expected values
    expected_values = {
        'SE_Y0': {'mean': 0.026004},
        'SE_Y1': {'mean': 0.048222},
        'SE_A': {'mean': -0.045086},
        'SE_S': {'mean': 0.022288}
    }
    
    for metric, values in expected_values.items():
        assert pytest.approx(df.loc[metric, 'mean'], rel=1e-4) == values['mean'], \
            f"Incorrect {metric} mean value"

def test_unbiased_chd_rf(get_processed_data):
    """Test unbiased selection with CHD outcome and RF model"""
    args = Namespace(
        selection_flag='unbiased',
        drop_mechanism='drop_some_excluded',
        censored=True,
        outcome_name='CHD',
        model_type='RF'
    )

    ctos_df, predictors = get_processed_data(args)
    
    # Run experiment
    df = run_experiment(ctos_df, predictors, args, save_file=False)
    
    # Check expected values
    expected_values = {
        'SE_Y0': {'mean': 0.028706},
        'SE_Y1': {'mean': 0.084569},
        'SE_A': {'mean': 0.000615},
        'SE_S': {'mean': -0.028586}
    }
    
    for metric, values in expected_values.items():
        assert pytest.approx(df.loc[metric, 'mean'], rel=1e-4) == values['mean'], \
            f"Incorrect {metric} mean value"

def test_manually_biased_chd_rf(get_processed_data):
    """Test unbiased selection with CHD outcome and RF model"""
    args = Namespace(
        selection_flag='manually_biased',
        drop_mechanism='drop_some_excluded',
        censored=True,
        outcome_name='CHD',
        model_type='RF'
    )

    ctos_df, predictors = get_processed_data(args)
    
    # Run experiment
    df = run_experiment(ctos_df, predictors, args, save_file=False)
    
    # Check expected values
    expected_values = {
        'SE_Y0': {'mean': 0.045295},
        'SE_Y1': {'mean': 0.106375},
        'SE_A': {'mean': 0.219426},
        'SE_S': {'mean': -0.008413}
    }
    
    for metric, values in expected_values.items():
        assert pytest.approx(df.loc[metric, 'mean'], rel=1e-4) == values['mean'], \
            f"Incorrect {metric} mean value"

def test_biased_stroke_rf(get_processed_data):
    """Test biased selection with Stroke outcome and RF model"""
    args = Namespace(
        selection_flag='biased',
        drop_mechanism='drop_some_excluded',
        censored=True,
        outcome_name='STROKE',
        model_type='RF'
    )

    ctos_df, predictors = get_processed_data(args)
    
    # Run experiment
    df = run_experiment(ctos_df, predictors, args, save_file=False)
    
    # Check expected values
    expected_values = {
        'SE_Y0': {'mean': 0.028134},
        'SE_Y1': {'mean': 0.034902},
        'SE_A': {'mean': -0.045724},
        'SE_S': {'mean': 0.048676}
    }
    
    for metric, values in expected_values.items():
        assert pytest.approx(df.loc[metric, 'mean'], rel=1e-4) == values['mean'], \
            f"Incorrect {metric} mean value"
        
def test_unbiased_stroke_rf(get_processed_data):
    """Test biased selection with Stroke outcome and RF model"""
    args = Namespace(
        selection_flag='unbiased',
        drop_mechanism='drop_some_excluded',
        censored=True,
        outcome_name='STROKE',
        model_type='RF'
    )
    
    ctos_df, predictors = get_processed_data(args)

    # Run experiment
    df = run_experiment(ctos_df, predictors, args, save_file=False)
    
    # Check expected values
    expected_values = {
        'SE_Y0': {'mean': 0.027479},
        'SE_Y1': {'mean': 0.107976},
        'SE_A': {'mean': 0.047934},
        'SE_S': {'mean': -0.033383}
    }
    
    for metric, values in expected_values.items():
        assert pytest.approx(df.loc[metric, 'mean'], rel=1e-4) == values['mean'], \
            f"Incorrect {metric} mean value"
        
def test_manually_biased_stroke_rf(get_processed_data):
    """Test biased selection with Stroke outcome and RF model"""
    args = Namespace(
        selection_flag='manually_biased',
        drop_mechanism='drop_some_excluded',
        censored=True,
        outcome_name='STROKE',
        model_type='RF'
    )
    ctos_df, predictors = get_processed_data(args)
    
    # Run experiment
    df = run_experiment(ctos_df, predictors, args, save_file=False)
    
    # Check expected values
    expected_values = {
        'SE_Y0': {'mean': 0.037654},
        'SE_Y1': {'mean': 0.188260},
        'SE_A': {'mean': 0.217989},
        'SE_S': {'mean': 0.028095}
    }
    
    for metric, values in expected_values.items():
        assert pytest.approx(df.loc[metric, 'mean'], rel=1e-4) == values['mean'], \
            f"Incorrect {metric} mean value"
        

def test_censored_false(get_processed_data):
    """Test biased selection with CHD outcome and RF model"""
    args = Namespace(
        selection_flag='biased',
        drop_mechanism='drop_some_excluded',
        censored=False,
        outcome_name='CHD',
        model_type='LR'
    )
    ctos_df, predictors = get_processed_data(args)
    
    # Run experiment
    df = run_experiment(ctos_df, predictors, args, save_file=False)
    print(df)
    # Check expected values
    expected_values = {
        'SE_Y0': {'mean': 0.030793},
        'SE_Y1': {'mean': 0.065783},
        'SE_A': {'mean': -0.064777},
        'SE_S': {'mean': 0.042758}
    }
    
    for metric, values in expected_values.items():
        assert pytest.approx(df.loc[metric, 'mean'], rel=1e-4) == values['mean'], \
            f"Incorrect {metric} mean value"
    
def test_model_lr(get_processed_data):
    """Test biased selection with CHD outcome and RF model"""
    args = Namespace(
        selection_flag='biased',
        drop_mechanism='drop_some_excluded',
        censored=True,
        outcome_name='CHD',
        model_type='LR'
    )
    ctos_df, predictors = get_processed_data(args)
    
    # Run experiment
    df = run_experiment(ctos_df, predictors, args, save_file=False)
    print(df)
    # Check expected values
    expected_values = {
        'SE_Y0': {'mean': 0.033897},
        'SE_Y1': {'mean': 0.057193},
        'SE_A': {'mean': -0.066946},
        'SE_S': {'mean': 0.035653}
    }
    
    for metric, values in expected_values.items():
        assert pytest.approx(df.loc[metric, 'mean'], rel=1e-4) == values['mean'], \
            f"Incorrect {metric} mean value"