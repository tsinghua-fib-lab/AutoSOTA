from .configs import *
from .lightgbm import LightGBMTuner, LightGBMTunerCV


PARAMS_ZOO = {
    'CatBoostClassifier': catboost_cls_params,
    'CatBoostRegressor': catboost_reg_params,
    'XGBClassifier': xgboost_params,
    'XGBRegressor': xgboost_params, 
    'SVC': svm_params, 
    'SVR': svm_params, 
    'RandomForestClassifier': random_forest_params,
    'RandomForestRegressor': random_forest_params,
}
