from pycaret.regression import *
import streamlit as st

@st.cache_resource
def prep_and_train_regr(target,_data, models,
                       train_sizes, data_split_stratify, 
                       fold_strategy, fold, numeric_imputation,
                       normalize,normalize_methods,feature_selection,feature_selection_method,
                       remove_multicollinearity,multicollinearity_threshold, remove_outliers
                       ):
    regr = setup(data=_data, 
                target=target,  
                train_size=train_sizes, 
                data_split_stratify=data_split_stratify, 
                fold_strategy=fold_strategy, 
                fold=fold,
                numeric_imputation=numeric_imputation,
                normalize=normalize,
                normalize_method=normalize_methods,
                feature_selection=feature_selection,
                feature_selection_method=feature_selection_method,
                remove_multicollinearity=remove_multicollinearity,
                multicollinearity_threshold=multicollinearity_threshold, remove_outliers=remove_outliers
                )
    r_df = pull()
    best = compare_models(include=models)
    best_df = pull()
    return best, r_df, best_df



def tuning_regr(model, n_iters, search_lib, opti):
    tuned_dt = tune_model(estimator=model, n_iter=n_iters, search_library=search_lib, choose_better=True, optimize=opti)
    info_df = pull()
    return tuned_dt, info_df

def setup_regr(data, target, session_id):
    clf = setup(data=data, target=target, session_id=session_id)
    return clf

def pred_regr(model,data=None):
    res = predict_model(estimator=model, data=data)
    return res



