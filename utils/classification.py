from pycaret.classification import *
import streamlit as st

# def setup_clf(data, target, session_id, 
#               train_sizes, data_split_stratify, 
#               fold_strategy, fold, numeric_imputation,
#               normalize,normalize_methods,fix_imbalance,
#               feature_selection,feature_selection_method,
#               remove_multicollinearity,multicollinearity_threshold,
#               pca,pca_method,pca_components,polynomial_features,polynomial_degree):
    
#     clf = setup(data=data, 
#                 target=target, 
#                 session_id=session_id, 
#                 train_size=train_sizes, 
#                 data_split_stratify=data_split_stratify, 
#                 fold_strategy=fold_strategy, 
#                 fold=fold,
#                 numeric_imputation=numeric_imputation,
#                 normalize=normalize,
#                 normalize_method=normalize_methods,
#                 fix_imbalance=fix_imbalance,
#                 feature_selection=feature_selection,
#                 feature_selection_method=feature_selection_method,
#                 remove_multicollinearity=remove_multicollinearity,
#                 multicollinearity_threshold=multicollinearity_threshold,
#                 pca=pca, pca_method=pca_method, pca_components=pca_components,
#                 polynomial_features=polynomial_features, polynomial_degree=polynomial_degree)
#     s_df = pull()
#     return clf, s_df

@st.cache_resource
def prep_and_train_clf(target,data, models,
                       train_sizes, data_split_stratify, 
                       fold_strategy, fold, numeric_imputation,
                       normalize,normalize_methods,fix_imbalance,
                       feature_selection,feature_selection_method,
                       remove_multicollinearity,multicollinearity_threshold
                       ):
    clf = setup(data=data, 
                target=target,  
                train_size=train_sizes, 
                data_split_stratify=data_split_stratify, 
                fold_strategy=fold_strategy, 
                fold=fold,
                numeric_imputation=numeric_imputation,
                normalize=normalize,
                normalize_method=normalize_methods,
                fix_imbalance=fix_imbalance,
                feature_selection=feature_selection,
                feature_selection_method=feature_selection_method,
                remove_multicollinearity=remove_multicollinearity,
                multicollinearity_threshold=multicollinearity_threshold
                )
    s_df = pull()
    best = compare_models(include=models)
    best_df = pull()
    return best, best_df, s_df

def tuning_clf(model, n_iters, search_lib, opti):
    tuned_dt = tune_model(estimator=model, n_iter=n_iters, search_library=search_lib, choose_better=True, optimize=opti)
    return tuned_dt

def pred_clf(model,data=None):
    res = predict_model(estimator=model, data=data)
    return res





