from pycaret.regression import *
from utils.plot_regr import plot_graph_regr
from utils.regression import prep_and_train_regr, tuning_regr, pred_regr
import streamlit as st
import pandas as pd 
from streamlit_pandas_profiling import st_profile_report
from utils.get_profile import get_profile
from datetime import datetime


def pretrain_regr(regr_dic):
    with st.container():
            with st.expander('Dataset'):
                    try:
                        st.dataframe(st.session_state.df)
                    except AttributeError:
                        pass
            with st.expander('Dataset main report'):
                if st.checkbox('Huge Dataset'):
                    speedup = 'data/config_minimal.yaml'
                else:
                    speedup= 'data/config_default.yaml'
                try:
                    st.session_state.df
                except AttributeError:
                    st.warning('Please Load Dataset')
                if st.button('Generate report'):
                    try:
                        report = get_profile(st.session_state.df, speedup)
                        export=report.to_html()
                        st.download_button(label="Download Full Report", data=export, file_name=f'report-{datetime.now().strftime("%Y_%m_%d")}.html')
                        st_profile_report(report)
                    except NameError:
                        st.error('Please upload dataset first')
            col1, col2 = st.columns([3,1.5])
            with col2:
                st.subheader('Choose Parameters', anchor=False)
                try:
                    st.session_state.targ_regr = st.selectbox('Choose target', st.session_state.df.columns)
                except AttributeError:
                    pass
                with st.expander('Params'):
                    st.session_state.model_regr = st.multiselect('Choose model',
                                            ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par','ransac', 'tr',
                                              'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada',
                                                'gbr', 'mlp', 'xgboost', 'lightgbm', 'dummy'],
                                            help='''Here you can choose what **model** of Maсhine learning to use.  
                                                    If you want to train all models at once, leave this field blank.''', 
                                            format_func=lambda x: regr_dic.get(x))
                    
                    train_size = st.number_input('Training Size:', value=0.7, help= 'Proportion of the dataset to be used for training and validation. Should be between 0.0 and 1.0.')

                    data_split_stratify = st.checkbox("Controls Stratification during Split", 
                                                       value=False, 
                                                       help='''Controls stratification during "train_test_split". 
                                                               When set to True, will stratify by target column.''')

                    fold_strategy = st.selectbox('Choice of Cross Validation Strategy',
                                                  options=['kfold','stratifiedkfold'], 
                                                  help='Choice of cross validation strategy.')

                    fold = st.number_input('Number of Folds to be Used in Cross Validation',
                                            min_value=2,
                                            value=10, 
                                            help='''Number of folds to be used in cross validation. Must be at least 2. 
                                                    Increasing this value **improves** the performance of the trained models, but also takes **longer** time to train.''')
                    
                    remove_outliers = st.checkbox('Remove outliers', 
                                                   value=False, 
                                                   help='When set to True, outliers from the training data are removed using an Isolation Forest.')

                with st.expander('Inputation and Normalisation'):

                    numeric_imputation = st.selectbox('Missing Value for Numeric Columns', 
                                                       options=['mean','median','mode'], 
                                                       help='Imputing strategy for numerical columns.')

                    normalize = st.checkbox('Normalization', 
                                             value=False, 
                                             help='When set to True, it transforms the features by scaling them to a given range.')

                    normalize_method = 'zscore'
                    if normalize:
                        normalize_method = st.selectbox('Method to be used for Normalization',
                                                         options=['zscore','minmax','maxabs','robust'], 
                                                         help='''Defines the method for scaling. 
                                                                 By default, normalize method is set to "zscore" 
                                                                 The standard zscore is calculated as z = (x - u) / s.''') 
                with st.expander('Feature selection'):

                    feature_selection = st.checkbox('Select a Subset of Features Using a Combination of various Permutation Importance', 
                                                     value=False, 
                                                     help='When set to True, a subset of features is selected based on a feature importance score')
                    feature_selection_method = 'classic'
                    if feature_selection:
                        feature_selection_method= st.selectbox('Algorithm for feature selection',
                                                                options=['classic','univariate','sequential'], 
                                                                help='Algorithm for feature selection')
                    remove_multicollinearity = st.checkbox('Remove Highly Linearly Correlated Features', 
                                                            value=False, 
                                                            help='''When set to True, features with the inter-correlations higher than the defined threshold are removed. 
                                                                    For each group, it removes all except the feature with the highest correlation to y.''')
                    multicollinearity_threshold = 0.9
                    if remove_multicollinearity:
                        multicollinearity_threshold = st.number_input('Threshold Used for Dropping the Correlated Features', 
                                                                       min_value=0.0, 
                                                                       value=0.9, 
                                                                       help='Minimum absolute Pearson correlation to identify correlated features. The default value removes equal columns.')
                if st.button('Try model'):
                    try:
                            # st.session_state.best_clf = None
                            st.session_state.best_regr, st.session_state.model_info_regr, st.session_state.metrics_info_regr = prep_and_train_regr(
                                st.session_state.targ_regr, st.session_state.df, st.session_state.model_regr, 
                                train_size, data_split_stratify, fold_strategy, fold, numeric_imputation,
                                normalize,normalize_method,feature_selection,feature_selection_method,
                                remove_multicollinearity,multicollinearity_threshold, remove_outliers
                                )
                            
                            # save_model(st.session_state.best, 'dt_pipeline')
                            with col1:
                                st.subheader('Actual Model')
                                st.session_state.model_info_last_regr = st.session_state.model_info_regr
                                st.session_state.metrics_info_last_regr = st.session_state.metrics_info_regr
                                col1, col2 = st.columns([3.5,1.8])
                                with col1:
                                    st.dataframe(st.session_state.metrics_info_regr)
                                with col2:
                                    st.dataframe(st.session_state.model_info_regr)     
                    except ValueError:
                                st.error('Please choose target with binary labels')
                else:
                    try:
                        with col1:
                            st.subheader('Your last teached model')
                            col1, col2 = st.columns([3.5,1.8])
                            with col1:
                                st.dataframe(st.session_state.metrics_info_last_regr)
                            with col2:
                                st.dataframe(st.session_state.model_info_last_regr)
                    except AttributeError: 
                        st.error('Teach your first model')
    st.divider()
    with st.container():
        col1,col2 = st.columns([2,1])
        with col2:
            st.subheader('Choose parameters for plots')
            st.session_state.plot_params_regr = st.multiselect('Choose model',
                                        ['residuals','error','cooks'],
                                        help='Select plots to build for detailed analysis of the model')
            if st.button('Plot'):
                with col1:
                    plot_graph_regr(st.session_state.best_regr, st.session_state.plot_params_regr)

def tunalyse_regr():
    st.title('Choose parameters to tune your model')
    st.subheader('Current model')
    try:
        st.table(st.session_state.metrics_info_last_regr.head(1))
    except AttributeError:
        pass
    col1, col2 = st.columns([2,4])
    with col2:
        option_regr = st.selectbox(
        'Choose the tuning engines',
        ('scikit-learn', 'optuna', 'scikit-optimize'))
        optimizer_regr = st.selectbox('Choose metric to optimize', ('MAE','MSE','RMSE'), 
                                       help='Metric name to be evaluated for hyperparameter tuning.')
        
        st.session_state.iters_regr = st.slider('n_estimators', 5, 20, 5, 1, 
                                                 help='Number of iterations in the grid search. Increasing "n_iter" may improve model performance but also increases the training time.')  
        if st.button('Tune'):
            # clf2 = setup(data = st.session_state.df, target = st.session_state.targ_regr, session_id=2)
            st.session_state.tuned_dt_regr, st.session_state.info_df_regr = tuning_regr(model=st.session_state.best_regr,n_iters=st.session_state.iters_regr,opti=optimizer_regr, search_lib=option_regr)
            # save_model(st.session_state.tuned_dt_regr, 'tuned_regr')
            st.write('Last best params')
            st.code(st.session_state.tuned_dt_regr)
        with col1:
            try:
                st.dataframe(st.session_state.info_df_regr)
                # st.write('Last best params')
                # st.code(st.session_state.tuned_dt)
            except (AttributeError, NameError):
                st.warning('Prepare and train model first')

def predict_regr():
    try:
        holdout_pred = pred_regr(st.session_state.best_regr)
    except AttributeError:
        st.warning('Teach model first')
    file_res = st.file_uploader("Upload Your Datasets")
    if file_res: 
        test_df = pd.read_csv(file_res, index_col=None)
    if st.button('Predict on test data'):
        try:
            st.dataframe(holdout_pred)
        except UnboundLocalError:
            pass
    if st.button('Predict on example'):
        # testing_cian = pd.read_csv('examples/test.csv')
        result_pred = pred_regr(st.session_state.best_regr, data=test_df)
        st.dataframe(result_pred)
        # result = result_pred
        # result.to_csv('result.csv')
         