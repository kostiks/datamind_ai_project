import streamlit as st
from streamlit_option_menu import option_menu
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os 
import pandas as pd
from pycaret.classification import *
from classification import prep_and_train
import time
from get_profile import get_profile
from classification import tuning
import numpy as np


with st.sidebar: #Side bar config
    selected = option_menu(menu_title=None,options=["Home", 'Classification','Regression', 'Time Series'], 
        icons=['house', 'file-binary','graph-up','bezier2'], menu_icon="cast", default_index=0)
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        st.session_state.df = pd.read_csv(file, index_col=None)
        st.session_state.df.to_csv('dataset.csv', index=None)


if selected == 'Home':
    nones = ['None' for i in range(5)]
    section= option_menu(None, ["Info", "Data profile",'AutoML'], 
    default_index=0, icons=nones,orientation="horizontal")
    if section== 'Info':
        st.title('Main info about service')
        st.write('Some Text about service')
    if section == 'Data profile':
        st.title('This section will give you main information about uploaded dataset')
        st.write('Simply click "Generate new profile" if you want to generate new profile data and click "View old report to load previous profile"')
        if st.checkbox('Huge Dataset'):
            speedup = 'config_minimal.yaml'
        else:
            speedup= 'config_default.yaml'
        if st.button('Generate report'):
            try:
                st_profile_report(get_profile(st.session_state.df, speedup))
            except NameError:
                st.error('Please upload dataset first')

if selected == 'Classification':
    section = option_menu(None, ["Prep & Train",'Tune & Analyse','Predict'], 
    default_index=0,icons=['1-square','1-square','1-square'],orientation="horizontal")
    if section == 'Prep & Train':
        st.title("Prepare you data and train best model")
        col1, col2 = st.columns([3,1.6])
        with col2:
            try:
                    st.session_state.targ = st.selectbox('Choose target', st.session_state.df.columns)
                    dic ={
                        'lr':'LogReg',
                        'ridge':'Ridge Classifier',
                        'lda':'Linear Discriminant Analysis',
                        'et':'Extra Trees Classifier',
                        'nb':'Naive Bayes',
                        'qda':'Quadratic Discriminant Analysis',
                        'rf':'Random Forest Classifier',
                        'gbc':'Gradient Boosting Classifier',
                        'lightgbm':'Light Gradient Boosting Machine',
                        'catboost':'CatBoost Classifier',
                        'ada':'Ada Boost Classifier',
                        'dt':'Decision Tree Classifier',
                        'knn':'K Neighbors Classifier',
                        'dummy':'Dummy Classifier',
                        'svm':'SVM - Linear Kernel'
                    }
                    np.save('classif_dic.npy', dic) 

                    model = st.multiselect('Choose model',
                                        ['lr',
                                            'ridge',
                                            'lda',
                                            'et',
                                            'nb',
                                            'qda',
                                            'rf',
                                            'gbc',
                                            'lightgbm',
                                            'catboost',
                                            'ada',
                                            'dt',
                                            'knn',
                                            'dummy',
                                            'svm'], help='Blablabla', format_func=lambda x: dic.get(x))
                    if st.button('Try model'):
                        try:
                            st.session_state.best, st.session_state.model_info, st.session_state.metrics_info = prep_and_train(st.session_state.targ, st.session_state.df, model)
                            save_model(st.session_state.best, 'dt_pipeline')
                            # model_info.to_csv('model_info.csv', index=None)
                            # metrics_info.to_csv('metrics_info.csv',index=None)
                            with col1:
                                st.subheader('Actual Model')
                                st.session_state.model_info_last = st.session_state.model_info
                                st.session_state.metrics_info_last = st.session_state.metrics_info
                                col1, col2 = st.columns([3.5,1.8])
                                with col1:
                                    st.dataframe(st.session_state.metrics_info)
                                with col2:
                                    st.dataframe(st.session_state.model_info)     
                        except ValueError:
                                st.error('Please choose target with binary labels')
                    else:
                            try:
                                with col1:
                                    st.subheader('Your last teached model')
                                    col1, col2 = st.columns([3.5,1.8])
                                    with col1:
                                        st.dataframe(st.session_state.metrics_info_last)
                                    with col2:
                                        st.dataframe(st.session_state.model_info_last)
                            except AttributeError:
                                st.write('teach the first model')
            except AttributeError: 
                st.error('Please load dataset first')
            # if st.button('plot'):
            #     col1,col2,col3 = st.columns(3)
            #     with col1:
            #         plot_model(st.session_state.best, plot = 'auc', display_format='streamlit')
            #     with col2:
            #         plot_model(st.session_state.best, plot = 'threshold', display_format='streamlit')
            #     with col3:
            #         plot_model(st.session_state.best, plot = 'confusion_matrix', display_format='streamlit')        

    if section == 'Tune & Analyse':
        st.title('Choose parameters to tune your model')
        st.subheader('Current model')
        st.table(st.session_state.metrics_info_last.head(1))
        col1, col2 = st.columns([2,4])
        with col2:
            option = st.selectbox(
            'Choose the tuning engine',
            ('scikit-learn', 'optuna', 'scikit-optimize'))
            st.session_state.optimize = st.selectbox('Choose metric to optimize', ('Accuracy','AUC','F1'))
            st.session_state.iters = st.slider('n_estimators', 5, 20, 5, 1)  
            if st.button('Tune'):
                clf1 = setup(data = st.session_state.df, target = st.session_state.targ)
                st.session_state.tuned_dt = tune_model(estimator=st.session_state.best,n_iter=st.session_state.iters,choose_better=True,optimize=st.session_state.optimize)
                st.session_state.info_df = pull()
            with col1:
                try:
                    st.dataframe(st.session_state.info_df)
                    st.write('Last best params')
                    st.write(st.session_state.tuned_dt)
                except AttributeError:
                    pass






            


            

