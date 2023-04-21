import streamlit as st
from streamlit_option_menu import option_menu
# from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# import os
from utils.clf_funcs import pretrain_clf
# import test_clf 
import pandas as pd
from pycaret.classification import *
# from utils.classification import prep_and_train_clf
# from utils.regression import prep_and_train_regr
from datetime import datetime
from utils.get_profile import get_profile
# from utils.classification import tuning_clf, pred_clf
# from utils.regression import tuning_regr, setup_regr, pred_regr
from utils.clf_funcs import *
from utils.regr_funs import *
import numpy as np
from utils.time_series import train_ts
from utils.plot_ts import plot_graph_ts
# from utils.plot_regr import plot_graph_regr
# from utils.plot_clf import plot_graph_clf
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(layout="wide")

classif_dictionary = np.load('data/classif_dic.npy',allow_pickle='TRUE').item()
regr_dic = np.load('data/regr_dic.npy',allow_pickle='TRUE').item()


with st.sidebar: #Side bar 
    selected = option_menu(menu_title=None,options=["Home",'AutoMl', 'Classification','Regression'], 
        icons=['house','gear','file-binary','graph-up'], menu_icon="cast", default_index=0)
    
# with st.sidebar: #Side bar 
#     selected = option_menu(menu_title=None,options=["Home", 'Classification','Regression'], 
#         icons=['house', 'file-binary','graph-up','bezier2'], menu_icon="cast", default_index=0)
#     st.title("Upload Your Dataset")
#     file = st.file_uploader("Upload Your Dataset")
#     if file: 
#         st.session_state.df = pd.read_csv(file, index_col=None)
    

if selected == 'Home':
    # with st.sidebar:
    #     if st.button('Load Classification example'):
    #         st.session_state.df = pd.read_csv('examples/Titanic.csv')
    #         st.write('Titanic.csv loaded')
    #     if st.button('Load regression example'):
    #         st.session_state.df = pd.read_csv('examples/House_prices.csv')
    #         st.write('House_pricing.csv loaded')
            st.image('images/logo.png')
            st.subheader('A machine learning service that helps you to quickly and effortlessly get acquainted with the data and build a baseline model')
            col1, col2 = st.columns([2,3])
            with col1:
                st.header('Easiest way(AutoML)')

                st.subheader('**Step 1:**')
                st.write('Upload your data for traning Machine Learning model, and the data you want to predict on the left sidebar')
                st.subheader('**Step 2:**')
                st.write('Click "Go fast baseline" to train your model and get the predict')
                st.subheader('**Step 3:**')
                st.write('Analyse your data with short features description and plots')
            with col2:
                st.video('images/streamlit-main-2023-04-20-23-04-81.webm')
            

            with col1:
                st.markdown('##')
                st.markdown('##')
                st.markdown('##')
                st.header('Deeper way(Classification & Regression)')
                st.subheader('**Step 1:**')
                st.write('Choose your task kind with panel of the left')
                st.subheader('**Step 2:**')
                st.write('Upload your data set with panel of the left or use example with "Load example" button')
                st.subheader('**Step 3:**')
                st.write('Choose Parameters that you need with panel of the right')
                st.subheader('**Step 4:**')
                st.write('Tap "Try model" model button, look at results and if you need you can draw some plots below')
                st.subheader('**Step 5:**')
                st.write('If you want you can tune your model at the "Tune & Analyse" tab')  
                st.subheader('**Step 6:**')
                st.write('Finaly you can see predict results at the "Predict" tab and also you can input new data and get preditions on it')
            with col2:
                st.markdown('##')
                st.video('images/streamlit-main-2023-04-20-23-04-81.webm')
        
if selected  == 'AutoMl':
    with st.sidebar:
        MODELS_DICT = {
        'Regression': {
            'class_type': RegressionExperiment(),
            'models': ['svm', 'xgboost', 'lightgbm', 'rf', 'ridge', 'knn', 'dummy']
            }, 
        'Classification': {
            'class_type': ClassificationExperiment(),
            'models': ['svm', 'xgboost', 'lightgbm', 'rf', 'ridge', 'knn', 'dummy']
            }
    }

    class Data:
        
        def __init__(self, dataframe=None, dataframe_test=None, target=None, problem_type=None, date_column=None):
            self.dataframe = dataframe
            self.dataframe_test = dataframe_test
            self.target = target
            self.problem_type = problem_type
            self.date_column = date_column

        def get_vars(self):
            return self.dataframe, self.dataframe_test, self.target, self.problem_type, self.date_column

        def read_file(self, file):
            if file.type == 'text/plain':
                dataframe = pd.read_csv(file, delim_whitespace=True, header=None, prefix='column_')
            elif file.type == 'text/csv':
                for sep in [',', ';', '\t']:
                    file.seek(0)
                    dataframe = pd.read_csv(file, sep=sep)
                    if dataframe.shape[1] > 1:
                        break
            elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                dataframe = pd.read_excel(file)
            else:
                st.write('The file should be only in .csv, .txt or .xlsx format!')

            first_column = dataframe.columns[0]
            if (dataframe[first_column] == dataframe.index).all() | (dataframe[first_column]-1 == dataframe.index).all():
                dataframe = dataframe.drop(first_column, axis=1)

            return dataframe
        
        def upload_test(self):
            st.sidebar.markdown('If you want to recieve prediction, please upload test dataset in .csv, .txt or .xlsx format')
            uploaded_file_test = st.sidebar.file_uploader("Choose a file", type = ['csv', 'txt', 'xlsx'], key='uploader_1')
            if uploaded_file_test:
                self.dataframe_test = self.read_file(uploaded_file_test)

        def upload_file(self):
            st.sidebar.markdown('Upload file in .csv, .txt or .xlsx format')
            uploaded_file = st.sidebar.file_uploader("Choose a file", type = ['csv', 'txt', 'xlsx'], key='uploader_0')
            self.upload_test()
            if uploaded_file:
                self.dataframe = self.read_file(uploaded_file)
                
                col1, col2 = st.columns([5,1])
                with col1:
                    st.markdown('## Dataframe')
                    st.write(self.dataframe)

                none_values = (
                    self.dataframe[self.dataframe.columns[self.dataframe.isna().any()]]
                    .isna().sum()
                    .sort_values(ascending=False)
                    .rename('None counter')
                    )
                
                with col2:
                    st.markdown('## None values')
                    st.write(none_values)

                    for col in list(none_values.index):
                        if self.dataframe[col].isna().sum() == self.dataframe.shape[0]:
                            self.dataframe = self.dataframe.drop(col, axis=1)

                self.problem_type = st.sidebar.selectbox('Please, choose the problem type.', ('Regression', 'Classification'))
                    
                self.target = st.sidebar.selectbox('Choose the target', self.dataframe.columns)

        def make_corr_plot(self):
            corr = self.dataframe.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig_corr = go.Figure(
                go.Heatmap(
                    z=corr.mask(mask),
                    x=corr.columns,
                    y=corr.columns,
                    colorscale=px.colors.diverging.RdBu,
                    zmin=-1,
                    zmax=1
                    )
                )
            st.markdown('## Correlation matrix')
            st.plotly_chart(fig_corr, theme="streamlit", use_container_width=True)

        def make_hist_plot(self):
            st.markdown('## Target distribution')
            target_hist_fig = go.Figure()
            target_hist_fig.add_trace(
                go.Histogram(
                    x=self.dataframe[self.target]
                )
            )
            target_hist_fig.update_layout(
                title=dict(text=self.target, font_size=40, x=0.5, xanchor='center')
            )
            st.plotly_chart(target_hist_fig, theme="streamlit", use_container_width=True)

            st.markdown('## Features distribution')
            with st.spinner('Wait for it...'):
                with st.expander("Tap to expand"):
                    cols_drop = [self.target]
                    for col in self.dataframe.columns:
                        if (self.dataframe[col].dtype == 'object') & (self.dataframe[col].nunique()/self.dataframe.shape[0] > 0.05):
                            cols_drop.append(col)

                    dataframe_ = self.dataframe.drop(cols_drop, axis=1).copy()
                    nrows = int(np.ceil(dataframe_.shape[1]/5))
                    fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(15, nrows*3))
                    fig.subplots_adjust(hspace=0.5)
                    plt.subplots_adjust(bottom=0.1, wspace=0.2,
                    hspace=0.9, top=1.2)

                    for ax, name in zip(axes.flatten(), dataframe_.columns):
                        sns.histplot(dataframe_[name], ax=ax)
                        ax.set_title(name)
                        ax.tick_params(axis='x', rotation=-90)
                        ax.set_xlabel(None)
                        ax.set_ylabel(None)

                    st.pyplot(fig)

    class Model():

        def __init__(self, dataframe, dataframe_test, target, problem_type, date_column):
            self.dataframe = dataframe
            self.dataframe_test = dataframe_test
            self.target = target
            self.date_column = date_column
            self.problem_class = MODELS_DICT[problem_type]['class_type']
            self.models = MODELS_DICT[problem_type]['models']

        def setup_model(self):
            with st.spinner('Wait for it...'):
                n_features_to_select = 15 if self.dataframe.shape[1] > 15 else self.dataframe.shape[1]-1
                self.problem_class.setup(
                    self.dataframe, target=self.target, fold=3, 
                    session_id=42, normalize=True, remove_outliers=True, 
                    feature_selection=True, n_features_to_select=n_features_to_select, max_encoding_ohe=2
                    )
                setup_df = self.problem_class.pull().astype(str)
                self.make_feature_importance_plot()

            col1, col2 = st.columns([1,3])
            with col1:
                st.markdown('## Data preparation')
                rows = ['Original data shape', 'Transformed data shape', 'Numeric imputation', 'Categorical imputation', 'Remove outliers', 'Normalize', 'Feature selection']
                st.write(setup_df[setup_df['Description'].isin(rows)].reset_index(drop=True))
            with col2:
                st.markdown('## Baseline algorithms')
                with st.spinner('Wait for it...'):
                    self.best = self.problem_class.compare_models(self.models)
                    best_df = self.problem_class.pull()
                    st.dataframe(best_df.reset_index(drop=True))
        
        def make_feature_importance_plot(self):
            st.markdown('## Feature importance')
            lr = self.problem_class.create_model('lr')
            path = self.problem_class.plot_model(lr, plot = 'feature_all', display_format='streamlit', save=True)
            image = Image.open(path)
            st.image(image)

        def convert_df(self, df):
            return df.to_csv(index=False).encode('utf-8')
        
        def make_predict(self):
            prediction = self.problem_class.predict_model(self.best, data = self.dataframe_test)
            try:
                prediction = prediction.drop(self.target, axis=1)
            except:
                pass
            prediction = prediction.rename(columns={'prediction_label': self.target})
            st.subheader('Prediction data')
            st.dataframe(prediction)
            file_name = 'Submission.csv'
            csv = self.convert_df(prediction)
            st.warning('The page will be restarted after downloading. Will be fixed soon.', icon='🚨')
            st.download_button(label='Download prediction file', data=csv, file_name=file_name, mime='text/csv')

    def main():
        st.sidebar.title('BASELINE')
        data = Data()
        data.upload_file()
        if data.dataframe is not None:
            button = st.sidebar.button('Get fast baseline.', key='button_0')
            if button:
                data.make_corr_plot()
                data.make_hist_plot()

                model = Model(*data.get_vars())
                model.setup_model()
                if model.dataframe_test is not None:
                    model.make_predict()

    if __name__=='__main__':
        main()


if selected == 'Classification':
    with st.sidebar:
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file: 
            st.session_state.df = pd.read_csv(file, index_col=None)
        if st.button('Load Classification example'):
            st.session_state.df = pd.read_csv('examples/Titanic.csv')
            st.write('Titanic.csv loaded')
        if st.button("Clear All"):
            st.cache_resource.clear()
    section = option_menu(None, ["Prep & Train",'Tune & Analyse','Predict'], 
    default_index=0,icons=['1-square','1-square','1-square'],orientation="horizontal")
        

    if section == 'Prep & Train':
        pretrain_clf(classif_dictionary)

    if section == 'Tune & Analyse':
        tunalyse_clf()

    if section == 'Predict':
        predict()


            
if selected == 'Regression':
    with st.sidebar:
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file: 
            st.session_state.df = pd.read_csv(file, index_col=None)
        if st.button('Load regression example'):
            st.session_state.df = pd.read_csv('examples/House_prices.csv')
            st.write('House_pricing.csv loaded')
        if st.button("Clear All"):
            st.cache_resource.clear() 
    section = option_menu(None, ["Prep & Train",'Tune & Analyse','Predict'], 
    default_index=0,icons=['1-square','1-square','1-square'],orientation="horizontal")

    if section == 'Prep & Train':
        pretrain_regr(regr_dic)

    if section == 'Tune & Analyse':
        tunalyse_regr()
              
    if section == 'Predict':
        predict_regr()






