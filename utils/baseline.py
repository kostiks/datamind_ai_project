import streamlit as st
# st.set_page_config(layout="wide")

import pandas as pd
import numpy as np

from PIL import Image

from pycaret.regression import RegressionExperiment
from pycaret.classification import ClassificationExperiment

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def auto_ml():
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

                    for ax, name in zip(axes.flatten(), dataframe_.columns):
                        sns.histplot(dataframe_[name], ax=ax)
                        ax.set_title(name)
                        ax.tick_params(axis='x', rotation=-45)
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
                    feature_selection=True, n_features_to_select=n_features_to_select
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
            prediction = prediction.drop('prediction_score', axis=1).rename(columns={'prediction_label': self.target})
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