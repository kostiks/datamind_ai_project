from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit as st

@st.cache_resource
def get_profile(data, config):
    profile_df = ProfileReport(data, config_file=config)
    return profile_df 