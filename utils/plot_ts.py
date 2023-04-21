from pycaret.time_series import *
import streamlit as st


def plot_graph_ts(model=None, type='ts'):
    plot_model(model, plot = type, display_format='streamlit')