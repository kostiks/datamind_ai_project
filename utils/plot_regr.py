from pycaret.regression import *
import streamlit as st

def plot_graph_regr(model,types):
    for i in types:
        plot_model(model, plot = i, display_format='streamlit')