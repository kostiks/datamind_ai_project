from pycaret.classification import *
import streamlit as st

def plot_graph_clf(model,types):
    for i in types:
        plot_model(model, plot = i, display_format='streamlit')