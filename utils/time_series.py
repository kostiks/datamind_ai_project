from pycaret.time_series import *
import streamlit as st


def train_ts(df_ts, fh, folds, target):
    ts = setup(data=df_ts, fh=fh, fold=folds, target=target)
    best = compare_models()
    best_ts = pull()
    return best, best_ts