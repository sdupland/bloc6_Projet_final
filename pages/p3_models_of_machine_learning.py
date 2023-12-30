import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Config
st.set_page_config(
    page_title="Bitcoin dashboard",
    layout="wide"
)

# App
st.title('Bitcoin modelisation')
dataset_btc = st.session_state["dataset_btc"] 

@st.cache_data()
def load_data():
    path = "btc_completed_optimized.csv"
    data = pd.read_csv(path, index_col=0)
    return data

dataset_btc = load_data()

# Content
st.sidebar.header("Table of content")
st.sidebar.markdown("""
    * [Plot 1](#plot-1) - Evolution of bitcoin price
    * [Plot 2](#plot-2) - Distribution xxxxxxxxxxxxxxx
    * [Plot 3](#plot-3) - Distribution xxxxxxxxxxxxxxxxxxxxxx
    * [Plot 4](#plot-4) - A different xxxxxxxxxxxxxxxxx
""")

model_lr = joblib.load("model_lr.joblib")
model_elasticnet = joblib.load("model_elasticnet.joblib")
model_xgb = joblib.load("model_xgb.joblib")

# Plot 1
st.markdown("---")
st.subheader('Plot 1')
st.markdown("### Comparison between original Close price vs predicted close price")
fig1 = px.line(dataset_btc, x=dataset_btc.index, y="Close", height=800, labels={"Close":"Close price in Eur"})
fig1.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])))
st.plotly_chart(fig1, use_container_width=True)
st.markdown("""
            Comment : 
            """)

# Footer
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("""
        [my Github](www.github.com/sdupland)
    """)
