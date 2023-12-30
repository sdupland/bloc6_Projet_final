import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Config
st.set_page_config(
    page_title="Bitcoin dashboard",
    layout="wide"
)

# App
st.title('Bitcoin analysis and prediction')

@st.cache_data()
def load_data():
    path = "btc_completed.csv"
    data = pd.read_csv(path, index_col=0)
    return data

dataset_btc = load_data()
st.session_state["dataset_btc"] = dataset_btc

# Loading Text
st.markdown("")
st.markdown("""   
    The goal of this dashboard has many dimensions.
    
    First to give some insights on the bitcoin based on historical data. To do that, you will find on some previews on data used in this dashboard, split into different parts :
    - raw data from yfinance (bitcoin prices and volume)
    - a set of technical indicators calculated from the raw data, and grouped in a few categories according to the main existing families (trend, volume, oscillator ..)
    
    Secondly to facilitate the comprehension of Bitcoin evolution ovee the period through some graphs about bitcoin prices and technical indicators.
    
    Thirdly to present the work done on data modelisation through the use of machine learning models in order to predict the price of Bitcoin.
    You will find in p3 a short presentation of different models, and the results we got.
    
    Fourth to predict the price of the bitcoin regarding different time horizons, and models.
    
    At least, we made a synthesis of this work which contains :
    - some limitations and reflections about the job done
    - various ways to improve it
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
        [my Github](https://github.com/sdupland)
    """)
