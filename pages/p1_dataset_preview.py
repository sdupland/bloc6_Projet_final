import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import itertools

# Config
st.set_page_config(
    page_title="Bitcoin dashboard",
    layout="wide"
)

# App
st.title('Dataset preview')
st.markdown("")
st.markdown("""
    Here are some previews of different parts of the global dataset we used :
    - raw data of bitcoin prices and volume, from yahoo finance
    - example of technical indicators grouped by main families :
        - trend
        - momentum
        - volatility
        - oscillator
        - volume
    
    Sometimes indicators can be adressed in different families.
    For simplification purpose, we made a choice in order to avoid to present it several times
""")

dataset_btc = st.session_state["dataset_btc"]

# Content
st.sidebar.header("Table of content")
st.sidebar.markdown("""
    * [Plot 1 - Overview of raw data](#plot-1)
    * [Plot 2 - Overview of momentum indicators](#plot-2)
    * [Plot 3 - Overview of trend indicators](#plot-3)
    * [Plot 4 - Overview of volume indicators](#plot-4)
    * [Plot 5 - Overview of volatility indicators](#plot-5)
    * [Plot 6 - Overview of oscillator indicators](#plot-6)
""")

# Plot 1
st.markdown("---")
st.subheader('Plot 1')
st.subheader('Overview of bitcoin prices and volume')
st.write(dataset_btc.iloc[-10:,0:5])
st.markdown("""
    comment
""")

# Plot 2
st.markdown("---")
st.subheader('Plot 2')
st.subheader('Overview of momentum technical indicators')
st.markdown("""
    Momentum indicators are used to identify the strength or weakness of a trend based on the speed of price changes.
""")
st.write(dataset_btc.iloc[-10:,5:16])

# Plot 3
st.markdown("---")
st.subheader('Plot 3')
st.subheader('Overview of trend technical indicators')
st.markdown("""
    Trend indicators help traders identify the direction of the price movement and whether an asset is in an uptrend, downtrend, or ranging.
""")
st.write(dataset_btc.iloc[-10:,16:25])

# Plot 4
st.markdown("---")
st.subheader('Plot 4')
st.subheader('Overview of volume technical indicators')
st.markdown("""
    Volume indicators are essential for analyzing trading activity and market sentiment.
""")
st.write(dataset_btc.iloc[-10:,65:-4])

# Plot 5
st.markdown("---")
st.subheader('Plot 5')
st.subheader('Overview of volatility technical indicators')
st.markdown("""
    Volatility indicators are crucial for assessing the price fluctuations and risk associated with an asset.
""")
st.write(dataset_btc.iloc[-10:,58:-16])

# Plot 6
st.markdown("---")
st.subheader('Plot 6')
st.subheader('Overview of oscillator technical indicators')
st.markdown("""
    Oscillator indicators are designed to identify potential overbought or oversold conditions in the market and help traders gauge the momentum of price movements.
""")
st.write(dataset_btc.iloc[-10:,50:65])

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
