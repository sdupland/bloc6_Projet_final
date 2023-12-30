import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import yfinance
# from functions import calendar_features, technic_features

# Config
st.set_page_config(
    page_title="Bitcoin dashboard",
    layout="wide"
)

# App
st.title('Conclusions')

# Content
st.sidebar.markdown("")
st.sidebar.markdown("")

# Loading Text
st.markdown("")
st.markdown("""   
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
""")


# Footer
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("""
        [my Github](www.github.com/sdupland)
    """)
