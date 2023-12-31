import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from functions import calculate_all_indicators_optimised
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE

# Config
st.set_page_config(
    page_title="Bitcoin dashboard",
    layout="wide"
)

# App
st.title('Bitcoin prediction')
st.markdown("")
st.markdown("")
st.markdown("### Please press the launch button to actualize values and trend for tomorrow !")

@st.cache_data()
def load_data():
    path = "btc_completed_optimized.csv"
    data = pd.read_csv(path, index_col=0)
    return data

dataset_btc = load_data()

# Content
st.sidebar.header("Options")
st.sidebar.markdown("---")
st.sidebar.markdown("")
option_model = st.sidebar.selectbox("Which models do you want to use ?", ("Linear Regression", "Elastic net","XGBoost"), index=None, placeholder="Choose an option")
st.sidebar.markdown("")
st.sidebar.markdown("---")
st.sidebar.markdown("")
#option_horizon = st.sidebar.selectbox("Which time horizon do you want ?", ("tomorrow morning","J+1", "J+2"), index=None, placeholder="Choose an option")
#st.sidebar.markdown("---")
st.sidebar.markdown("")
state_button = st.sidebar.button("Launch prediction")

st.sidebar.markdown("")
st.sidebar.markdown("")

if option_model =="Linear Regression" :
    path = "model_rfe_lr.joblib"
    model = joblib.load(path)
elif option_model == "Elastic net" :
    path = "model_elasticnet.joblib"
    model = joblib.load(path)
else :
    path = "model_rfe_xgb.joblib"
    model = joblib.load(path)  

def price_prediction(model_name, dataset) :
    """
    Returns:
        price of bitcoin for tomorrow :
    """
    scaler = joblib.load("scaler.bin")
    btc_ticker = yf.Ticker("BTC-EUR")
    btc_actual = btc_ticker.history(period="200d", actions=False)
    btc_actual = btc_actual.tz_localize(None)
    btc_actual = calculate_all_indicators_optimised(btc_actual)
    btc_actual = btc_actual.sort_index(ascending=False)
    input_data = btc_actual.iloc[0,:]
    input_data = input_data.fillna(0)
    input_data = input_data.array
    input_data = input_data.reshape(1,-1)
    input_data = scaler.transform(input_data)
    futur_price = model_name.predict(input_data)
    last_price = btc_actual.iloc[0,3]
    return futur_price, last_price

if state_button :
    futur_price, last_price = price_prediction(model,dataset_btc)

col1, col2 = st.columns(2)

with col1:
    # Plot 1
    st.markdown("---")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("### The last price known is :")
    st.markdown("")
    if state_button :
        st.markdown("### {} euros".format(round(int(last_price))))
    else :
        st.markdown("to be updated")
    st.markdown("")
    st.markdown("### The prediction will be for tomorrow :") 
    st.markdown("")
    if state_button :
        st.markdown("### {} euros".format(int(futur_price[0])))
    else :
        st.markdown("to be estimated")
    st.markdown("")
    st.markdown("")
    st.markdown("")

with col2:
    # Plot 2
    st.markdown("---")
    st.markdown("")
    st.markdown("")
    if state_button :
        futur_price = int(futur_price[0])
        fig1 = go.Figure(go.Indicator(
                    mode = "number+delta",
                    value = futur_price,
                    number = {"prefix":"€", "valueformat":".2d"},
                    delta={"reference":int(last_price),"relative":True,"valueformat":".2%"}
                    ))
        st.plotly_chart(fig1, use_container_width=True)
    else :
        st.markdown("")
        st.markdown("")
        st.markdown("### See the trend for tomorrow")
        st.markdown("")
        st.markdown("To be estimated")
    st.markdown("")

# Footer
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("""
        [my Github](www.github.com/sdupland)
    """)
