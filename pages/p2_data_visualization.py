import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Config
st.set_page_config(
    page_title="Bitcoin dashboard",
    layout="wide"
)

# App
st.title('Data visualization')

# Loading Text
st.markdown("")
st.markdown("""   
    This page simply gives you a look at how the bictoin has evolved since a few years, and what is the actual trend (day and week).
    
    In addition, you can select a specific technical indicator in order to see its evolution.
""")

dataset_btc = st.session_state["dataset_btc"] 

# Content
st.sidebar.header("Table of content")
st.sidebar.markdown("""
    * [Plot 1 - Evolution trend of bitcoin price since last week](#plot-1)
    * [Plot 2 - Evolution trend of bitcoin price since yesterday](#plot-2)
    * [Plot 3 - Evolution of bitcoin price over the period](#plot-3) 
    * [Plot 4 - Evolution of a specific indicator over the period](#plot-4)
""")

col1, col2, col3, col4 = st.columns(4)

with col1:    
    # Plot 1
    st.markdown("---")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.subheader('Plot 1')
    st.markdown("### Trend over the last week")

with col2 :
    st.markdown("---")
    last_price = dataset_btc.iloc[-1,3]
    last_week_price = dataset_btc.iloc[-7,3]
    fig1 = go.Figure(go.Indicator(
                    mode = "delta",
                    value = last_price,
                    #number = {"prefix":"€", "valueformat":".2d"},
                    delta={"reference":last_week_price,"relative":True,"valueformat":".2%"}
                    ))
    fig1.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=200)
    st.plotly_chart(fig1, use_container_width=True)

with col3:    
    # Plot 2
    st.markdown("---")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.subheader('Plot 2')
    st.markdown("### Trend since yesterday")

with col4 :
    st.markdown("---")
    last_price = dataset_btc.iloc[-1,3]
    yesterday_price = dataset_btc.iloc[-2,3]
    fig2 = go.Figure(go.Indicator(
                    mode = "delta",
                    value = last_price,
                    #number = {"prefix":"€", "valueformat":".2d"},
                    delta={"reference":yesterday_price,"relative":True,"valueformat":".2%"}
                    ))
    fig2.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=200)
    st.plotly_chart(fig2, use_container_width=True)

# Plot 3
st.markdown("---")
st.subheader('Plot 3')
st.markdown("### Evolution of bitcoin prices since 2019")
fig3 = px.line(dataset_btc, x=dataset_btc.index, y="Close", height=800, labels={"Close":"Close price in Eur"})
fig3.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])))
st.plotly_chart(fig3, use_container_width=True)

# Plot 4
st.markdown("---")
st.subheader('Plot 4')
st.markdown("### Visualisation of a technical indicator of your choice since first of august 2023")
ta_dictionnary = {"Relative strenght index (RSI)" : ["rsi_7","rsi_14","rsi_28"],
                  "Bollinger bands" : ["upper_band20","lower_band20"],
                  "Ichimoku Cloud" : ['Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B','Chikou_Span'],
                  "Exponential Moving Average (EMA)" :["ema_3","ema_15","ema_50", "ema_100"],
                  "Kurtosis" : ["kurtosis_5", "kurtosis_10","kurtosis_20"],
                  "Williams %R" : ["Williams_%R14"],
                  "Exponential moving average (EMA)" : ["ema_3","ema_8","ema_15","ema_50","ema_100"],
                  "Average Directional indeX (ADX)" : ["ADX_14"],
                  "Donchian Channel" : ["Donchian_Upper_10", "Donchian_Lower_10", "Donchian_Upper_20", "Donchian_Lower_20"],
                  "Arnaud Legoux Moving Average (ALMA)" : ["ALMA_10"],
                  "True Strength Index (TSI)" : ["TSI_13_25"],
                  "Z Score" : ["Z_score_20"],
                  "Log return" : ["LogReturn_10","LogReturn_20"],
                  "Vortex Indicator" : ["Positive_VI_7", "Negative_VI_7"],
                  "Aroon Indicator" : ["Aroon_Up_16","Aroon_Down_16"],
                  "Elder's Bull Power and Bear Power" : ["Bull_Power_14", "Bear_Power_14"],
                  "Acceleration Bands" : ["Upper_Band_20","Lower_Band_20","Middle_Band_20"],
                  "Short Run" : ["Short_Run_14"],
                  "Bias" : ["Bias_26"],
                  "TTM Trend" : ["TTM_Trend_5_20"],
                  "Percent return" : ["Percent_Return_1", "Percent_Return_5","Percent_Return_10","Percent_Return_20"],
                  "Standard deviation" : ["Stdev_5","Stdev_10","Stdev_20"]
                  }
ta_name = st.selectbox("Which indicator do you want to see ?", (ta_dictionnary.keys()))
columns_name = ta_dictionnary[ta_name]                    
fig4 = px.line(dataset_btc.loc[dataset_btc.index > "2023-07-31", columns_name],
               x=dataset_btc.loc[dataset_btc.index > "2023-07-31"].index,
               y=columns_name, labels={"Close":"Close price in Eur"})
fig4.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ])))
st.plotly_chart(fig4, use_container_width=True)

# Footer
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.markdown("")
    st.markdown("---")
    st.markdown("")
    st.write("")

with footer:
    st.markdown("")
    st.markdown("---")
    st.markdown("")
    st.markdown("""
        [my Github](https://github.com/sdupland)
    """)
