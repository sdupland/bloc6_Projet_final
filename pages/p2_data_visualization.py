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

col1, col2 = st.columns(2)

with col1:    
    # Plot 1
    st.markdown("---")
    st.subheader('Plot 1')
    st.markdown("### Trend over the last week")
    last_price = dataset_btc.iloc[-1,3]
    last_week_price = dataset_btc.iloc[-7,3]
    fig1 = go.Figure(go.Indicator(
                    mode = "number+delta",
                    value = last_price,
                    number = {"prefix":"€", "valueformat":".2d"},
                    delta={"reference":last_week_price,"relative":True,"valueformat":".2%"}
                    ))
    st.plotly_chart(fig1, use_container_width=True)

with col2:    
    # Plot 2
    st.markdown("---")
    st.subheader('Plot 2')
    st.markdown("### Trend since yesterday")
    last_price = dataset_btc.iloc[-1,3]
    yesterday_price = dataset_btc.iloc[-2,3]
    fig2 = go.Figure(go.Indicator(
                    mode = "number+delta",
                    value = last_price,
                    number = {"prefix":"€", "valueformat":".2d"},
                    delta={"reference":yesterday_price,"relative":True,"valueformat":".2%"}
                    ))
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
st.markdown("### Visualisation of a technical indicator of your choice over the last 6 months")
ta_name = st.selectbox("Which indicator do you want to see ?", (dataset_btc.iloc[-180:,5:-12].columns))
fig4 = px.line(dataset_btc.iloc[-180:,:], x=dataset_btc[-180:].index, y=ta_name, labels={"Close":"Close price in Eur"})
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

# Plot 5
st.markdown("---")
st.subheader('Plot 5')
st.markdown("### Visualisation of a technical indicator of your choice over the last 6 months")
ta_dictionnary = {"Relative strenght index (RSI)" : ["rsi_7","rsi_14","rsi_28"], "Bollinger bands" : ["upper_band20","lower_band20"]}
ta_name = st.selectbox("Which indicator do you want to see ?", (ta_dictionnary.keys()))                      
fig5 = px.line(dataset_btc.loc[dataset_btc.index > "2023-10-31",ta_dictionnary[ta_name]], 
               x=dataset_btc.loc[dataset_btc.index > "2023-10-31",ta_dictionnary[ta_name]].index,
               y=ta_name, labels={"Close":"Close price in Eur"})
fig5.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ])))
st.plotly_chart(fig5, use_container_width=True)

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
