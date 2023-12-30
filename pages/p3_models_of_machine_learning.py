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


@st.cache_data()
def load_data():
    data = pd.read_csv("predictions.csv", index_col=0)
    data1 = pd.read_csv("best_results.csv", index_col=0)
    return data, data1

predictions, best_results = load_data()

# Content
st.sidebar.header("Table of content")
st.sidebar.markdown("""
    * [Plot 1](#plot-1) - Performance on test set
    * [Plot 2](#plot-2) - Comparison of original prices and predictions
""")

st.markdown("")
st.markdown("#### The dataset used to test our models has {} rows".format(len(predictions)))
st.markdown("")
st.markdown("#### It goes from {} to {}".format(predictions.index[-1], predictions.index[0]))

# Plot 1
st.markdown("---")
st.subheader('Plot 1')
st.markdown("### Performance on test set")
st.write(best_results)
st.markdown("""
            Comment : 
            """)

# Plot 2
st.markdown("---")
st.subheader('Plot 2')
st.markdown("### Comparison between original Close price vs predicted close price")
fig2 = px.line(predictions, x=predictions.index, y="Original price", height=800, labels={"Original price":"Close price in Eur"})
fig2.add_scatter(x=predictions.index, y=predictions["Elasticnet with fine tuning"], mode='lines')
fig2.add_scatter(x=predictions.index, y=predictions["XGBoost with RFE"], mode='lines')
fig2.add_scatter(x=predictions.index, y=predictions["Linear regression with RFE"], mode='lines')
st.plotly_chart(fig2, use_container_width=True)
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
