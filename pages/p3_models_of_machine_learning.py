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
st.title('Bitcoin modelisation')


@st.cache_data()
def load_data():
    data = pd.read_csv("predictions.csv", index_col=0)
    data1 = pd.read_csv("best_results.csv", index_col=0)
    data2 = pd.read_csv("models_comparison.csv", index_col=0)
    return data, data1, data2

predictions, best_results, models_comparison = load_data()

# Content
st.sidebar.header("Table of content")
st.sidebar.markdown("""
    * [Plot 1](#plot-1) - Performance on train and validation set
    * [Plot 2](#plot-2) - Performance on test set
    * [Plot 3](#plot-3) - Comparison of original prices and predictions
""")

col1, col2 = st.columns(2)

with col1: 
    # Plot 1
    st.markdown("---")
    st.subheader('Plot 1')
    st.markdown("### Performance of all the models on train and validation set")
    st.markdown("")
    st.markdown("##### The dataset used to train and validate our models has 1150 rows") # to adjust
    st.markdown("##### It goes from 2019-05-01 to 2022-10-01") # to adjust
    st.markdown("")
    st.write(models_comparison)
    st.markdown("""
            """)

with col2 :
    # Plot 2
    st.markdown("---")
    st.subheader('Plot 2')
    st.markdown("### Performance on test set")
    st.markdown("")
    st.markdown("##### The dataset used to test our models has {} rows".format(len(predictions)))
    st.markdown("##### It goes from {} to {}".format(predictions.index[-1], predictions.index[0]))
    st.markdown("")
    st.write(best_results)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("""
            """)
    
    
# Plot 3
st.markdown("---")
st.subheader('Plot 3')
st.markdown("### Comparison between original Close price vs predicted close price on test set")
fig1 = px.line(predictions, x=predictions.index, y="Original price", height=800, labels={"Original price":"Close price in Eur"})
fig1.add_scatter(x=predictions.index, y=predictions["Elasticnet with fine tuning"], mode="lines", line={"dash" :"dot"}, name="Elasticnet with fine tuning")
fig1.add_scatter(x=predictions.index, y=predictions["XGBoost with RFE"], mode="lines", line={"dash" :"dash"}, name="XGBoost with RFE")
fig1.add_scatter(x=predictions.index, y=predictions["Linear regression with RFE"], mode="lines", line={"dash" :"dashdot"}, name="Linear regression with RFE")
st.plotly_chart(fig1, use_container_width=True)

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
