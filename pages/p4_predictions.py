import streamlit as st
import requests
import pandas as pd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Config
st.set_page_config(
    page_title="Bitcoin dashboard",
    layout="wide"
)

# App
st.title('Bitcoin prediction')
st.markdown("")
st.markdown("")
st.markdown("### Please choose a model on the sidebar and press the launch button to actualize values and trend for tomorrow !")

@st.cache_data()
def load_data():
    path = "btc_completed_optimized.csv"
    data = pd.read_csv(path, index_col=0)
    return data

# Content
st.sidebar.header("Options")
st.sidebar.markdown("---")
st.sidebar.markdown("")
option_model = st.sidebar.selectbox("Which models do you want to use ?", ("Linear regression", "Elastic net","XGBoost"), index=None, placeholder="Choose an option")
st.sidebar.markdown("")
st.sidebar.markdown("---")
st.sidebar.markdown("")
#option_horizon = st.sidebar.selectbox("Which time horizon do you want ?", ("tomorrow morning","J+1", "J+2"), index=None, placeholder="Choose an option")
#st.sidebar.markdown("---")
st.sidebar.markdown("")
state_button = st.sidebar.button("Launch prediction")

st.sidebar.markdown("")
st.sidebar.markdown("")

if state_button :
    # dictionary with all the charasteristics of our car and a set of values we want
    payload = {"Model": option_model}
    # request used when api is deployed on heroku
    r = requests.post("https://bitcoin-api-1f113f1280ef.herokuapp.com/predict", json=payload).json()
    last_price = r["last_price"]
    futur_price = r["futur_price"]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("")
    st.markdown("")
    st.markdown("")  
    st.markdown("### The last price known is :")
    st.markdown("")
    st.markdown("")  
    if state_button :
        st.markdown("### {} euros".format(last_price))
    else :
        st.markdown("waiting for launching prediction")
    st.markdown("")
    
with col2 :
    st.markdown("")
    st.markdown("")  
    st.markdown("")  
    st.markdown("### The prediction will be for tomorrow :") 
    st.markdown("")
    st.markdown("")  
    if state_button :
        st.markdown("### {} euros".format(futur_price))
    else :
        st.markdown("waiting for launching prediction")
    st.markdown("")
    st.markdown("")
    st.markdown("")

with col3:
    st.markdown("")
    st.markdown("")
    st.markdown("")  
    if state_button :
        fig1 = go.Figure(go.Indicator(
                    mode = "delta",
                    value = futur_price,
                    number = {"prefix":"€", "valueformat":".2d"},
                    delta={"reference":last_price,"relative":True,"valueformat":".2%"}
                    ))
        fig1.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=150)
        st.plotly_chart(fig1, use_container_width=True)
    else :
        st.markdown("### See the trend for tomorrow")
        st.markdown("")
        st.markdown("")  
        st.markdown("waiting for launching prediction")
    st.markdown("")

# Footer
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("""
        [my Github](www.github.com/sdupland)
    """)