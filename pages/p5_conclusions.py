import streamlit as st
import pandas as pd

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
    Performance of our models aren't so bad for a first step with for the best one an error for around 1,3%.
    The work I did in order to improve performances leads us to reduce this score from 75% which is very significant.
    
    Nevertheless, we have detected two main mitigants :
    
    - the number of data is limited, and first years could be not really representative of the current period.
    To adress it, we can try to reduce the frequency from a day to an hour or less in order to get data from current period of time.
    
    - with supervised machine learning model, the value of an indicator doesn’t tell us anything about is evolution.
    To adress it, for some indicators, we can try to add features that give informations about their trend.

    In addition, we could also :
    
    - add features such as other technical indicators, other financial indicators (price of other crypto, or stock), behavioral indicators (sentimental score for  example from tweet or articles), macro-economics indicators (inflation rate for example)
    - play with parameters of each technical indicators (often period of times) or with data frequency (hour, minutes)
    - try others models
""")

# Footer
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("")
    st.markdown("")
    st.markdown("""
        [my Github](www.github.com/sdupland)
    """)
