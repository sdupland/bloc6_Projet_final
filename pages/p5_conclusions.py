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
    Mitigants :
    - Number of data is limited, and first years could be not really representative of the current period.
    We can try to reduce the frequency from a day to an hour or less in order get data from current period of time.
    - With supervised machine learning model, the value of an indicator doesn’t tell us anything about is evolutionFor some indicators.
    we can try to add features that give information about their trend.

    Next steps :
    - Complete the dashboard (dictionary of indicators)
    - Add features such as other technical indicators, other financial indicators (price of other crypto, or stock), behavioral indicator (sentimental score for  example from tweet or articles), macro-economics indicators (inflation rate for example)
    - Play with parameters of each technical indicators (often period of times) or with data frequency (hour, minutes)
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
