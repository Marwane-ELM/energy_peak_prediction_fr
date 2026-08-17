import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests

#st_autorefresh(interval=1000, limit = 10)
# IN the streamlit app I'll 
response = requests.get("http://127.0.0.1:8000/predict")
status_code = response.status_code

assert status_code == 200, st.title(status_code)

data = response.json()
    
st.dataframe(data)
