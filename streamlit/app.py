import streamlit as st
import requests

# IN the streamlit app I'll 
response = requests.get("http://127.0.0.1:8000/predict")
status_code = response.status_code

assert status_code == 200, st.title(status_code)

data = response.json()
    
st.dataframe(data)
