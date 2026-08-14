import streamlit as st
import requests

response = requests.get("http://127.0.0.1:8000/predict")
status_code = response.status_code

assert status_code == 200, st.title(status_code)

dates, preds = response.json()

#for i, j in zip(dates, preds):
    
st.title(dates)
st.title(preds)
