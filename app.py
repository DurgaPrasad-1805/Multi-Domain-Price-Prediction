import streamlit as st

st.set_page_config(
    page_title="Multi-Domain Price Prediction",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Multi-Domain Price Prediction Dashboard")

st.markdown("""
Welcome to the Multi-Domain Machine Learning System.

This project includes:

- House Price Prediction  
- Car Price Prediction  
- Gold Price Forecasting  
- Unified Model Comparison  

Use the sidebar to explore each domain.
""")