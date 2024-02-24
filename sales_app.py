import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
@st.cache_data
def load_model():
    return joblib.load('rf_regressor.joblib')

model = load_model()

# Set background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFB6C1; 
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome message
st.title('Welcome to the Sales Prediction Platform!')
st.write("This platform is designed to help you predict sales based on your advertising expenditure.")

# Introduction to advertising channels and costs
st.header('Advertising Channels and Costs')
st.write("Before we make predictions, let's introduce you to the advertising channels ")
st.write("- TV Advertisment")
st.write("- Radio Advertisment")
st.write("- Newspaper Advertisment")
st.write("Please adjust the sliders in the sidebar to set your advertising expenditure.")

# Sidebar for user input
st.sidebar.header('Advertising Channels Parameters')
tv = st.sidebar.slider('TV Advertising', 0.0, 500.0, 150.0)
radio = st.sidebar.slider('Radio Advertising', 0.0, 50.0, 25.0)
newspaper = st.sidebar.slider('Newspaper Advertising', 0.0, 100.0, 50.0)

# Make predictions
input_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
prediction = model.predict(input_data)

# Display prediction
st.header('Sales Prediction')
st.success(f"Predicted Sales: {prediction[0]}")

