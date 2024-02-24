import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
@st.cache_data()
def load_model():
    return joblib.load('rf_regressor.joblib')

model = load_model()

# Sidebar for user input
st.sidebar.header('User Input Parameters')
tv = st.sidebar.slider('TV Advertising', 0.0, 500.0, 150.0)
radio = st.sidebar.slider('Radio Advertising', 0.0, 50.0, 25.0)
newspaper = st.sidebar.slider('Newspaper Advertising', 0.0, 100.0, 50.0)

# Make predictions
input_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
prediction = model.predict(input_data)

# Display prediction
st.header('Sales Prediction')
st.write(f"Predicted Sales: {prediction[0]}")