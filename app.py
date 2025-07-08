import pandas as pd
import numpy as np 
import joblib
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

import streamlit as st

# Suppress scikit-learn InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

model = joblib.load('pollutant_model.pkl')
model_cols = joblib.load('model_columns .pk1') 

st.title('Water Pollution Prediction')
st.write('Prediction of the water pollutant based on the year and station id')

year_input = st.number_input('Enter the year', min_value=2000, max_value=2100, value=2022)
station_id_input = st.text_input('Enter the station id', value='1')

input_dict = {'year': [year_input], 'station_id': [station_id_input]}
input_df = pd.DataFrame(input_dict)

input_df = input_df.reindex(columns=model_cols, fill_value=0)

# Remove the immediate prediction and display, instead wait for button click
# st.write(f'Predicted water pollutant level: {prediction}')

if st.button('Predict'):
    if not station_id_input:
        st.warning('Please enter a station id')
    else:
        input_dict = {'year': [year_input], 'station_id': [station_id_input]}
        input_df = pd.DataFrame(input_dict)
        input_df = input_df.reindex(columns=model_cols, fill_value=0)

        predicted_pollutant = model.predict(input_df)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.subheader(f"Predicted Pollutant Levels for the station '{station_id_input}' in {year_input}:")
        for p, val in zip(pollutants, predicted_pollutant):
            st.write(f'{p}: {val:.2f}')
