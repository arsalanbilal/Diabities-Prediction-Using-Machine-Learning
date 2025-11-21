import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load saved model

with open('best_model.pkl', 'rb')as f:
  best_model = pickle.load(f)

# Create Streamlit app

st.title('DIABITIES PREDICTION USING MACHINE LEARNING')
st.info('This is a Logistic Regression model ')

# Sidebar header
with st.sidebar:
    st.header('Input Features')
    Pregnancies = st.slider('Pregnancies', min_value= 00.00, max_value= 13.00, value = 1.0 )
    Glucose = st.slider('Glucose', min_value=44.00, max_value=199.00, value = 44.1)
    BloodPressure = st.slider('BloodPressure', min_value=40.00, max_value=104.00, value = 40.1)
    SkinThickness = st.slider('SkinThickness', min_value=7.00, max_value=49.00, value = 7.01)
    Insulin = st.slider('Insulin', min_value=00.00, max_value=258.00, value = 1.0)
    BMI = st.slider('BMI', min_value=18.20, max_value=48.80, value = 18.21)
    DiabetesPedigreeFunction = st.slider('DiabetesPedigreeFunction', min_value=0.07, max_value=1.02, value = 0.08 )
    Age = st.slider('Age', min_value=21.00, max_value=61.00, value = 21.00)

# Make Predictions :-

if st.button("PREDICT"):
  data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
  scalar = StandardScaler()
  data = scalar.fit_transform(data)
  predictions = best_model.predict(data)
  if predictions[0] ==1:
    st.success("You are likely to have Diabities.")
  else:
    st.success("You are Unlikely to have Diabities")  




