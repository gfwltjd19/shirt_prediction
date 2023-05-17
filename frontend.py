import joblib
import pandas as pd
import streamlit as st

st.title("Shirt Size Prediction")

#Interface table for input
age = st.number_input("Enter your age:", 1, 117)
weight = st.number_input("Enter your weight:", 1, 136)
height = st.number_input("Enter your height:", 1, 193)

#save input from user into dataframe
user_input = pd.DataFrame([[weight, age, height]], columns=["weight", "age", "height"])

#import model file named clf.pkl 
model = joblib.load('clf.pkl')

#create variable named prediction to predict model from user input data
prediction = model.predict(user_input)

#create button, prediction[0] must zero 
if st.button("Predict"):
    st.write("Your size is", prediction[0])

pd.show_versions()