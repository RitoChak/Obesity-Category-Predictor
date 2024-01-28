import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer

st.title('Obesity Category Predictor')

def user_input_features():
    col1, col2 = st.columns(2)
    with col1:
        Height = st.number_input('Height(in cms)')
        Gender = st.selectbox('Gender',('Male','Female'))
        Age = st.number_input('Age', min_value =0, step =1) 
    with col2:
        Weight = st.number_input('Weight(in kgs)')
        PhysicalActivityLevel = st.selectbox('Physical Activity Level', (1,2,3,4))
        
    # Check for zero height to avoid division by zero
    if Height != 0:
        BMI = Weight / ((Height / 100) ** 2)
    else:
        BMI = np.nan
    
    data = {
        'Age': Age,
        'Gender': Gender,
        'Height': Height,
        'Weight': Weight,
        'BMI' : BMI,
        'PhysicalActivityLevel': PhysicalActivityLevel
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button('Predict'):
    data = pd.read_csv('Cleaned.csv')
    data = pd.concat([input_df, data], axis=0)
    
    numerical_features = ['Age', 'Height', 'Weight', 'BMI']
    categorical_features = ['Gender', 'PhysicalActivityLevel']
    
    preprocessor = ColumnTransformer([
    ('Oh', OneHotEncoder(sparse_output = False, drop = 'first'), categorical_features),
    ('Scaler', StandardScaler(), numerical_features)
]).set_output(transform = 'pandas')
    
    data_prep = preprocessor.fit_transform(data)
    data = data_prep[:1]
    
    file = open('Obesity_Model.pkl', 'rb')
    model = pickle.load(file)
    
    prediction = model.predict(data)
    
    st.title('Prediction')
    st.markdown(f'<p style="font-size : 25px"> Predicted Obesity Category is : <strong>{prediction[0][0]}</strong></p>', unsafe_allow_html=True)
    
