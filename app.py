import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

model=tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_geo=pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo=pickle.load(file)
with open('scalar.pkl', 'rb') as file:
    scalar=pickle.load(file)

##streamlit app

st.title('Customer Churn Prediction')
st.write('This is a simple web app to predict customer churn using a neural network model')
st.write('Please fill in the details of the customer')
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_geo.classes_)
age=st.slider('Age',18,100)
tenure=st.slider('Tenure',0,10)
balance=st.number_input('Balance')
num_products=st.slider('Number of Products',1,4)
credit_score=st.number_input('Credit Score')
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])
estimated_salary=st.number_input('Estimated Salary')

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_geo.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df.reset_index(drop=True)],axis=1)
input_data_scaled=scalar.transform(input_data)

prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write(f'The probability of the customer churning is:{prediction_prob:.2f}')

if prediction_prob>0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')
