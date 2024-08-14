# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:55:41 2024

@author: Dell
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st
#import sympy as sp

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Bankruptcy Prediction :hourglass:")
st.markdown('This project models the probability of a business going bankrupt using a Support Vector Classifier approach. It analyzes six key features, Industrial risk, Management risk, Financial flexibility, Credibility, Competitiveness and Operating risk to predict bankruptcy.')

# Load the model
load = open("bankrupty.pkl", "rb")
model = pickle.load(load)

def predict(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk):
    # Convert the input features to a DataFrame for compatibility with the pipeline
    input_data = pd.DataFrame([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]],
                              columns=['industrial_risk', 'management_risk', 'financial_flexibility', 'credibility', 'competitiveness', 'operating_risk'])

    # Predict using the trained model
    prediction = model.predict(input_data)
    y_prob = model.predict_proba(input_data)

    return prediction, y_prob

def main():
    
    with st.sidebar:
        st.title(":blue[_Group 6_]")
        st.text('1.	Vishnu Tanpure')
        st.text('2.	Meka Vamshi')        
        st.text('3.	Manasi Sardesai')
        st.text('4.	Sukrta G A')
        st.text('5.	Affan Chaus')
        
        footer_html = """<div style='text-align: left;'>
        <p style="margin-bottom:2cm;"> </p>
        <p style="color:#000080";> <b> Designed and Developed by </b> </br> <i> Vishnu Tanpure </i> </p>
        </div>"""
        st.markdown(footer_html, unsafe_allow_html=True)
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        industrial_risk = st.radio(r'${\large Industrial \ \ Risk}$', options = ['Low', 'Medium', 'High'])  
    
        management_risk = st.radio(r'${\large Management \ \ Risk}$', options = ['Low', 'Medium', 'High'])
    
    with col2:
        financial_flexibility = st.radio(r'${\large Financial \ \ Flexibility}$', options = ['Low', 'Medium', 'High'])
   
        credibility = st.radio(r'${\large Credibility}$', options = ['Low', 'Medium', 'High'])
        
    with col3:
        competitiveness = st.radio(r'${\large Competitiveness}$', options = ['Low', 'Medium', 'High'])
    
        operating_risk = st.radio(r'${\large Operating \ \ Risk}$', options = ['Low', 'Medium', 'High'])
    
    # Convert categorical inputs to numerical
    risk_map = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}
    industrial_risk = risk_map[industrial_risk]
    management_risk = risk_map[management_risk]
    financial_flexibility = risk_map[financial_flexibility]
    credibility = risk_map[credibility]
    competitiveness = risk_map[competitiveness]
    operating_risk = risk_map[operating_risk]

    #st.subheader('Prediction result')
    if st.button('Predict'):
        result, y_prob = predict(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk)
        st.write('Probability of Bankruptcy is',np.round(y_prob[0][1],4))
        
        if(result == [1]):
            st.success('Business is heading towards Bankruptcy',icon= "❌")
                            
        else:
            st.success('Business is safe, no threat of Bankruptcy', icon="✅")
                      

if __name__ == '__main__':
    main()
