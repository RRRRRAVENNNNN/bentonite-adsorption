import joblib
import xgboost as xgb
import numpy as np
import streamlit as st

import streamlit as st

st.image('JLU.png')


st.header('ML prediction for heavy metal adsorption capacity of bentonite')

model = joblib.load('ML model.joblib')
ss = joblib.load('StandardScaler.joblib')

col1, col2, col3 = st.columns(3)

with col1:
    col1.subheader(':red[Bentonite properties]')
    feature1 = st.number_input(u'$\mathrm{CEC}$', step=0.01, format='%.2f')
    feature2 = st.number_input(u'$\mathrm{specific\;surface\;area}$', step=0.01, format='%.2f')

with col2:
    col2.subheader(':blue[Adsorption conditions]')
    feature3 = st.number_input(u'$\mathrm{dosage}$', step=0.01, format='%.2f')
    feature4 = st.number_input(u'$\mathrm{pH}$', step=0.01, format='%.2f')
    feature5 = st.number_input(u'$\mathrm{temperature}$', step=0.01, format='%.2f')
    feature6 = st.number_input(u'$\mathrm{initial\;concentration}$', step=0.01, format='%.2f')
    feature7 = st.number_input(u'$\mathrm{contact\;time}$', step=0.01, format='%.2f')

with col3:
    col3.subheader(':orange[Heavy metal properties]')
    feature8 = st.number_input(u'$\mathrm{electronegativity}$', step=0.01, format='%.2f')
    feature9 = st.number_input(u'$\mathrm{radius\;of\;hydrated\;heavy\;metal\;ions}$', step=0.01, format='%.2f')
    
feature_values = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]

if st.button('Predict', type='primary'):
    input_data = np.array([feature_values])
    input_data_scaled = ss.transform(input_data)
    prediction = model.predict(input_data_scaled)
    st.success(f'Predicted heavy metal adsorption capacity: {prediction[0]:.2f} mg/g', icon="✅")
