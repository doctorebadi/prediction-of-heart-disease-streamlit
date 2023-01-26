import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split # to split the dataset for training and testing 
from sklearn.metrics import accuracy_score

st.write("""
# Simple Prediction of heart disease""")


st.sidebar.header('User Input Parameters')


def user_input_features():
    age = st.sidebar.slider('age', 1, 79, 50)
    Height = st.sidebar.slider('Height', 10, 105, 80)
    Weight = st.sidebar.slider('Weight', 69, 170, 85)
    QRS_duration = st.sidebar.slider('QRS_duration',0, 294, 20)
    P_R_interval = st.sidebar.slider('P_R_interval', 241, 450, 260)
    Q_T_interval = st.sidebar.slider('Q_T_interval', 0, 294, 158)
    T_interval = st.sidebar.slider('T_interval', 108, 381, 270)
    P_interval = st.sidebar.slider('P_interval', 0, 183, 25)
    QRS = st.sidebar.slider('QRS',-137, 169, -50)
    T = st.sidebar.slider('T',-165, 175, 110)
    P = st.sidebar.slider('P',-17, 154, 15)
    QRST = st.sidebar.slider('QRST', -135, 139, 76)
    J = st.sidebar.slider('J', 50, 104, 94)
    Heart_rate = st.sidebar.slider('Heart_rate', 0, 88, 55)
    data = {'age': age,'Height': Height,'Weight': Weight,'QRS_duration': QRS_duration,'P_R_interval': P_R_interval,'Q_T_interval': Q_T_interval,'T_interval': T_interval,'P_interval': P_interval,'QRS': QRS,'T': T,'P': P,'QRST': QRST,'J': J,'Heart_rate': Heart_rate,'age2': age,'Height2': Height,'Weight2': Weight,'QRS_duration2': QRS_duration,'P_R_interval2': P_R_interval,'Q_T_interval2': Q_T_interval,'T_interval2': T_interval,'P_interval2': P_interval,'QRS2': QRS,'T2': T,'P2': P,'QRST2': QRST,'J2': J,'Heart_rate2': Heart_rate,'Heart_rate3': Heart_rate,'Heart_rate4': Heart_rate,}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()






data = pd.read_csv('Data1.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred) * 100
st.subheader('accuracy')
st.write(accuracy)




import time
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)
st.write(df)
    
st.subheader('Prediction')
z=svclassifier.predict(df)
st.write(z[0])
st.balloons()
  

