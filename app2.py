import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split # to split the dataset for training and testing 
from sklearn.metrics import accuracy_score

st.write("""
# Simple Prediction of heart disease Prediction App
""")


st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
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
st.write(accuracy)

n=np.array([75,0,190,80,91,193,371,174,121,-16,13,64,-2,63,0,75,0,190,80,95,190,401,169,125,-15,13,68,12,63,0]).reshape(1,-1)
z=svclassifier.predict(n)
st.write(z[0])
