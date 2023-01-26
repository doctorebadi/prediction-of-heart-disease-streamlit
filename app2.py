import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split # to split the dataset for training and testing 
from sklearn.metrics import accuracy_score

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
st.write(z)
