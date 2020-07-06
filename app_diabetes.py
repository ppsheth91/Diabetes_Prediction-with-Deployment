# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:52:28 2020

@author: PRAYAG
"""

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('diabetes_predict.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
     if request.method == 'POST':
        Pregnancies= float(request.form['Pregnancies'])
        Glucose=float(request.form['Glucose'])
        BloodPressure=float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin=float(request.form['Insulin'])
        BMI=float(request.form['BMI'])
        DiabetesPedigreeFunction =float(request.form['DiabetesPedigreeFunction'])
        Age=float(request.form['Age'])
        m_prediction = model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        return render_template('result.html', prediction=m_prediction)

if __name__ == '__main__':
	app.run(debug=True)