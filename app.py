import numpy as np
import pandas as pd
# matplotlib will be used for visually representing our data
import matplotlib.pyplot as plt
# Quandl will be used for importing historical oil prices
import quandl
from flask import Flask, render_template, redirect, request, url_for
from pymongo import MongoClient
import json,requests
import pickle

from project import X_test

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

model = pickle.load(open('silver.pkl', 'rb'))
@app.route('/silver',methods=['GET','POST'])
def silverTest():
    prediction = model.predict(X_test.iloc[-1])
    return render_template('silver_show.html', prediction = prediction) 

    # data = request.get_json(force=True)
    # prediction = model.predict([[np.array(data['date'])]])
    # return render_template('silver_show.html')

if __name__ == '__main__':
   app.run(debug=True)