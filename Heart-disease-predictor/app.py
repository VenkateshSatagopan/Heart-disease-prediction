from flask import Flask, render_template,request,jsonify
import pandas as pd
import pickle
import numpy as np
from pycaret.classification import *

model=load_model('best-model')

app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def get_details():
    d = None
    if request.method=='POST':
        d = request.form.to_dict()
        df = pd.DataFrame([d.values()], columns=d.keys())
        prediction_val=predict_model(model, data=df, round = 0)
        prediction_val=int(prediction_val.Label[0])

        if prediction_val==0:
            prediction='that the person have heart disease'
        else:
            prediction='that the person do not have heart disease'

        return render_template("index.html",prediction='The model predicts {}'.format(prediction))

    return render_template("index.html",prediction="No data has been provided yet")
    
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
      int_features = [x for x in request.form.values()]
      return render_template('index.html', prediction=-1)
    return render_template("index.html",prediction="No data has been provided yet")

if __name__=='__main__':
    app.run(debug=True,port=300)