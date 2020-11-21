## Flask app for Vehical Insurance prediction ML project #####

from flask import Flask, render_template, request
import numpy as np
import pickle
from model_files.Insurance_model import preprocessing


model = pickle.load(open("model.pkl",'rb'))


app= Flask(__name__)


@app.route('/')
def home():
    return render_template("main.html")


@app.route('/predict',methods=['POST'])
def predict():

        values = np.array([int(i) for i in request.form.values()]).reshape(-1,1)
        final = preprocessing(values)
        result = model.predict(final)
        output = {1:"interested",0:"not interested"}
        return render_template('main.html', prediction_text='Person is {} in vehicle insurance!'.format(output[result[0]]))


if __name__=='__main__':
    app.run(debug=True)
    
        
