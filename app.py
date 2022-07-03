from flask import Flask, request, render_template, jsonify
from flask_navigation import Navigation
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import pickle
import sys

app = Flask(__name__)
nav = Navigation(app)

nav.Bar('top', [
    nav.Item('About', 'index'),
    nav.Item('ML Test', 'input')
])

loaded_model = pickle.load(open("model_pkl", "rb"))
encoder = pickle.load(open("model_ohe", "rb"))
scaler = pickle.load(open("model_scaler", "rb"))
scaled_vars = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
       'pcv', 'pc_normal', 'pcc_present', 'ba_present', 'htn_yes', 'dm_yes',
       'cad_yes', 'appet_poor', 'ane_yes', 'pe_yes']
categorical_vars = ['rbc','pc','pcc','ba','htn','dm','cad','appet','ane','pe']
columns = ['sg', 'al', 'bu', 'sc', 'sod', 'hemo', 'pcv', 'pc', 'htn', 'dm']

def get_df(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 10)
    data = pd.DataFrame(to_predict, columns=columns)
    return data

def encode_data(data):
    data['rbc'], data['pe'], data['cad'], data['pcc'], data['appet'], data['ba'], data['ane'], data['su']= ['normal','no','no','notpresent','good','notpresent','no','3']
    encoded = encoder.transform(data[categorical_vars])
    encoder_feature_names = encoder.get_feature_names(categorical_vars)
    encoder_vars_df = pd.DataFrame(encoded, columns = encoder_feature_names)
    data_new = pd.concat([data.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis = 1)
    data_new.drop(categorical_vars, axis = 1, inplace = True)
    data = data_new
    return data

def drop_features(data):
    data = data.drop(['age', 'bp', 'bgr', 'pot', 'pcc_present', 'ba_present', 'cad_yes', 'appet_poor', 'ane_yes', 'pe_yes', 'su'], 1)
    return data

def scale_data(data):
    data['age'], data['bp'], data['bgr'], data['pot'] = ['43','80','131','3.5']
    data_reorg = data[['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
           'pcv', 'pc_normal', 'pcc_present', 'ba_present', 'htn_yes', 'dm_yes',
           'cad_yes', 'appet_poor', 'ane_yes', 'pe_yes']]
    scaled = scaler.transform(data_reorg)
    scaled = pd.DataFrame(scaled, columns = scaled_vars)
    print(scaled, file=sys.stdout)
    print(data, file=sys.stdout)
    return scaled

def predict_case(data):
    result = loaded_model.predict(data)
    if int(result)== 0:
        message ='Likely positve'
    else:
        message ='Likely negative'
    raw_proba = loaded_model.predict_proba(data)
    result_list = [result, message, raw_proba]
    return result_list

def get_model_params():
    params = loaded_model.get_params()
    return params

@app.route('/')
def index():
    return render_template('about.html')

@app.route('/about')
def about():
        return render_template('about.html')

@app.route('/input')
def input():
    return render_template('user-input.html')

@app.route('/pred', methods = ['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        data = get_df(to_predict_list)
        encoded = encode_data(data)
        scaled_data = scale_data(encoded)
        df = drop_features(scaled_data)
        result_list = predict_case(df)
        # if int(result_list[0])== 1:
        #     prediction ='Patient may have Chronic Kidney Disease'
        # else:
        #     prediction ='Patient likely does not have Chronic Kidney Disease'
        return render_template("result.html", prediction = result_list[1], proba = result_list[2], params = get_model_params())

if __name__ == '__main__':
    app.run()
