from flask import Flask, render_template, request, redirect, url_for
from bson import ObjectId
from pymongo import MongoClient
import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.linear_model import LogisticRegression, LinearRegression

from fairml import audit_model
#from fairml import plot_generic_dependence_dictionary
from dotenv import load_dotenv
load_dotenv()

# get env variables
MONGO_CLIENT = os.getenv('MONGO_CLIENT')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')

app = Flask(__name__)
title = "Eval ML"
heading = "Eval ML Heading"

client = MongoClient(MONGO_CLIENT)
db = client['fairrank']
collection = db['scores_prod']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # User is requesting the form
        return render_template('form.html')
    elif request.method == 'POST':
        #data = pd.read_csv(request.files['datafile'])
        #model_pickle = request.files['modelfile']
        #model = pickle.load(model_pickle)

        ### TEMP 
    
        loans_data = pd.read_csv(
            filepath_or_buffer="test_data/full_data_loans.csv")

        loans_data = loans_data.drop("Loan_ID", 1)
        cat_list = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', "Property_Area", "Loan_Status"]
        for name in cat_list:
            loans_data[name] = loans_data[name].astype('category')

        loans_data = loans_data.dropna()

        cat_columns = loans_data.select_dtypes(['category']).columns
        loans_data[cat_columns] = loans_data[cat_columns].apply(lambda x: x.cat.codes)

        y = loans_data.Loan_Status.values
        loans_data = loans_data.drop("Loan_Status", 1)


        #loans_data.to_csv('../test_data/loans.csv', index=False)

        model = LogisticRegression()
        model_name = "LogisticRegression"

        # train model
        model.fit(loans_data.values, y)

        ### TEMP

        #  call audit model with model
        importances, _ = audit_model(model.predict, loans_data)
        print('printing importances')
        print(importances)

        importances_dict = {}
        for key, value in dict(importances).items():
            importances_dict[key] = np.median(np.array(value))

        data_dict = {'importances': importances_dict}
        data_dict['dataset'] = request.files['datafile'].filename
        data_dict['model'] = request.files['modelfile'].filename
        data_dict['public_sharing'] = False

        document_id = collection.insert(data_dict)

        features_array = list(importances_dict.keys())
        importances_array = list(importances_dict.values())

        return render_template('result.html', document_id=document_id, features_array=json.dumps(features_array), 
            importances_array=json.dumps(importances_array))

@app.route('/result', methods=['POST'])
def results():
    if request.method == 'POST':
        mongo_id = request.form['dataId']

        data_dict = {}
        data_dict['organization'] = request.form['organization']
        data_dict['model-description'] = request.form['model-description']
        data_dict['data-description'] = request.form['data-description']
        data_dict['public_sharing'] = True

        new_values = { "$set": data_dict } 
        collection.update({'_id': ObjectId(mongo_id)}, new_values, upsert=False)

        return redirect(url_for('ranking'))


@app.route('/ranking', methods=['GET'])
def ranking():
    if request.method == 'GET':

        collection.find({'dataset'})
        # 
        """

        {
            "datasetname1": {
                features: []
                models: {
                    modelname1: [list of scores]
                    modelname2: [list of scores]
                }
            },
            "datasetname1": {
                features: []
                models: {
                    modelname1: [list of scores]
                    modelname2: [list of scores]
                }
            }

        }
        """

        return render_template('ranking.html')

if __name__ == "__main__":
    app.run(debug=True)