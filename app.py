from flask import Flask, render_template, request, redirect, url_for
from bson import ObjectId
from pymongo import MongoClient
import os
import pandas as pd
import numpy as np
import pickle

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
collection = db['scores']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # User is requesting the form
        return render_template('form.html')
    elif request.method == 'POST':
        data = pd.read_csv(request.files['datafile'])
        model_pickle = request.files['modelfile']
        model = pickle.load(model_pickle)
        #  call audit model with model
        importances, _ = audit_model(model.predict, data)

        importances_dict = {}
        for key, value in dict(importances).items():
            importances_dict[key] = np.median(np.array(value))

        data_dict = {'importances': importances_dict}
        data_dict['dataset'] = request.files['datafile'].name
        data_dict['public_sharing'] = False

        document_id = collection.insert(data_dict)

        return render_template('result.html', document_id=document_id, importances_dict=importances_dict)

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
        return render_template('ranking.html')

if __name__ == "__main__":
    app.run(debug=True)