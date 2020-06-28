from flask import Flask, render_template, request, redirect, url_for
from bson import ObjectId
from pymongo import MongoClient
import os

from fairml import audit_model
from fairml import plot_generic_dependence_dictionary
from dotenv import load_dotenv
load_dotenv()

# get env variables
MONGO_CLIENT = os.getenv('MONGO_CLIENT')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')

app = Flask(__name__)
title = "Eval ML"
heading = "Eval ML Heading"

client = MongoClient(MONGO_CLIENT)
db = client.scores


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # User is requesting the form
        return render_template('form.html')
    elif request.method == 'POST':
        # User has sent us data
        model = request.files['model']
        data = request.files['data']

        #  call audit model with model
        importances, _ = audit_model(model.predict, data)

        # print feature importance
        print(importances)
        fig = plot_dependencies(
            importances.get_compress_dictionary_into_key_median(),
            reverse_values=False,
            title="FairML feature dependence"
        )

        plt.show()


        return render_template('result.html', message=message)

if __name__ == "__main__":
    app.run()