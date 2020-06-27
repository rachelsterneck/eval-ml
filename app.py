from flask import Flask, render_template, request, redirect, url_for
from bson import ObjectId
from pymongo import MongoClient
import os

from dotenv import load_dotenv
load_dotenv()

# get env variables
MONGO_CLIENT = os.getenv('MONGO_CLIENT')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')

app = Flask(__name__)
title = "Eval ML"
heading = "Eval ML Heading"

client = MongoClient(MONGO_CLIENT)
db = client.test
db.authenticate(name="localhost",password=MONGO_PASSWORD)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # User is requesting the form
        return render_template('form.html')
    elif request.method == 'POST':
        # User has sent us data
        image = request.files['model']
        data = request.files['data']
        client = ComputerVisionClient(COGSVCS_CLIENTURL, CognitiveServicesCredentials(COGSVCS_KEY))
        result = client.describe_image_in_stream(image)
        message = 'No dog found. How sad. :-('
        if 'dog' in result.tags:
            message = 'There is a dog! Wonderful!!'
        return render_template('result.html', message=message)

if __name__ == "__main__":
    app.run()