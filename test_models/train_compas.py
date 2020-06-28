import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# read in the propublica data to be used for our analysis.
propublica_data = pd.read_csv(
    filepath_or_buffer="../test_data/propublica_data_for_fairml.csv")

# create feature and design matrix for model building.
compas_rating = propublica_data.score_factor.values
propublica_data = propublica_data.drop("score_factor", 1)

propublica_data.to_csv('../test_data/propublica.csv', index=False)

# this is just for demonstration, any classifier or regressor
# can be used here. fairml only requires a predict function
# to diagnose a black-box model.

# we fit a quick and dirty logistic regression sklearn
# model here.
# clf = LogisticRegression(penalty='l2', C=0.01)
# clf.fit(propublica_data.values, compas_rating)

# compas_path = './compas_model.obj'
# filehandler = open(compas_path, 'wb')
# pickle.dump(clf, filehandler)
# filehandler.close()

# print(clf)


model_names = ['LinearRegression', 'LogisticRegression', 'RandomForest', 'Adaboost', 'DecisionTree']
models = [LinearRegression(), LogisticRegression(penalty='l2', C=0.01), RandomForestClassifier(random_state=0), AdaBoostRegressor(), DecisionTreeRegressor()]

for model_name, model in zip(model_names, models):

    # train model
    model.fit(propublica_data.values, compas_rating)

    # pickle model
    path = './' + model_name + '.pkl'
    filehandler = open(path, 'wb')
    pickle.dump(model, filehandler)
    filehandler.close()

