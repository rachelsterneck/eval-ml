import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# read in the propublica data to be used for our analysis.
# propublica_data = pd.read_csv(
#     filepath_or_buffer="../test_data/propublica_data_for_fairml.csv")

# # create feature and design matrix for model building.
# compas_rating = propublica_data.score_factor.values
# propublica_data = propublica_data.drop("score_factor", 1)

# propublica_data.to_csv('../test_data/propublica.csv', index=False)


loans_data = pd.read_csv(
    filepath_or_buffer="../test_data/full_data_loans.csv")

# create feature and design matrix for model building.

loans_data = loans_data.drop("Loan_ID", 1)
cat_list = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', "Property_Area", "Loan_Status"]
for name in cat_list:
    loans_data[name] = loans_data[name].astype('category')

loans_data = loans_data.dropna()

cat_columns = loans_data.select_dtypes(['category']).columns
loans_data[cat_columns] = loans_data[cat_columns].apply(lambda x: x.cat.codes)

y = loans_data.Loan_Status.values
loans_data = loans_data.drop("Loan_Status", 1)


loans_data.to_csv('../test_data/loans.csv', index=False)

model = LogisticRegression()
model_name = "LogisticRegression"

# train model
model.fit(loans_data.values, y)

# pickle model
path = './loans/' + model_name + '.pkl'
filehandler = open(path, 'wb')
pickle.dump(model, filehandler)
filehandler.close()



# model_names = ['LinearRegression', 'LogisticRegression', 'RandomForest', 'Adaboost', 'DecisionTree']
# models = [LinearRegression(), LogisticRegression(penalty='l2', C=0.01), RandomForestClassifier(random_state=0), AdaBoostRegressor(), DecisionTreeRegressor()]

# for model_name, model in zip(model_names, models):
#     # train model
#     model.fit(loans_data.values, y)

#     # pickle model
#     path = './loans/' + model_name + '.pkl'
#     filehandler = open(path, 'wb')
#     pickle.dump(model, filehandler)
#     filehandler.close()


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


