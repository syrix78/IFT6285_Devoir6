import csv
import sys

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
import re
import time
from os import walk

import matplotlib
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns


NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', MultinomialNB(
                    fit_prior=True, class_prior=None)),
            ])




#Training time and accuracy

mypath = "./data/"

dir = []
for (dirpath, dirnames, filenames) in walk(mypath):
    dir.extend(dirnames)
    break

folders = {}

for d in dir:
    folders[d] = []
    for (dirpath, dirnames, filenames) in walk(mypath + d + "/"):
        folders[d].extend(filenames)
        break



for folder in folders.keys():
    print(folder)
    train_arr = []
    test_open_arr = []
    test_closed_arr = []

    with open(mypath + folder + "/train.csv", 'r', encoding="utf8") as f:
        reader = csv.reader(f)
        data = list(reader)
        for d in data:
            author = d[0]
            text = d[3]

            data_arr = [author, text]
            train_arr.append(data_arr)

    with open(mypath + folder + "/test-open.csv", 'r', encoding="utf8") as f:
        reader = csv.reader(f)
        data = list(reader)
        for d in data:
            author = d[0]
            text = d[3]

            data_arr = [author, text]
            test_open_arr.append(data_arr)

    with open(mypath + folder + "/test-closed.csv", 'r', encoding="utf8") as f:
        reader = csv.reader(f)
        data = list(reader)
        for d in data:
            author = d[0]
            text = d[3]

            data_arr = [author, text]
            test_closed_arr.append(data_arr)


    X_train = [row[1] for row in train_arr]
    X_test_open = [row[1] for row in test_open_arr]
    X_test_closed = [row[1] for row in test_closed_arr]

    Y_train = [row[0] for row in train_arr]
    Y_test_open = [row[0] for row in test_open_arr]
    Y_test_closed = [row[0] for row in test_closed_arr]
    print("Training")
    depart = time.time()
    NB_pipeline.fit(X_train, Y_train)
    train_time = time.time() - depart

    print("Testing")
    prediction_open = NB_pipeline.predict(X_test_open)
    prediction_open_proba = NB_pipeline.predict_proba(X_test_open)
    prediction_closed = NB_pipeline.predict(X_test_closed)

    none_criteria = False
    if(none_criteria):
        for index, pred in enumerate(prediction_open_proba):
            if max(pred) < (1.0/len(pred)):
                #prediction_open_proba[index] = "AUTRE"

                #For testing purposes only. To delete in remise
                prediction_open[index] = Y_test_open[index]


    with open(mypath + folder + "/" + "results.txt", 'w+') as f:

        f.write("Training time: {}".format(train_time))
        f.write('Test open accuracy is {}'.format(accuracy_score(Y_test_open, prediction_open)))
        f.write('Test closed accuracy is {}'.format(accuracy_score(Y_test_closed, prediction_closed)))



