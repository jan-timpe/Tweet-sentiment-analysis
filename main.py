"""
Tweet sentiment analysis
Jan Timpe | jantimpe@uark.edu

Collects tweets, uses machine learning to predict sentament
Meant to be deployed with Apache Spark
"""
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def readdata(filename):
    file = open(filename)
    reader = csv.reader(file)

    xdata = []
    ydata = []
    for row in reader:
        xdata.append(row[3].strip())
        ydata.append(int(row[1].strip()))

    return (xdata, ydata)


# read the training data
xdata, ydata = readdata('./data/train.csv')

# create and fit vectorizer and transformer objects
vect = CountVectorizer()
counts = vect.fit_transform(np.array(xdata))
transf = TfidfTransformer()
freqs = transf.fit_transform(counts)

# train the classifier
classifier = MultinomialNB()
classifier.fit(freqs, ydata)

# grab the test data
xtest, ytest = readdata('./data/test.csv')
res_counts = vect.transform(xtest)
res_freqs = transf.transform(res_counts)

# woohoo
predictions = classifier.predict(res_freqs)
y = 0
correct = 0
wrong = 0
for pred in predictions:
    if pred == ytest[y]:
        correct += 1
    else:
        wrong += 1
    y += 1

perc = int(correct/(wrong + correct) * 100)
print('Howd it go? Correct:', correct, '; Wrong:', wrong, '; Percent:', perc)

# time for some mongo
from pymongo import MongoClient, errors
from settings import ATLAS_URI
import time

try:
    mongo = MongoClient(ATLAS_URI)
except errors.InvalidURI:
    print('Oops, couldnt connect')
    exit(1)

db = mongo.tweetstream.harvey

for tweet in db.tweets.find():
    text = [tweet['text']]
    res_counts = vect.transform(text)
    res_freqs = transf.transform(res_counts)
    prediction = classifier.predict(res_freqs)
    print(text, ' ^^ ', prediction)
    time.sleep(0.5)