import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
vect = CountVectorizer()
transf = TfidfTransformer()

def readdata(filename):
    file = open(filename)
    reader = csv.reader(file)

    xdata = []
    ydata = []
    for row in reader:
        xdata.append(row[3].strip())
        ydata.append(int(row[1].strip()))
    file.close()
    return (xdata, ydata)

def tolibsvm(data):
    counts = vect.fit_transform(np.array(data))
    freqs = transf.fit_transform(counts).toarray()
    return freqs

def fit_transform(data):
    counts = vect.fit_transform(np.array(data))
    freqs = transf.fit_transform(counts)
    return freqs

def train():
    xdata, ydata = readdata('./data/train.csv')
    freqs = fit_transform(xdata)
    clf.fit(freqs, ydata)
    return clf

def transform(str_array):
    counts = vect.transform(np.array(str_array))
    freqs = transf.transform(counts)
    return freqs

def test():
    xtest, ytest = readdata('./data/test.csv')
    predictions = clf.predict(transform(xtest))
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