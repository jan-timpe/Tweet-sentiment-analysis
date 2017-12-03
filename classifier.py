import csv, pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

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

def tolibsvm(data, countvect=CountVectorizer(), transformer=TfidfTransformer()):
    counts = countvect.fit_transform(np.array(data))
    freqs = transformer.fit_transform(counts).toarray()
    return freqs

def fit_transform(data, countvect, transformer):
    counts = countvect.fit_transform(np.array(data))
    freqs = transformer.fit_transform(counts)
    return countvect, transformer, freqs

def transform(data, countvect, transformer):
    counts = countvect.transform(np.array(data))
    freqs = transformer.transform(counts)
    return countvect, transformer, freqs

def train(data, labels):
    model = MultinomialNB()
    countvect = CountVectorizer()
    transformer = TfidfTransformer()

    countvect, transformer, freqs = fit_transform(data, countvect, transformer)
    model.fit(freqs, labels)
    return countvect, transformer, model

def test(model, data, labels, countvect, transformer):
    countvect, transformer, freqs = transform(data, countvect, transformer)
    pred = model.predict(freqs)
    preds = []
    for i in range(len(pred)):
        preds.append((pred[i], labels[i]))

    acc = 1.0 * len([p for p in preds if p[0] == p[1]]) / len(data)
    return acc, model

def predict(model, textarr, countvect, transformer):
    countvect, transformer, freqs = transform(textarr, countvect, transformer)
    pred = model.predict(freqs)
    return pred

def save(path, model, countvect, transformer):
    modelpath = str(path) + '/model.pkl'
    with open(modelpath, 'wb') as file:
        pickle.dump(model, file)

    vectpath = str(path) + '/countvectorizer.pkl'
    with open(vectpath, 'wb') as file:
        pickle.dump(countvect, file)

    transfpath = str(path) + '/tfidftransformer.pkl'
    with open(transfpath, 'wb') as file:
        pickle.dump(transformer, file)

def load(path):
    model = None
    modelpath = str(path) + '/model.pkl'
    with open(modelpath, 'rb') as file:
        model = pickle.load(file)

    countvect = None
    vectpath = str(path) + '/countvectorizer.pkl'
    with open(vectpath, 'rb') as file:
        countvect = pickle.load(file)

    transformer = None
    transfpath = str(path) + '/tfidftransformer.pkl'
    with open(transfpath, 'rb') as file:
        transformer = pickle.load(file)

    return countvect, transformer, model