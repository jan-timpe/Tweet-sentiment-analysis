import csv, pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import re
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

def preprocess_text(data):
    proc = []
    for d in data:
        d = d.lower()
        # yeah this is real sexy
        # gets rid of all urls
        # how does it work? god only knows...
        d = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', d)
        
        words = word_tokenize(d)
        
        proc.append(' '.join(filtered_words))
    
    return proc

def tolibsvm(data, countvect=CountVectorizer(), transformer=TfidfTransformer(), array=False):
    # data = preprocess_text(data)
    counts = countvect.fit_transform(np.array(data))
    freqs = transformer.fit_transform(counts)

    if array:
        return countvect, transformer, freqs.toarray()

    return countvect, transformer, freqs

def transform(data, countvect, transformer):
    # data = preprocess_text(data)
    counts = countvect.transform(np.array(data))
    freqs = transformer.transform(counts)
    return countvect, transformer, freqs

def train(data, labels):
    model = MultinomialNB()
    countvect = CountVectorizer()
    transformer = TfidfTransformer()

    countvect, transformer, freqs = tolibsvm(data, countvect, transformer)
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
    conf = model.predict_proba(freqs)
    
    result = [(pred[i], conf[i]) for i in range(len(pred))]
    return result

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