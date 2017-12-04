import classifier
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel, LabeledPoint
from pyspark.mllib.util import MLUtils
import shutil

def preprocess(sc, data, labels=None):
    data = classifier.tolibsvm(data)
    points = []
    for i in range(len(data)):
        wordarr = data[i]
        label = 0.0
        if labels:
            label = labels[i]
        point = LabeledPoint(label, wordarr)
        points.append(point)
 
    rdd = sc.parallelize(points)
    return rdd

def pre(sc, data):
    data = classifier.tolibsvm(data)
    points = []
    for i in range(len(data)):
        points.append(LabeledPoint(0.0, data[i]))
    rdd = sc.parallelize(points)
    return rdd

def traintestsplit(data):
    return data.randomSplit([0.6, 0.4])

def context(appname):
    sc = SparkContext(appName=appname)
    return sc

def train(data):
    model = NaiveBayes.train(data, 1.0)
    return model

def test(model, data):
    pred = data.map(lambda p: (model.predict(p.features), p.label))
    acc = 1.0 * pred.filter(lambda pl: pl[0] == pl[1]).count() / data.count()
    return acc, model

def save(model, sc, filename):
    shutil.rmtree(filename, ignore_errors=True)
    model.save(sc, filename)

def load(sc, filename):
    model = NaiveBayesModel.load(sc, filename)
    return sc, model

def predict(sc, model, data):
    return data.map(lambda p: model.predict(p.features))
