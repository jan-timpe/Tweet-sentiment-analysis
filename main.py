"""
Tweet sentiment analysis
Jan Timpe | jantimpe@uark.edu

Collects tweets, uses machine learning to predict sentament
Meant to be deployed with Apache Spark
"""
import csv, classifier, db, spark, time, write
import numpy as np

TRAIN_FILENAME = './data/smtrain.csv'
TEST_FILENAME = './data/smtest.csv'
SKLEARN_PATH = './data/models/sklearn'
SPARK_PATH = './data/models/spark'

if __name__ == '__main__':
    # read the training data
    xtrain, ytrain = classifier.readdata(TRAIN_FILENAME)
    xtest, ytest = classifier.readdata(TEST_FILENAME)
    countvect, transformer, model = classifier.train(xtrain, ytrain)
    acc, model = classifier.test(model, xtest, ytest, countvect, transformer)

    print('Sklearn acc: {}'.format(acc))

    print('Saving sklearn')
    classifier.save(SKLEARN_PATH, model, countvect, transformer)
    countvect, transformer, model = classifier.load(SKLEARN_PATH)
    print(classifier.predict(model, ['i like something'], countvect, transformer))

    # start looping over tweets
    # for tweet in db.gettweets():
    #     text = [tweet['text']]
    #     prediction = clf.predict(classifier.transform(text))
    #     print(text, ' ^^ ', prediction)
    #     time.sleep(0.5) 

    sc = spark.context('TwitterSentimentAnalysis')
    xdata, ydata = classifier.readdata(TRAIN_FILENAME)
    proc = spark.preprocess(sc, xdata, labels=ydata)

    traindata, testdata = spark.traintestsplit(proc)
    model = spark.train(traindata)
    acc, model = spark.test(model, testdata)
    print('Spark acc: {}'.format(acc))

    print('Saving spark')
    spark.save(model, sc, SPARK_PATH)

    sc, model = spark.load(sc, SPARK_PATH)
    xdata = ['oh my', 'cant believe i let that in', 'oops']
    print('Prediction:', spark.predict(sc, model, xdata))
