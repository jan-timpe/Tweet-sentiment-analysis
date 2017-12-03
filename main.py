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

if __name__ == '__main__':
    # read the training data
    xtrain, ytrain = classifier.readdata(TRAIN_FILENAME)
    xtest, ytest = classifier.readdata(TEST_FILENAME)
    countvect, transformer, model = classifier.train(xtrain, ytrain)
    acc, model = classifier.test(model, xtest, ytest, countvect, transformer)

    print('Sklearn acc: {}'.format(acc))

    # start looping over tweets
    # for tweet in db.gettweets():
    #     text = [tweet['text']]
    #     prediction = clf.predict(classifier.transform(text))
    #     print(text, ' ^^ ', prediction)
    #     time.sleep(0.5) 

    xdata, ydata = classifier.readdata(TRAIN_FILENAME)
    sc = spark.context('TwitterSentimentAnalysis')
    proc = spark.preprocess(sc, xdata, ydata)

    traindata, testdata = spark.traintestsplit(proc)
    model = spark.train(traindata)
    acc, model = spark.test(model, testdata)
    print('Spark acc: {}'.format(acc))