"""
Tweet sentiment analysis
Jan Timpe | jantimpe@uark.edu

Collects tweets, uses machine learning to predict sentament
Meant to be deployed with Apache Spark
"""
import csv, classifier, db, spark, time, write
import numpy as np

if __name__ == '__main__':
    # read the training data
    # clf = classifier.train()
    # classifier.test()

    # # start looping over tweets
    # for tweet in db.gettweets():
    #     text = [tweet['text']]
    #     prediction = clf.predict(classifier.transform(text))
    #     print(text, ' ^^ ', prediction)
    #     time.sleep(0.5)

    xdata, ydata = classifier.readdata('./data/smtrain.csv')
    sc = spark.context('TwitterSentimentAnalysis')
    samp = spark.readsample(sc)
    proc = spark.preprocess(sc, xdata, ydata)

    traindata, testdata = spark.traintestsplit(proc)
    model = spark.train(traindata)
    acc, model = spark.test(model, testdata)
    print('Acc: {}'.format(acc))