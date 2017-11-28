"""
Tweet sentiment analysis
Jan Timpe | jantimpe@uark.edu

Collects tweets, uses machine learning to predict sentament
Meant to be deployed with Apache Spark
"""
import csv, classifier, db, time
import numpy as np

if __name__ == '__main__':
    # read the training data
    clf = classifier.train()

    # test the thing
    classifier.test()

    # start looping over tweets
    # for tweet in db.gettweets():
    #     text = [tweet['text']]
    #     prediction = clf.predict(classifier.transform(text))
    #     print(text, ' ^^ ', prediction)
    #     time.sleep(0.5)