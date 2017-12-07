"""
Tweet sentiment analysis
Jan Timpe | jantimpe@uark.edu

Collects tweets, uses machine learning to predict sentament
Meant to be deployed with Apache Spark
"""
import csv, classifier, db, spark, time, write
import numpy as np
import re

TRAIN_FILENAME = './data/smtrain.csv'
TEST_FILENAME = './data/smtest.csv'
SKLEARN_PATH = './data/models/sklearn'
SPARK_PATH = './data/models/spark'

def readargs(argv):
    try:
        opts, args = getopt.getopt(
            argv,
            'dhrsob',
            ['destroy', 'help', 'refresh', 'summary', 'output', 'bucketize']
        )
    except getopt.GetoptError:
        die_with_usage_help()

    for opt, arg in opts:
        if opt in ('-d', '--destroy'):
            rebuild_tables()
        elif opt in ('-h', '--help'):
            die_with_usage_help()
        elif opt in ('-r', '--refresh'):
            refresh_data()
        elif opt in ('-s', '--summary'):
            print_db_summary()
        elif opt in ('-o', '--output'):
            output.dump()
        elif opt in ('-b', '--bucketize'):
            output.dump(bucketsout=True)

if __name__ == '__main__':
    # read the training data
    # print('Extracting...')
    # xtrain, ytrain = classifier.readdata(TRAIN_FILENAME)
    # xtest, ytest = classifier.readdata(TEST_FILENAME)

    # # train the model
    # print('Training...')
    # countvect, transformer, model = classifier.train(xtrain, ytrain)

    # # test the model
    # print('Testing...')
    # acc, model = classifier.test(model, xtest, ytest, countvect, transformer)
    # print('Test accuracy: {}'.format(acc))

    # # save the model
    # print('Saving...')
    # classifier.save(SKLEARN_PATH, model, countvect, transformer)

    # load the model
    # print('Loading...')
    # countvect, transformer, model = classifier.load(SKLEARN_PATH)

    # start looping over tweets
    print('Starting...')
    pos = 0
    neg = 0
    num = 0
    for tweet in db.gettweets():
        text = str(tweet['text'])
        pred = classifier.predict(model, [text], countvect, transformer)

        ispos = pred[0][0] == 0 # yeah it's flipped. of course it's flipped.
        conf = pred[0][1][0] if ispos else pred[0][1][1]

        if conf > 0.6:
            pos += 1 if ispos else 0
            num += 1

            if num % 10 == 0:
                print('{} tweets: [ {}+, {}- ]'.format(
                    num, 
                    pos, 
                    num-pos
                ))
                print('last: [ {} ] [ {}, {}% ]'.format(
                    text, 
                    'positive' if ispos else 'negative', 
                    int(conf*100)
                ))

    print('Done!')
    print(num, 'tweets processed')
    print(pos, 'with positive sentiment')
    print(num-pos, 'with negative')



    # sc = spark.context('TwitterSentimentAnalysis')
    # xdata, ydata = classifier.readdata(TRAIN_FILENAME)
    # proc = spark.preprocess(sc, xdata, labels=ydata)

    # traindata, testdata = spark.traintestsplit(proc)
    # model = spark.train(traindata)
    # acc, model = spark.test(model, testdata)
    # print('Spark acc: {}'.format(acc))

    # xdata = ['oh my', 'cant believe i let that in', 'oops']
    # xdata = spark.pre(sc, xdata)
    # pred = spark.predict(sc, model, xdata)
    # print('Prediction:', pred.collect())

    # print('Saving spark')
    # spark.save(model, sc, SPARK_PATH)

    # sc, model = spark.load(sc, SPARK_PATH)
    # xdata = ['oh my', 'cant believe i let that in', 'oops']
    # xdata = spark.preprocess(sc, xdata)
    # print('Prediction:', spark.predict(sc, model, xdata))
