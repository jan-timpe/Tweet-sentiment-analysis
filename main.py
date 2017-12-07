"""
Tweet sentiment analysis
Jan Timpe | jantimpe@uark.edu

Collects tweets, uses machine learning to predict sentament
Meant to be deployed with Apache Spark
"""
import csv, classifier, db, getopt, re, spark, sys, time, write
import numpy as np
from settings import SKLEARN_PATH, SPARK_APPNAME, SPARK_PATH, TERMS, TEST_FILENAME, TRAIN_FILENAME
from tweetstream.collect import stream

def die_with_usage_help():
    helptext = """USAGE
 $ python main.py -{o}{o} --{option}
  + -h, --help      Show this dialogue
  + -l, --load      Load saved models   [specify -p or -k]
  + -r, --run       Run the predictor   [specify -p or -k]
  + -s, --save      Save models         [specify -p or -k]
  + -p, --spark     Specify spark
  + -k, --sklearn   Specify sklearn
  + -t, --test      Test the models     [specify -p or -k]
  + -n, --train     Train the models    [specify -p or -k]
    """
    print(helptext)
    sys.exit(2)

def _loadtraindata(path=TRAIN_FILENAME):
    data, labels = classifier.readdata(path)
    return data, labels

def _loadtestdata(path=TEST_FILENAME):
    data, labels = classifier.readdata(path)
    return data,  labels

def _loadspark(appname=SPARK_APPNAME, path=SPARK_PATH):
    sc = spark.context(appname)
    sc, model = spark.load(sc, path)
    return {
        'context': sc,
        'model': model
    }

def _loadsklearn():
    countvect, transformer, model = classifier.load(SKLEARN_PATH)
    return {
        'countvect': countvect,
        'model': model,
        'transformer': transformer
    }

def _trainsklearn(data, labels):
    countvect, transformer, model = classifier.train(data, labels)
    return {
        'countvect': countvect,
        'model': model,
        'transformer': transformer
    }

def _trainspark(data, labels, appname=SPARK_APPNAME):
    sc = spark.context(appname)
    proc = spark.preprocess(sc, data, labels=labels)
    model = spark.train(proc)
    return {
        'context': sc,
        'model': model
    }

def _runsklearn(model, countvect, transformer):
    # start looping over tweets
    print('Starting...')
    pos = 0
    neg = 0
    num = 0
    totconf = 0
    for tweet in stream(TERMS):
        if num > 1000:
            break

        text = str(tweet['text'])
        pred = classifier.predict(model, [text], countvect, transformer)[0]

        posval = 0 # this motherfucker gets confusing
        negval = 1
        ispos = pred[0] == posval
        conf = pred[1][posval] if ispos else pred[1][negval]

        if conf > 0.6:
            pos += 1 if ispos else 0
            num += 1
            totconf += conf

            if num % 10 == 0:
                print('{} tweets: [ {}+, {}- ]'.format(num, pos, num-pos))
                print('last: [ {} ] [ {}, {}% ]'.format(text, 'positive' if ispos else 'negative', int(conf*100)))

    print('Done!')
    print(num, 'tweets processed')
    print(pos, 'with positive sentiment')
    print(num-pos, 'with negative')
    print((num-pos)*100/num, '% negative')
    print('Avg confidence:', (100*totconf)/num)

def readargs(argv):
    try:
        opts, args = getopt.getopt(
            argv,
            'hlrspktn',
            ['help', 'load', 'run', 'save', 'spark', 'sklearn', 'test', 'train']
        )
    except getopt.GetoptError:
        die_with_usage_help()

    runopts = {
        'load': False,
        'run': False,
        'save': False,
        'spark': False,
        'sklearn': False,
        'test': False,
        'train': False
    }

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            die_with_usage_help()
        if opt in ('-l', '--load'):
            runopts['load'] = True
        elif opt in ('-r', '--run'):
            runopts['run'] = True
        elif opt in ('-s', '--save'):
            runopts['save'] = True
        elif opt in ('-p', '--spark'):
            runopts['spark'] = True
        elif opt in ('-k', '--sklearn'):
            runopts['sklearn'] = True
        elif opt in ('-t', '--test'):
            runopts['test'] = True
        elif opt in ('-n', '--train'):
            runopts['train'] = True

    return runopts

if __name__ == '__main__':
    
    # print(classifier.preprocess_text(['suck mah tay tays', 'th1s @is #fucking http://DEd.net']))

    opts = readargs(sys.argv[1:])

    sklearnclf = None
    sparkclf = None

    if opts['load'] and opts['sklearn']:
        print('Loading sklearn')
        sklearnclf = _loadsklearn()

    if opts['load'] and opts['spark']:
        print('Loading spark')
        sparkclf = _loadspark()



    if opts['train']:
        print('Loading training data')
        data, labels = _loadtraindata()

        print('Preprocessing raw text')
        data = classifier.preprocess_text(data)

        if opts['sklearn']:
            print('Training sklearn')
            sklearnclf = _trainsklearn(data=data, labels=labels)

        if opts['spark']:
            print('Training spark')
            sparkclf = _trainspark(data=data, labels=labels)


    if opts['test']:
        print('Loading testing data')
        data, labels = _loadtestdata()

        print('Preprocessing raw text')
        data = classifier.preprocess_text(data)

        if opts['sklearn'] and sklearnclf:
            print('Testing sklearn')
            
            acc, model = classifier.test(
                sklearnclf['model'],
                data,
                labels,
                sklearnclf['countvect'],
                sklearnclf['transformer']
            )
            print('Accuracy: ', acc)

        if opts['spark'] and sparkclf:
            print('Testing spark')
            data = spark.preprocess(
                sparkclf['context'],
                data,
                labels=labels
            )
            acc, model = spark.test(
                sparkclf['model'], 
                data
            )
            print('Accuracy: ', acc)



    if opts['save'] and opts['sklearn'] and sklearnclf:
        print('Saving sklearn')
        classifier.save(
            SKLEARN_PATH,
            sklearnclf['model'],
            sklearnclf['countvect'],
            sklearnclf['transformer']
        )

    if opts['save'] and opts['spark'] and sparkclf:
        print('Saving spark')
        spark.save(
            sparkclf['model'],
            sparkclf['context'],
            SPARK_PATH
        )

    if opts['run'] and opts['sklearn'] and sklearnclf:
        print('Running sklearn')
        _runsklearn(
            sklearnclf['model'],
            sklearnclf['countvect'],
            sklearnclf['transformer']
        )

    if opts['run'] and opts['spark'] and sparkclf:
        print('Running spark')
        ha = 'haha'
