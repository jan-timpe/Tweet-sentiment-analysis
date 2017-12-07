"""
Tweet sentiment analysis
Jan Timpe | jantimpe@uark.edu

Collects tweets, uses machine learning to predict sentament
Meant to be deployed with Apache Spark
"""
import csv, classifier, db, getopt, re, spark, sys, time, write
import numpy as np
from settings import SKLEARN_PATH, SPARK_APPNAME, SPARK_PATH, TEST_FILENAME, TRAIN_FILENAME

def die_with_usage_help():
    helptext = """USAGE
 $ python main.py -{o}{o} --{option}
 + -d, --destroy    Delete all data
 + -h, --help       Bring up this prompt
 + -r, --refresh    Delete and reparse all data
 + -s, --summary    How much shit is in the database? Find out.
 + -o, --output     Dump the db into data/datasets/outputs
 + -b, --bucketize  Dump the db AND the buckets into data/datasets/outputs"""
    print(helptext)
    sys.exit(2)

def _loadtraindata(path=TRAIN_FILENAME):
    data, labels = classifier.readdata(path)
    return {
        'data': data,
        'labels': labels
    }

def _loadtestdata(path=TEST_FILENAME):
    data, labels = classifier.readdata(path)
    return {
        'data': data, 
        'labels': labels
    }

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
    
    opts = readargs(sys.argv[1:])

    run = {
        'data': {
            'train': None,
            'test':  None
        },
        'sklearn': None,
        'spark': None
    }

    if opts['load'] and opts['sklearn']:
        print('Loading sklearn')
        run['sklearn'] = _loadsklearn()

    if opts['load'] and opts['spark']:
        print('Loading spark')
        run['spark'] = _loadspark()



    if opts['train']:
        print('Loading training data')
        run['data']['train'] = _loadtraindata()

        if opts['sklearn']:
            print('Training sklearn')
            data = classifier.preprocess_text(run['data']['train']['data'])
            run['sklearn'] = _trainsklearn(
                data=data,
                labels=run['data']['train']['labels']
            )

        if opts['spark']:
            print('Training spark')
            run['spark'] = _trainspark(
                data=run['data']['train']['data'],
                labels=run['data']['train']['labels']
            )



    if opts['test']:
        print('Loading testing data')
        run['data']['test'] = _loadtestdata()

        if opts['sklearn']:
            print('Testing sklearn')
            data = classifier.preprocess_text(run['data']['test']['data'])
            acc, model = classifier.test(
                run['sklearn']['model'], 
                data,
                run['data']['train']['labels'], 
                run['sklearn']['countvect'], 
                run['sklearn']['transformer']
            )
            print('Accuracy: ', acc)

        if opts['spark']:
            print('Testing spark')
            data = spark.preprocess(
                run['spark']['context'], 
                run['data']['test']['data'], 
                labels=run['data']['test']['lables']
            )
            acc, model = spark.test(
                run['spark']['model'], 
                data
            )
            print('Accuracy: ', acc)



    if opts['save'] and opts['sklearn']:
        print('Saving sklearn')
        classifier.save(
            SKLEARN_PATH, 
            run['sklearn']['model'], 
            run['sklearn']['countvect'], 
            run['sklearn']['transformer']
        )

    if opts['save'] and opts['spark']:
        print('Saving spark')
        spark.save(
            run['spark']['model'], 
            run['spark']['context'],  
            SPARK_PATH
        )

    if opts['run'] and opts['sklearn']:
        print('Running sklearn')
        _runsklearn(
            run['sklearn']['model'], 
            run['sklearn']['countvect'], 
            run['sklearn']['transformer']
        )

    if opts['run'] and opts['spark']:
        print('Running spark')
        ha = 'haha'
