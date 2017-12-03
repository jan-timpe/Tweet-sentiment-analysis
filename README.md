# Tweet sentiment analysis

Machine learning to predict the sentiment of tweets

## Setup and run

Fill out `example_settings.py` with your own credentials and options, then rename the file to `settings.py`

Make sure python3 is installed

Create a virtualenv by running
```$ virtualenv venv -p python3```

Activate the virtualenv by running 
```$ source ./venv/bin/activate```

Install requirements with pip
```$ pip install -r requirements.txt```

Run the sklearn model
```$ python main.py```

Run the spark model
```$ python spark.py```

## How it works

Two classifiers are running here, sklearn and Apache Spark examples of the Multinomial Naive Bayes classifier. The MultinomialNB is the classic choice for text analysis in machine learning. 

The best way to see what's happening behind-the-scenes is to look at the sklearn implementation in `classifier.py`, following the `main.py` entrypoint.

### sklearn

In `main.py`, `readdata()` is called twice, producing two raw datasets, having undergone no preprocessing. In effect, this is the tweet text and the 'sentiment rating' (1 for positive sentiment, 0 for negative) for each. Next, the classifier is trained with the call to `classifier.train()`. 

In `classifier.py`, the `train()` function takes `data` (tweet text) and `labels` (sentiment rating) as arguments and creates three new objects, a `MultinomialNB` classifier, a `CountVectorizer` to produce a word count vector from each tweet, and a `TfidfTransformer` that converts the count vector's innards into a frequency vector. The array representation of this output is also known as LibSVM format, which will be important later!

Next, a call to `fit_transform()` fits the vectorizer and transformer with each tweet. The model will use this fitted object to make predictions on supplied data. The model is fitted with the frequency vector and the sentiment ratings of each tweet.

The `test()` function takes a similarly structured set of data and asks the model to predict the sentiment rating based on its training and returns the accuracy of the test results. The trained models can be saved and reloaded using `save()` and `load()` and the options in `main.py`.

### Spark

The same processes are mimicked in `spark.py` using the Apache Spark library. The preprocessing of data here is the main difference; each tweet needs to be converted to LibSVM format, then to a `LabeledPoint` object, which then must be packaged in an RDD (Resiliant Distributed Dataset). We use `classifier.py` to compute the LibSVM array of each tweet and do the conversions. The data is then fed into the model in `train()` and tested in `test()` in the same way as in its sklearn counterpart.