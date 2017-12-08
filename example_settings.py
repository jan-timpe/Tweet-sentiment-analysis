# fill out with your credentials then rename this file 'settings.py'

# Database

ATLAS_URI = ''

# Learning

TRAIN_FILENAME = './data/train.csv'
TEST_FILENAME = './data/test.csv'
SKLEARN_PATH = './data/models/sklearn'
SPARK_PATH = './data/models/spark'
SPARK_APPNAME = 'TwitterSentimentAnalysis'

# Stream

NUM_THREADS = 4
MAX_TWEETS = 100
STATUS_THRESHOLD = 1
TERMS =[
    'soccer'
]

TWITTER_CONSUMER_KEY = ''
TWITTER_CONSUMER_SECRET = ''
TWITTER_ACCESS_TOKEN = ''
TWITTER_ACCESS_TOKEN_SECRET = ''