from pymongo import MongoClient, errors
from settings import ATLAS_URI

try:
    mongo = MongoClient(ATLAS_URI)
except errors.InvalidURI:
    print('Failed to connect to', ATLAS_URI)
    exit(1)

database = mongo.tweetstream.harvey

def gettweets(conditions={}):
    return database.tweets.find(conditions)