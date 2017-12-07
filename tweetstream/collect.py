""" Runs a tweet streamer """

from .models import Client, Listener
from settings import MAX_TWEETS, NUM_THREADS, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET
from tweepy import OAuthHandler, Stream, streaming

def stream(terms):
    client = Client(workers=NUM_THREADS, max_tweets=MAX_TWEETS)
    listener = Listener(client=client)
    auth = OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)

    tweet_stream = Stream(auth=auth, listener=listener)
    tweet_stream.filter(track=terms)
    tweet_stream.disconnect()

    return client.raw_tweets