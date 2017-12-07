""" Models needed for streaming tweets """

import json
from threading import Thread
from tweepy.streaming import StreamListener

class Client():
    """ Controls the stream """

    def __init__(self, workers=1, max_tweets=100):
        self.max_tweets = max_tweets
        self.raw_tweets = []
        self.thread_pool = []
        self.num_workers = workers
        self.start_workers()

    def start_workers(self):
        for w in range(self.num_workers):
            self.thread_pool.append(Worker(client=self, name=str(w)))
            self.thread_pool[w].start()

    def post(self, tweet_data):
        self.raw_tweets.append(tweet_data)

    def done(self):
        return len(self.raw_tweets) >= self.max_tweets

    def finish(self):
        """ Brings back workers """
        for worker in self.thread_pool:
            print('joining worker')
            worker.join()

        print('Collected', len(self.raw_tweets), 'tweets!')

class Listener(StreamListener):
    """ Listens to the stream """

    def __init__(self, client):
        self.client = client

    def on_data(self, data):
        """ Called when a new tweet is delivered """
        print(data)
        self.client.post(data)

        if self.client.done():
            self.client.finish()
            return False

        return True

    def on_error(self, status):
        """ Something went wrong, what's the status? """
        print(status)
        self.client.finish()
        return False


class Worker(Thread):
    """ Thread to post to db """

    def __init__(self, name, client):
        Thread.__init__(self)
        self.name = name
        self.client = client

    def run(self):
        """ This is the worker's task """
        while len(self.client.raw_tweets) > 0:
            raw = self.client.raw_tweets.pop()
            data = json.loads(raw)
            print(data)
