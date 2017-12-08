""" Models needed for streaming tweets """

import classifier, json
from settings import STATUS_THRESHOLD
from threading import Thread
from tweepy.streaming import StreamListener

class Client():
    """ Controls the stream """

    def __init__(self, clf, cv, tf, workers=1, max_tweets=100):
        self.max_tweets = max_tweets
        self.raw_tweets = []
        self.thread_pool = []
        self.num_workers = workers

        self.clf = clf
        self.cv = cv
        self.tf = tf

        self.pos = 0
        self.num = 0
        self.totconf = 0

        self.running = False
        self.start_workers()

    def post(self, tweet_data):
        self.raw_tweets.append(tweet_data)
        data = json.loads(tweet_data)
        if not 'text' in data:
            return
        text = str(data['text'])
        # self.predict(text)

    def done(self):
        return self.num >= self.max_tweets

    def show(self):
        print(self.num, 'tweets processed')
        print(self.pos, 'with positive sentiment [', 100*self.pos/self.num, ']')
        print(self.num-self.pos, 'with negative [', 100*(self.num-self.pos)/self.num, ']')
        print('Avg confidence:', (100*self.totconf)/self.num)

    def predict(self, text):
        pred = classifier.predict(self.clf, [text], self.cv, self.tf)[0]
        posval = 1 # this motherfucker gets confusing
        negval = 0
        ispos = pred[0] == posval
        conf = pred[1][posval] if ispos else pred[1][negval]

        if conf > 0.6:
            self.pos += 1 if ispos else 0
            self.num += 1
            self.totconf += conf

            if self.num % STATUS_THRESHOLD == 0:
                print('{} tweets: [ {}+, {}- ]'.format(self.num, self.pos, self.num-self.pos))
                print('last: [ {} ] [ {}, {}% ]'.format(text, 'positive' if ispos else 'negative', int(conf*100)))

    def start_workers(self):
        self.running = True
        for w in range(self.num_workers):
            print('starting')
            self.thread_pool.append(Worker(
                client=self,
                name=str(w),
                clf=self.clf,
                cv=self.cv,
                tf=self.tf
            ))
            self.thread_pool[w].start()

    def finish(self):
        """ Brings back workers """
        self.running = False
        for worker in self.thread_pool:
            print('joining worker')
            worker.join()

        print('Collected', len(self.raw_tweets), 'tweets!')
        self.show()

class Listener(StreamListener):
    """ Listens to the stream """

    def __init__(self, client):
        self.client = client

    def on_data(self, data):
        """ Called when a new tweet is delivered """
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

    def __init__(self, name, client, clf, cv, tf):
        Thread.__init__(self)
        self.name = name
        self.client = client

    def run(self):
        """ This is the worker's task """
        while self.client.running:
            if not len(self.client.raw_tweets) > 0:
                continue

            raw = self.client.raw_tweets.pop()
            data = json.loads(raw)

            if 'retweeted_status' in data:
                continue

            if not 'text' in data:
                continue

            text = str(data['text'])
            self.client.predict(text)