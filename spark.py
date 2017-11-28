from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils

def tokenize():
    analyzer = EnglishAnalyzer()
    tokens = analyzer.token

def train():
    model = NaiveBayes.train()