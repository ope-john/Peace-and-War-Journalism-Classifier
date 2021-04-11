import pandas as pd
import numpy as np
import nltk
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from ML import headlinePredictor

def headlinePreprocessor(headline):
    #Remove Punctuations
    headlineWithoutPunctuations = "".join([charac for charac in headline if charac not in string.punctuation])
    #Tokenization
    token = re.split('\W+', headlineWithoutPunctuations)
    #Removing Stop Words
    stopwords = nltk.corpus.stopwords.words('english')
    headlineWithoutStopwords = [word for word in token if word not in stopwords]
    #Lemmatization
    wn = nltk.WordNetLemmatizer()
    lemmatizedHeadline = [wn.lemmatize(word) for word in headlineWithoutStopwords]
    s = str(lemmatizedHeadline)
    predicted = headlinePredictor(s)
    return predicted

