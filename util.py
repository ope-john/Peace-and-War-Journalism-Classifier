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

def galtungCriteria(criteria: str):
    if criteria == "Focus on invisible effects of violence (trauma and glory, damage to structure/ culture)":
        value = 3
    elif criteria == "Focus on conflict arena, 2 parties,  1 goal (win), war general zero-sum orientation.":
        value = 2
    elif criteria == "Focus on suffering all over. On aged children, women, giving voice to the voiceless.":
        value = 4
    elif criteria ==  "Focus only on visible effect of violence (killed, wounded and material damage)":
        value = 5
    elif criteria == "Peace = Non-violence + Creativity":
        value = 6
    elif criteria == "Peace = Victory + Ceasefire":
        value = 7
    elif criteria == "Explore conflict formation, x parties, y goals, z issues general 'win, win, orientation'":
        value = 0
    elif criteria == "Focus on 'our' suffering; on able-bodied elite males, being their mouth-piece.":
        value == 1
    else:
        pass
    return value
        