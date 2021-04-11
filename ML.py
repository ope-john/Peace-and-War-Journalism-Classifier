import nltk
import string
import pickle
import re
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn import svm
from sklearn.metrics import accuracy_score

labelEn = LabelEncoder()
wn = nltk.WordNetLemmatizer()
#cv = CountVectorizer()
tfidf  = TfidfVectorizer()
stopwords = nltk.corpus.stopwords.words('english')

def dataPreprocessor():
    df = pd.read_csv('War_Peace.csv')
    dfColumns = df[['Headline', 'Class']]
    headline = dfColumns['Headline']
    classes = dfColumns['Class']

    binaryClass = labelEn.fit_transform(classes)
    #Where 0 == Peace and 1 == War

    #Punctuations
    def dfRemovePunc(headline):
        newsWithoutPunc = "".join([charac for charac in headline if charac not in string.punctuation])
        return newsWithoutPunc
    df['Headline Without Punc'] = df['Headline'].apply(lambda headline: dfRemovePunc(headline))
    #Tokenization
    def tokenize(headline):
        tokens = re.split('\W+', headline)
        return tokens
    df['Tokenized Headline'] = df['Headline Without Punc'].apply(lambda words: tokenize(words.lower()))
    #Removing Stopwords
    def removeStopwords(tokenizedHeadline):
        cleanedHeadlines = [word for word in tokenizedHeadline if word not in stopwords]
        return cleanedHeadlines
    df['Headline Without Stop Words'] = df['Tokenized Headline'].apply(lambda headline: removeStopwords(headline))
    #Lemmatization
    def lemmatizer(cleanedHeadlines):
        lemmatizedWords =  [wn.lemmatize(word) for word in cleanedHeadlines]
        return lemmatizedWords
    df['Lemmatized Words'] = df['Headline Without Stop Words'].apply(lambda word: lemmatizer(word))
    #Vectorization
    data = {
        'Lemma': df['Lemmatized Words'],
        'Classes': binaryClass
    }
    dfv1 = pd.DataFrame(data)
    return dfv1.to_csv('Vectorize Dataset.csv', index = False)

def interpretation(predict:int):
    if predict == 0:
        predict = 'Peace'
    else:
        predict = 'War'
    return predict


def headlinePredictor(headline):
    dfv1TrainTest = pd.read_csv('Vectorize Dataset.csv')
    vectorizeHeadline = dfv1TrainTest['Lemma']

    x = tfidf.fit_transform(vectorizeHeadline)
    vectorizedPrediction = tfidf.transform([headline])

    encodedClasses = dfv1TrainTest['Classes']
    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(x, encodedClasses, train_size=0.7, random_state=10)

    response = {
        
    }
    #LR
    lr = LogisticRegression()
    lr.fit(train_inputs, train_classes)
    lrScore = lr.score(test_inputs, test_classes)
    lrPredict = lr.predict(vectorizedPrediction)
    lrWordPredict = interpretation(lrPredict)
    lrProbability = lr.predict_proba(vectorizedPrediction)
    #KNN Classifier
    response['LR'] = {
        'ML Classifier': 'Logistic Regression',
        'Accuracy': lrScore,
        'Prediction': lrWordPredict,
    }

    #Naive Bayes
    nb = MultinomialNB()
    nb.fit(train_inputs, train_classes)
    nbScore = nb.score(test_inputs, test_classes)
    nbPredict = nb.predict(vectorizedPrediction)
    nbWordPredict = interpretation(nbPredict)
    nbProbability = nb.predict_proba(vectorizedPrediction)
    #Naive Bayes Classifier
    response['Naive Bayes'] = {
        'ML Classifier': 'Multinomial Naive Bayes',
        'Accuracy': nbScore,
        'Prediction': nbWordPredict,
    }

    #Random Forest
    rf = RandomForestClassifier(n_estimators = 100, bootstrap = True, max_features = 'sqrt')
    rf.fit(train_inputs, train_classes)
    rfScore = rf.score(test_inputs, test_classes)
    rfPredict = rf.predict(vectorizedPrediction)
    rfWordPredict = interpretation(rfPredict)
    rfProbability = rf.predict_proba(vectorizedPrediction)
    #Random Forest Classifier
    response['Random Forest'] = {
        'ML Classifier': 'Random Forest Classifier',
        'Accuracy': rfScore,
        'Prediction': rfWordPredict,  
    }

    #Support Vectors
    sv = svm.SVC(kernel = 'linear')
    sv.fit(train_inputs, train_classes)
    svScore = sv.score(test_inputs, test_classes)
    svPredict = sv.predict(vectorizedPrediction)
    svWordPredict = interpretation(svPredict)
    #Random Forest Classifier
    response['SVM'] = {
        'ML Classifier': 'Support Vector Machine',
        'Accuracy': svScore,
        'Prediction': svWordPredict,  
    }

    #Decision Tree
    dtc = DecisionTreeClassifier()
    dtc.fit(train_inputs, train_classes)
    dtcScore = dtc.score(test_inputs, test_classes)
    dtcPredict = dtc.predict(vectorizedPrediction)
    dtcWordPredict = interpretation(dtcPredict)
    dtcProbability = dtc.predict_proba(vectorizedPrediction)
    #Random Forest Classifier
    response['Decision Tree'] = {
        'ML Classifier': 'Decision Tree Classifier',
        'Accuracy': dtcScore,
        'Prediction': dtcWordPredict,  
    }


    
    return response
