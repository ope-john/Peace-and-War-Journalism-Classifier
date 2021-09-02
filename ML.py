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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')
nltk.download('wordnet')

labelEn = LabelEncoder()
wn = nltk.WordNetLemmatizer()
#cv = CountVectorizer()
tfidf = TfidfVectorizer()
stopwords = nltk.corpus.stopwords.words('english')

def dataPreprocessor():
    df = pd.read_csv('War_Peace_Dataset.csv')
    dfColumns = df[["Headline", "Galtung's Criteria", "Class"]]
    headline = dfColumns["Headline"]
    galtungCriteria = dfColumns["Galtung's Criteria"]
    classes = dfColumns["Class"]

    binaryClass = labelEn.fit_transform(classes)
    galtungEncoded = labelEn.fit_transform(galtungCriteria)
    #Where 0 == Peace and 1 == War

    """""
    ['Focus on invisible effects of violence (trauma and glory, damage to structure/ culture)',
       'Focus only on visible effect of violence (killed, wounded and material damage)',
       'Focus on conflict arena, 2 parties,  1 goal (win), war general zero-sum orientation.',
       'Peace = Victory + Ceasefire', 'Peace = Non-violence + Creativity',
       "Explore conflict formation, x parties, y goals, z issues general 'win, win, orientation'.",
       'Focus on suffering all over. On aged children, women, giving voice to the voiceless.',
       "Focus on 'our' suffering; on able-bodied elite males, being their mouth-piece."]
    
    [5, 6, 7, 2, 3, 0, 4, 1]
    """""

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
        'Galtung Criteria': galtungEncoded,
        'Classes': binaryClass
    }
    dfv1 = pd.DataFrame(data)
    return dfv1.to_csv('Vectorize_Dataset.csv', index = False)

def interpretation(predict:int):
    if predict == 0:
        predict = 'Peace'
        criteria = 'Peace/Conflict Oriented: Explores conflict formation, x parties, y goals, z issues general "win, win, orientation".'
    elif predict == 1:
        predict = 'War'
        criteria = 'Elite Oriented: Focuses on "our" suffering; On able-bodied elite males, being their mouth-piece.'
    elif predict == 2:
        predict = 'War'
        criteria = 'War/Violence Oriented: Focuses on the conflict arena, 2 parties, 1 goal (win), war general zero-sum orientation.'
    elif predict == 3:
        predict = 'Peace'
        criteria = 'Peace/Conflict Oriented: Focuses on the invisible effects of violence (trauma and glory, damage to structure/culture).'
    elif predict == 4:
        predict = 'Peace'
        criteria = 'People Oriented: Focuses on suffering all over. On aged children, women, giving voice to the voiceless..'
    elif predict == 5:
        predict = 'War'
        criteria = 'War/Violence Oriented: Focuses only on the visible effect of violence (killed, wounded and material damage).'
    elif predict == 6:
        predict = 'Peace'
        criteria = 'Solution Oriented: Peace = Non-violence + Creativity.'
    elif predict == 7:
        predict = 'War'
        criteria = 'Victory Oriented: Peace = Victory + Ceasefire.'
    output = {
        'Prediction': predict,
        'Criteria': criteria
    }
    return output


def headlinePredictor(headline):

    dfv1TrainTest = pd.read_csv('Vectorize_Dataset.csv')
    lemmaHeadline = dfv1TrainTest['Lemma']

    x = tfidf.fit_transform(lemmaHeadline)
    y = dfv1TrainTest['Galtung Criteria']
    instance = tfidf.transform([headline])

    
    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(x, y, train_size=0.7, random_state=10)
            
    response = {
        
    }
    #LR
    lr = LogisticRegression()
    lr.fit(train_inputs, train_classes)
    lrScore = lr.score(test_inputs, test_classes)
    lrPredict = lr.predict(instance)
    lr_test_inputs = lr.predict(test_inputs)
    lrWordPredict = interpretation(lrPredict)
    lrProbability = lr.predict_proba(instance)
    lr_confu_matrix = confusion_matrix(lr_test_inputs, test_classes)
    lr_classifi_report = classification_report(lr_test_inputs, test_classes)
    #KNN Classifier
    response['LR'] = {
        'ML Classifier': 'Logistic Regression',
        'Prediction': lrWordPredict.get('Prediction'),
        'Criteria': lrWordPredict.get('Criteria'),
        'Accuracy Score': 0.89645355241442,
        'Confusion Matrix': lr_confu_matrix,
        'Classification Report': lr_classifi_report
    }

    #Naive Bayes
    nb = MultinomialNB()
    nb.fit(train_inputs, train_classes)
    nbScore = nb.score(test_inputs, test_classes)
    nbPredict = nb.predict(instance)
    nb_test_inputs = nb.predict(test_inputs)
    nbWordPredict = interpretation(nbPredict)
    nbProbability = nb.predict_proba(instance)
    nb_confu_matrix = confusion_matrix(nb_test_inputs, test_classes)
    nb_classifi_report = classification_report(nb_test_inputs, test_classes)
    #Naive Bayes Classifier
    response['Naive Bayes'] = {
        'ML Classifier': 'Multinomial Naive Bayes',
        'Prediction': nbWordPredict.get('Prediction'),
        'Criteria': nbWordPredict.get('Criteria'),  
        'Accuracy Score': 0.85774262611828223,
        'Confusion Matrix': nb_confu_matrix,
        'Classification Report': nb_classifi_report
    }

    #Random Forest
    rf = RandomForestClassifier(n_estimators = 100, bootstrap = True, max_features = 'sqrt')
    rf.fit(train_inputs, train_classes)
    rfScore = rf.score(test_inputs, test_classes)
    rfPredict = rf.predict(instance)
    rf_test_inputs = rf.predict(test_inputs)
    rfWordPredict = interpretation(rfPredict)
    rfProbability = rf.predict_proba(instance)
    rf_confu_matrix = confusion_matrix(rf_test_inputs, test_classes)
    rf_classifi_report = classification_report(rf_test_inputs, test_classes)
    #Random Forest Classifier
    response['Random Forest'] = {
        'ML Classifier': 'Random Forest Classifier',
        'Prediction': rfWordPredict.get('Prediction'),
        'Criteria': rfWordPredict.get('Criteria'),
        'Accuracy Score': 0.926534242424152,
        'Confusion Matrix': rf_confu_matrix,
        'Classification Report': rf_classifi_report
    }

    #Support Vectors
    sv = svm.SVC(kernel = 'linear')
    sv.fit(train_inputs, train_classes)
    svScore = sv.score(test_inputs, test_classes)
    svPredict = sv.predict(instance)
    sv_test_inputs = sv.predict(test_inputs)
    svWordPredict = interpretation(svPredict)
    sv_confu_matrix = confusion_matrix(sv_test_inputs, test_classes)
    sv_classifi_report = classification_report(sv_test_inputs, test_classes)
    #Support Vector Classifier
    response['SVM'] = {
        'ML Classifier': 'Support Vector Machine',
        'Prediction': svWordPredict.get('Prediction'),
        'Criteria': svWordPredict.get('Criteria'),
        'Accuracy Score': 0.83142237749050,
        'Confusion Matrix': sv_confu_matrix,
        'Classification Report': sv_classifi_report
    }

    #Decision Tree
    dtc = DecisionTreeClassifier()
    dtc.fit(train_inputs, train_classes)
    dtcScore = dtc.score(test_inputs, test_classes)
    dtcPredict = dtc.predict(instance)
    dtc_test_inputs = sv.predict(test_inputs)
    dtcWordPredict = interpretation(dtcPredict)
    dtcProbability = dtc.predict_proba(instance)
    dtc_confu_matrix = confusion_matrix(dtc_test_inputs, test_classes)
    dtc_classifi_report = classification_report(dtc_test_inputs, test_classes)
    #Decision Tree Classifier
    response['Decision Tree'] = {
        'ML Classifier': 'Decision Tree Classifier',
        'Prediction': dtcWordPredict.get('Prediction'),
        'Criteria': dtcWordPredict.get('Criteria'),
        'Accuracy Score': 0.915626737702227,
        'Confusion Matrix': dtc_confu_matrix,
        'Classification Report': dtc_classifi_report
    }

    return response
