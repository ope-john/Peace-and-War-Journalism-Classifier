import numpy as np
import pandas as pd
import streamlit as st
from ML import headlinePredictor, dataPreprocessor
from util import headlinePreprocessor, galtungCriteria


st.title('Peace and War Journalism Classifier')

navi = st.sidebar.selectbox(
    "Navigation",
    ['Home', 'Prediction', 'Documentation'])
st.write(navi)

if navi == 'Home':
    st.header('Dataset')
    df = pd.read_csv("War_Peace.csv")
    st.write(df.head(10))

if navi == 'Prediction':
    Headline = st.text_area('News headline goes here')
    galtungCriteriaText = st.selectbox(
        "Galtung's Criteria",
        ["Select a Criteria associated with the headline",
        "Focus on invisible effects of violence (trauma and glory, damage to structure/ culture)",
        "Focus on conflict arena, 2 parties,  1 goal (win), war general zero-sum orientation.",
        "Focus on suffering all over. On aged children, women, giving voice to the voiceless.",
        "Focus only on visible effect of violence (killed, wounded and material damage)",
        "Focus on 'our' suffering; on able-bodied elite males, being their mouth-piece.",
        "Peace = Non-violence + Creativity", "Peace = Victory + Ceasefire",
        "Explore conflict formation, x parties, y goals, z issues general 'win, win, orientation'."
        ]
    )
    if st.button('Make Prediction'):
        if len(Headline) < 15:
            st.error("Your News Headline is too short...")
        elif galtungCriteriaText == 'Select a Criteria associated with the headline':
            st.error('Please select a criteria')
        else:
            lowercasedHeadline = Headline.lower()
            prediction = headlinePreprocessor(lowercasedHeadline)
            st.write(prediction)

if navi == 'Documentation':
    subNav = st.sidebar.selectbox(
    "Docs",
    ['Introduction', 'Pipeline', 'Tools'])
    if subNav == 'Introduction':
        st.write('This is an implementation of a natural language processing application using five machine learning algorithms () in Predicting if a news headline is Peace or War related. The application is built on python. The machine learning models were built with the Scikit-Learn Machine Learning Library and the visualization is built with Streamlit.')
    if subNav == 'Pipeline':
        st.header('Natural Language Processing')
        st.subheader('Raw text')
        st.write('Involves gathering raw text data of news headlines from various global news media e.g BBC, Aljezeera, CNN...')
        st.subheader('Removing Punctuations')
        st.write('Using the string module from the Natural Language Tool Kit Library (NLTK), All punctuations will be removed from the news headlines')
        st.subheader('Tokenization')
        st.write('Using the regular expression module from python, news headlines will be divided into a list of comma separated words. These words are called tokens')
        st.code('import re')
        st.subheader('Removing Stopwords')
        st.write('Using the stopword function from the Natural Language Tool Kit Library (NLTK), Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence')
        st.code('import nltk')
        st.code('stopwords = nltk.corpus.stopwords.words("english")')

        st.subheader('Lemmatization')
        st.write('Using the lemmatization function from the Natural Language Tool Kit Library (NLTK), lemmatization is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the words lemma, or dictionary form.')
        st.code('import nltk')
        st.code('wn = nltk.WordNetLemmatizer()')
        st.subheader('Vectorization')
        st.write('TF-IDF are word frequency scores that try to highlight words that are more interesting, e.g. frequent words but not across the news headlines. The TfidfVectorizer will tokenize words, learn the vocabulary and inverse words frequency weightings, then encode.')
        st.code('from sklearn.feature_extraction.text import TfidfVectorizer')
        st.code('tfidf  = TfidfVectorizer()')
        st.subheader('Machine Learning')
        st.write('In this process the vectorized news headlines will be split into a training and testing set on 70% to 30% ratio with the train_test_split function from sklearn.model_selection module')
        st.code('from sklearn.model_selection import train_test_split')
        st.code('(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(vectorizeHeadlines, vectorizedClasses, train_size=0.7, random_state=10)')
        st.subheader('Training')
        st.code('MachineLearningAlgorithm.fit(train_inputs, train_classes)')
        st.subheader('Accuracy')
        st.code('MachineLearningAlgorithm.score(test_inputs, test_classes)')
    if subNav == 'Tools':
        st.subheader('Visualization')
        st.write('Stramlit')
        st.subheader('Natural Language Processing')
        st.write('Natural Language Toolkit')
        st.subheader('Machine Learning')
        st.write('Pandas')
        st.write('Numpy')
        st.write('Scikit-Learn')
        
        

