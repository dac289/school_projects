"""
Using natural language processing to build a program
that can use language in the reviews to predict the
rating given by user
"""

# Regualar imports
import pandas as pd
import numpy as np
import string

# Graph and image imports
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Imports for Natural Language Processing
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer

# Imports from sklearn
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

import re

def get_data(filename):
    """Create dataframe and separating ratings of 4,5 from 1,2,3"""
    data = pd.read_csv(filename)
    data.dropna(inplace=True)
    naive_bayes_col = []
    [naive_bayes_col.append(1) if x==4|5 else naive_bayes_col.append(0) for x in data.Rating]
    data['naive_bayes'] = naive_bayes_col
    less_3 = data[data['naive_bayes']==0]
    more_4 = data[data['naive_bayes']==1]
    return less_3, more_4, data

def _stopword_():
    """initialize stopwords from nltk module"""
    stopword = stopwords.words('english')
    return stopword

def _wordcloud_(dataset, stopword, title):
    """Create a word cloud of the words used in reviews"""
    punctuation = string.punctuation
    strings = [str(x).lower().split() for x in dataset['Reviews']]
    word_dict = {}
    for x in strings:
        for y in x:
            y = y.strip(punctuation + string.whitespace)
            if y in stopword:
                pass
            else:
                word_dict[y] = word_dict.get(y, 0) +1
    df = pd.DataFrame.from_dict(word_dict.items())
    print(df.sort_values(by=[1], ascending=False)[:10])
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(word_dict)
    plt.figure(figsize=(9,6))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title, fontsize=24)
    plt.show()


def add_new_words(stopword):
    """add the top 5 words used in the 1,2,3 rating reviews"""
    new_words = ['phone', 'work','screen','one','good']
    for x in new_words:
        stopword.append(x)
    return stopword

def subsetter(data, column, stopwords):
    """Use CountVectorizer to actually measure the prediction ability using Naive Bayes"""
    texttotokens = []
    flatlist = []
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    for i in data[column]:
        i = i
        texttotokens.append(i)
    for y in i:
        flatlist.append(y)

    frequency_dist = nltk.FreqDist([word for word in tokenizer.tokenize(str(flatlist)) \
                                    if word not in stopwords])
    top50n = sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:50]

    X_train, X_test, y_train, y_test = train_test_split(data['Reviews'].values,data['naive_bayes'].values,
                                                        test_size=0.2,random_state=1)
    REGEX = re.compile(r",\s*")
    tokenize = [tok.strip().lower() for tok in REGEX.split(str(stopwords))]
    cv = CountVectorizer(lowercase=True, stop_words='english', binary=True)

    X_train_cv = cv.fit_transform(X_train)
    naive_bayes = BernoulliNB()
    naive_bayes.fit(X_train_cv, y_train)
    X_test_cv = cv.transform(X_test)
    predictions = naive_bayes.predict(X_test_cv)

    print('Accuracy score: ', accuracy_score(y_test, predictions))
    print(sum(y_test == predictions) / len(predictions), "/n")

    print('Precision score: ', precision_score(y_test, predictions))
    print(sum(y_test[predictions == 1] == 1) / len(y_test[predictions == 1]), "/n")

    print('Recall score: ', recall_score(y_test, predictions))
    print(sum(predictions[y_test == 1] == 1) / len(predictions[y_test == 1] == 1), "/n")


if __name__ == "__main__":
    filename = "Amazon_Unlocked_Mobile.csv"
    less_3, more_4, data = get_data(filename)
    stopword = _stopword_()
    _wordcloud_(less_3, stopword,"3 or less Rating")
    _wordcloud_(more_4,stopword, "4 or more Rating")
    stopword = add_new_words(stopword)
    _wordcloud_(less_3, stopword,"3 or less Rating")
    _wordcloud_(more_4, stopword, "4 or more Rating")
    subsetter(data, 'Reviews', stopword)