# -*- coding: utf-8 -*-


import re
from string import punctuation

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

for dependency in (
    "stopwords",
    "wordnet",
    "omw-1.4",
):
    nltk.download(dependency)

dataset = pd.read_csv('data/01-advance.csv')
dataset.head()

dataset.shape

dataset.isnull().sum()

#Selecting Required Columns:
dataset = dataset[['reviews','stars']]

#Review Ratings Distribution:
data = dataset['stars'].value_counts()

sns.barplot(x=data.index, y=data.values)

text = ""
for i in dataset.reviews:
    text += i
print(text)

stop_words = set(stopwords.words("english"))
print(stop_words)


from wordcloud import WordCloud

wc = WordCloud(background_color="black", max_words=300, stopwords=stop_words, max_font_size=50, random_state=42)

"""Display the wordcloud"""

import matplotlib.pyplot as plt

# Generate and display the word cloud

plt.figure(figsize= (15,15)) # Figure initialization
wc.generate(text) # "Calculation" from the wordcloud
plt.imshow(wc) # Display
plt.show()


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean up the text, with the possibility of removing stop_words and lemmatizing the word
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers

    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])

    # Optional, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optional, shorten the words to their root
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # Return a list of words
    return text

dataset['cleaned_review'] = dataset['reviews'].apply(text_cleaning)

train_data=dataset['cleaned_review']
y_target=dataset['stars'].map({1:'Unhappy',2:'Unhappy',3:'Happy',4:'Happy',5:'Happy'})

X_train, X_test, y_train, y_test = train_test_split(train_data, y_target, test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train).todense()
X_test = vectorizer.transform(X_test).todense()

clf = GradientBoostingClassifier(
    n_estimators=100, random_state=44
).fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

print("Accuracy Train: {}".format(accuracy_score(y_test,y_pred )))

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.show()

confusion_matrix = pd.crosstab(
    y_test, y_pred, rownames=["Real Class"], colnames=["Predicted Class"]
)
confusion_matrix



import joblib

joblib.dump(clf, "data/sentiment_model_pipeline.pkl")
