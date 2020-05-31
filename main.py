import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), "lxml")
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), "lxml")
negative_reviews = negative_reviews.findAll('review_text')

review_labels = ["Bad review", "Good Review"]

np.random.shuffle(positive_reviews)

positive_reviews = positive_reviews[:len(negative_reviews)]


def get_tokens(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    return tokens


word_to_int = {}
index = 0

positive_tokens = []
negative_tokens = []

for review in positive_reviews:
    tokens = get_tokens(review.text)
    positive_tokens.append(tokens)
    for token in tokens:
        if token not in word_to_int:
            word_to_int[token] = index
            index += 1

for review in negative_reviews:
    tokens = get_tokens(review.text)
    negative_tokens.append(tokens)
    for token in tokens:
        if token not in word_to_int:
            word_to_int[token] = index
            index += 1

N = len(positive_tokens) + len(negative_tokens)
data = np.zeros((N, len(word_to_int) + 1))
index = 0


def token_to_vector(tokens, label):
    x = np.zeros(len(word_to_int) + 1)
    for t in tokens:
        i = word_to_int[t]
        x[i] += 1
    x = x / x.sum()
    x[-1] = label
    return x


for token in positive_tokens:
    xy = token_to_vector(token, 1)
    data[index, :] = xy
    index += 1

for token in negative_tokens:
    xy = token_to_vector(token, 0)
    data[index, :] = xy
    index += 1

np.random.shuffle(data)

inputs_data = data[:, :-1]
output_data = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(inputs_data, output_data, test_size=0.33)

model = LogisticRegression()

try:
    model = joblib.load("model.simple")
    print("Model Loaded..")
except:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(score)
    joblib.dump(model, "model.simple")


def input_token_to_vector(token):
    x = np.zeros(len(word_to_int))
    for t in token:
        i = word_to_int[t]
        x[i] += 1
    x = x / x.sum()
    return x


def get_inputs():
    user_input = input("Enter Review(quite for exit) : ")
    return user_input;


def input_vector(user_input):
    xy = get_tokens(user_input)
    x = input_token_to_vector(xy)
    return x;


while True:
    get_user_input = get_inputs()

    if get_user_input == "quite":
        break

    vector_input = input_vector(get_user_input)
    prediction = model.predict([vector_input])
    print(review_labels[int(prediction)])
