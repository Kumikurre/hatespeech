from empath import Empath
import json
import nltk
import pysentiment2 as ps
import pickle

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from nltk.corpus import stopwords

def get_tweet_texts(filename):
    nltk.download('punkt')
    nltk.download('stopwords')

    # set of stopwords (could be expanded)
    en_stops = set(stopwords.words('english') + stopwords.words('italian') + stopwords.words('french') + ['.', ','])

    f = open(filename, "r")
    json_data = f.read()

    dataset = json.loads(json_data)

    # Form lists of tweets
    hate_texts = []
    counter_texts = []
    for tweet in dataset["conan"]:
        hate_speech = tweet["hateSpeech"]
        counter_speech = tweet["counterSpeech"]
        hate_texts.append(hate_speech)
        counter_texts.append(counter_speech)

    return hate_texts, counter_texts

def get_categories(text):
    categories = lexicon.analyze(text, normalize=True)
    return categories

def get_inquiries(text):
    tokens = hiv4.tokenize(text)
    scores = hiv4.get_score(tokens)
    return scores

def get_parsed(hate_tweets, counter_tweets):
    try:
        print("Loading data...")
        with open ('train_items', 'rb') as fp:
            train_items = pickle.load(fp)
        with open ('train_labels', 'rb') as fp:
            train_labels = pickle.load(fp)
    except:
        print("No existing data. Parsing training data")
        train_items = []
        train_labels = []
        print("parsing hate tweets...")
        for tweet in hate_tweets:
            categories = list(get_categories(tweet).values())
            scores = list(get_inquiries(tweet).values())
            categories_score_combined = categories + scores
            train_items.append(categories_score_combined)
            # hatespeech is label 1
            train_labels.append(1)

        print("parsing counter tweets...")
        for tweet in counter_tweets:
            categories = list(get_categories(tweet).values())
            scores = list(get_inquiries(tweet).values())
            categories_score_combined = categories + scores
            train_items.append(categories_score_combined)
            # counterspeech is label 0
            train_labels.append(0)

        print("writing to file")
        with open('train_items', 'wb') as fp:
            pickle.dump(train_items, fp)

        with open('train_labels', 'wb') as fp:
            pickle.dump(train_labels, fp)
    
    return train_items, train_labels

### Task 4
def teach(hate_tweets, counter_tweets):


    train_items, train_labels = get_parsed(hate_tweets, counter_tweets)

    model = tf.keras.Sequential([
        # each train_items item has 197 fields, so that is the number of neurons we need in the first layer of network
        tf.keras.layers.Input(198,),
        tf.keras.layers.Dense(50, activation='relu'), # Not sure what to do with this layer. Do we even need it
        tf.keras.layers.Dense(35),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])


    # Optimizer uses adaptive learning rates (adam optimizer)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    print("Starting to teach")
    model.fit(train_items, train_labels, epochs=10)
    model.summary()
    return model


def evaluate(model, hate_tweets, counter_tweets):
    evaluate_items = []
    evaluate_labels = []
    print("Parsing evaluation data")
    for tweet in hate_tweets:
        categories = list(get_categories(tweet).values())
        scores = list(get_inquiries(tweet).values())
        categories_score_combined = categories + scores
        evaluate_items.append(categories_score_combined)
        # hatespeech is label 1
        evaluate_labels.append(1)

    for tweet in counter_tweets:
        categories = list(get_categories(tweet).values())
        scores = list(get_inquiries(tweet).values())
        categories_score_combined = categories + scores
        evaluate_items.append(categories_score_combined)
        # counterspeech is label 0
        evaluate_labels.append(0)

    print("Evaluating the model")
    test_loss, test_acc = model.evaluate(evaluate_items, evaluate_labels, verbose=2)

    print("Accuracy of the heuristic against CONAN dataset: ", test_acc)

if __name__ == '__main__':
    import sys
    hiv4 = ps.HIV4()
    lexicon = Empath()
    filename = 'CONAN.json'
    DIVIDER_FOR_DATASET = 2

    hate_words, counter_words = get_tweet_texts(filename)

    data_split = int(len(hate_words)/DIVIDER_FOR_DATASET)
    print(f'Using {data_split} of {data_split * DIVIDER_FOR_DATASET} hate and counter tweets to teach.')

    # Use the first part of the dataset to teach the network, and second part to evaluate it
    model = teach(hate_words[:data_split], counter_words[:data_split])
    evaluate(model, hate_words[data_split:], counter_words[data_split:])

    # Save the model to disk
    model.save('./heuristic')