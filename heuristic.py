from empath import Empath
import json
import nltk
import pysentiment2 as ps

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


def heuristic(categories, scores):
    # TODO Here we should decide for each tweet whether it is hate speech or not
    print(categories)
    print(scores)
    return 


### Task 4
def task4(hate_tweets, counter_tweets):
    print("Starting task 4")
    for tweet in hate_tweets:
        categories = get_categories(tweet)
        scores = get_inquiries(tweet)
        heuristic(categories, scores)

    for tweet in counter_tweets:
        categories = get_categories(tweet)
        scores = get_inquiries(tweet)
        heuristic(categories, scores)



if __name__ == '__main__':
    import sys
    hiv4 = ps.HIV4()
    lexicon = Empath()
    filename = 'CONAN.json'
    hate_words, counter_words = get_tweet_texts(filename)
    task4(hate_words, counter_words)