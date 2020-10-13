import json
import nltk
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

# set of stopwords (could be expanded)
en_stops = set(stopwords.words('english') + stopwords.words('italian') + stopwords.words('french') + ['.', ','])

f = open("CONAN.json", "r")
json_data = f.read()

dataset = json.loads(json_data)

# form a dict with the word as a key and its frequency as the value
wordcount = {}
for tweet in dataset["conan"]:
    hate_speech = tweet["hateSpeech"]
    words = nltk.word_tokenize(hate_speech)
    words = [word for word in words if word not in en_stops]
    for word in words:
        wordcount.setdefault(word, 0)
        wordcount[word] += 1

# wordcloud takes a string as input, so go over the dict and concatenate a string
texts = ""
counts = []
for key, value in wordcount.items():
    texts = texts + key + ' '
    counts.append(value)

# form the wordcloud and print it out
wordcloud = WordCloud().generate(texts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
