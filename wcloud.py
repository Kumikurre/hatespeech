import json
import nltk
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

en_stops = set(stopwords.words('english') + stopwords.words('italian') + stopwords.words('french') + ['.', ','])

f = open("CONAN.json", "r")
json_data = f.read()

dataset = json.loads(json_data)

wordcount = {}
for tweet in dataset["conan"]:
    hate_speech = tweet["hateSpeech"]
    words = nltk.word_tokenize(hate_speech)
    words = [word for word in words if word not in en_stops]
    for word in words:
        wordcount.setdefault(word, 0)
        wordcount[word] += 1

print(wordcount)


texts = []
counts = []

for key, value in wordcount.items():
    texts.append(key)
    counts.append(value)


test1 = ColumnDataSource({'names':texts,'weights':counts})
wordcloud = WordCloud2(source=test1,wordCol="names",sizeCol="weights",color=['pink','blue','green'])
show(wordcloud)
