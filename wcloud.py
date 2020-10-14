import json
import nltk
import matplotlib

from nltk.corpus import stopwords

from wordcloud import WordCloud
import matplotlib.pyplot as plt 

nltk.download('punkt')
nltk.download('stopwords')

# set of stopwords (could be expanded)
en_stops = set(stopwords.words('english') + stopwords.words('italian') + stopwords.words('french') + ['.', ','])

f = open("CONAN.json", "r")
json_data = f.read()

dataset = json.loads(json_data)

# form a dict with the word as a key and its frequency as the value
hatespeech_count = {}
counterspeech_count = {}
for tweet in dataset["conan"]:
    hate_speech = tweet["hateSpeech"]
    counter_speech = tweet["counterSpeech"]
    hatewords = nltk.word_tokenize(hate_speech)
    hatewords = [word for word in hatewords if word not in en_stops]
    counterwords = nltk.word_tokenize(counter_speech)
    counterwords = [word for word in counterwords if word not in en_stops]
    # go through both hate speech and counter speech
    for word in hatewords:
        hatespeech_count.setdefault(word, 0)
        hatespeech_count[word] += 1
    for word in counterwords:
        counterspeech_count.setdefault(word, 0)
        counterspeech_count[word] += 1

# wordcloud takes a string as input, so go over the dict and concatenate a string
hate_texts = ""
hate_counts = []
for key, value in hatespeech_count.items():
    hate_texts = hate_texts + key + ' '
    hate_counts.append(value)

counter_texts = ""
counter_counts = []
for key, value in counterspeech_count.items():
    counter_texts = counter_texts + key + ' '
    counter_counts.append(value)

# form the wordcloud and print it out
wordcloud = WordCloud().generate(hate_texts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("hatespeech.png")
# plt.show()

counter_wordcloud = WordCloud().generate(counter_texts)
plt.imshow(counter_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("counterspeech.png")
# plt.show()



#### Task 2
from empath import Empath
lexicon = Empath()
hate_analysis = lexicon.analyze(hate_texts)
counter_analysis = lexicon.analyze(counter_texts)


for key, value in list(hate_analysis.items()):
    if value == 0.0:
        hate_analysis.pop(key)
for key, value in list(counter_analysis.items()):
    if value == 0.0:
        counter_analysis.pop(key)


print(hate_analysis)
print(counter_analysis)
