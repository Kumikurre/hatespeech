import json
import nltk
import matplotlib

from nltk.corpus import stopwords


def get_texts(filename):
    nltk.download('punkt')
    nltk.download('stopwords')

    # set of stopwords (could be expanded)
    en_stops = set(stopwords.words('english') + stopwords.words('italian') + stopwords.words('french') + ['.', ','])

    f = open(filename, "r")
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

    return hate_texts, counter_texts

def task1(hate_texts, counter_texts):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt 

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
def task2(hate_texts, counter_texts):
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


### Task 3
def task3(hate_texts, counter_texts):
    import pysentiment2 as ps
    hiv4 = ps.HIV4()
    hate_tokens = hiv4.tokenize(hate_texts)
    hate_score = hiv4.get_score(hate_tokens)
    print(f'hate_score: {hate_score}')

    counter_tokens = hiv4.tokenize(counter_texts)
    counter_score = hiv4.get_score(counter_tokens)
    print(f'counter_score: {counter_score}')


all_tasks = [task1, task2, task3]

if __name__ == '__main__':
    import sys
    filename = 'CONAN.json'
    hate_words, counter_words = get_texts(filename)
    run_all = len(sys.argv) == 1 or 'all' in sys.argv

    for i, task in enumerate(all_tasks):
        if run_all or str(i + 1) in sys.argv:
            print(f'Running task #{i + 1}')
            task(hate_words, counter_words)
