import json
import os
from collections import defaultdict

import nltk
import requests

from nltk.corpus import stopwords
from wordcloud import WordCloud


def download(url, filename=None):
    """Downloads file from the selected url and stores to file
    - If file already exists, skips download
    - If filename not provided, tries to infer it from url
    """
    if filename is None:
        filename = url.split('/')[-1]
    filepath = os.path.abspath(filename)

    if os.path.isfile(filepath):
        print(f'{filepath} exists, skipping download')
    else:
        r = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(r.content)
        print(f'Fetched {url} to {filepath}')

    return filepath


def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data


def parse_conan(conan_data, languages=['EN']):
    hate_texts = []
    counter_texts = []
    for entry in conan_data['conan']:
        if any(entry['cn_id'].startswith(lang) for lang in languages):
            hate_texts.append(entry['hateSpeech'])
            counter_texts.append(entry['counterSpeech'])
    
    return hate_texts, counter_texts


def count_words(texts, stops=None):
    if stops is None:
        stops = set(stopwords.words('english') + ['.', ','])
    word_count = defaultdict(int)
    for text in texts:
        words = [word for word in nltk.word_tokenize(text) if word not in stops]
        for word in words:
            word_count[word] += 1
    return word_count
