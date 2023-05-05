import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

## NEW NEW NEW NEW ###

def lemmatize_word(word, pos='v'):
    
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, pos=pos)
    return lemma

def pos_tag(sentence):
    
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags

# Analyse syntaxique (Parsing)