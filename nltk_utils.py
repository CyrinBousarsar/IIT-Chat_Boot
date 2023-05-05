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

#
def remove_stop_words(sentence):
    stop_words = set(stopwords.words('french')) # ou une autre langue
    words = word_tokenize(sentence)
    clean_words = [word for word in words if word.lower() not in stop_words]
    clean_sentence = " ".join(clean_words)
    return clean_sentence
 
def calculate_word_frequency(sentence):
    # Tokeniser le sentencee en mots
    words = nltk.word_tokenize(sentence.lower())

    # Calculer la fréquence des mots
    fdist = FreqDist(words)

    # Retourner la distribution de fréquence des mots
    return fdist

def find_collocations(text):
    # Tokeniser le texte en mots
    words = nltk.word_tokenize(text.lower())

    # Créer un objet de texte NLTK
    text_obj = nltk.Text(words)

    # Trouver les collocations
    collocations = text_obj.collocations()

    # Retourner les collocations
    return collocations
