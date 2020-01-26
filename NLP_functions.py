
# coding: utf-8

# In[19]:

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

#NLP
import re
import string
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing import PorterStemmer
import nltk
from nltk.util import ngrams as nltkgrams
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import csv
from nltk.util import ngrams

from itertools import islice
from scipy import io
from scipy.sparse import csr_matrix, hstack
from ast import literal_eval

#TM
from gensim import models
from gensim.models import tfidfmodel
from gensim.corpora import Dictionary 
import pyLDAvis.gensim as gensimvis
import pyLDAvis


# In[23]:

def normalized_target(data):
    race_max = data.groupby('raceId')['positionOfficial'].transform('max')
    race_min = data.groupby('raceId')['positionOfficial'].transform('min')
    normalized = (data['positionOfficial'] - race_min)/(race_max-race_min)
    normalTarget =  normalized.apply(lambda x: -0.5 + x)
    return normalTarget

def winner_coding(data):
    data['winner'] = np.where(data['positionOfficial'] <=3, 1, 0)
    return data

def tokenize(text):
    """Remove punctuation and build tokens from text
    Args: Text as a single string
    Output: List of tokens.
    """
    #lowers = text.lower()
    #remove the punctuation(Satzzeichen) using the character deletion step of translate
    no_punctuation = text.translate((str.maketrans('','',string.punctuation)))
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def remove_stopwords(tokens):
    """Remove stopwords from tokens.
    Args: List of tokens.
    Output: List of filtered tokens.
    """
    filtered = [w for w in tokens if not w in STOPWORDS]
    return filtered

def remove_uppercase(tokens):
    """Remove all uppercase appearances (horse and city names)."""
    lower = [w for w in tokens if not w.istitle()]
    return lower

def pos_filter(tokens):
    """Assign part of speech tags to each token.
    Keep only tokens that are nouns, proper nouns or foreign words, verbs and adjectives of any kind.
    Arg: List of tokens.
    Output: List of filtered tokens.
    """
    #create list of tuples with pos for each token
    text_pos = nltk.pos_tag(tokens)
    keep = ['NN', 'NNP', 'NNPS', 'NNS', 'FW', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
    text = [pos[0] for pos in text_pos if pos[1] in keep]
    return text

def stem(tokens):
    stemmer = gensim.parsing.PorterStemmer()
    stemmed = [stemmer.stem(item) for item in tokens if not item.istitle()]
    return stemmed

def raceId_conversion(raceId):
    raceId = raceId.replace('.', '')
    return raceId

def split_sentences(text):
    """Splits text at :, ; and .
    Retruns a list"""
    splitted = re.split(r'[:;.]', t)
    return splitted

def ngram(token, n):
    grams = [token[i:i+n] for i in range(len(token)-n+1)]
    return [' '.join(gram) for gram in grams]

def build_grams(text, n):
    sents = split_sentences(text)
    tokens = [tokenize(sent) for sent in sents]
    grams = [ngram(token, n) for token in tokens]
    grams = [item for sublist in grams for item in sublist]
    return grams


# ## Load Data, Define Target

# In[230]:

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[232]:

train['raceId'] = train['raceId'].apply(lambda x: int(raceId_conversion(x)))
test['raceId'] = test['raceId'].apply(lambda x: int(raceId_conversion(x)))


# In[233]:

train['text'] = train['text'].fillna('')
test['text'] = test['text'].fillna('')


# ## NLP

# ### Single Stemmed Tokens 

# In[235]:

train['singles'] = train['text'].apply(lambda x: tokenize(x)).apply(lambda x: remove_stopwords(x)).apply(lambda x: stem(x))
test['singles'] = test['text'].apply(lambda x: tokenize(x)).apply(lambda x: remove_stopwords(x)).apply(lambda x: stem(x))


