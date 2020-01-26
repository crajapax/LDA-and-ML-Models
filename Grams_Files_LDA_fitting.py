
# coding: utf-8

# In[1]:

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


# In[2]:

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
    stemmed = [stemmer.stem(item) for item in tokens]
    return stemmed

def raceId_conversion(raceId):
    raceId = raceId.replace('.', '')
    return raceId

def split_sentences(text):
    """Splits text at :, ; and .
    Retruns a list"""
    splitted = re.split(r'[:;.]', text)
    return splitted

def ngram(token, n):
    grams = [token[i:i+n] for i in range(len(token)-n+1)]
    return [' '.join(gram) for gram in grams]

def build_grams(text, n):
    sents = split_sentences(text)
    tokens = [tokenize(sent) for sent in sents]
    tokens = [remove_stopwords(token) for token in tokens]
    tokens = [stem(token) for token in tokens]
    grams = [ngram(token, n) for token in tokens]
    grams = [item for sublist in grams for item in sublist]
    return grams


# ## Load Data, Define Target

# In[ ]:

import pandas as pd
train = pd.read_csv("train_singles.csv")
test = pd.read_csv("test_singles.csv")


# In[44]:

train['raceId'] = train['raceId'].apply(lambda x: int(raceId_conversion(x)))
test['raceId'] = test['raceId'].apply(lambda x: int(raceId_conversion(x)))


# In[45]:

train['text'] = train['text'].fillna('')
test['text'] = test['text'].fillna('')


# In[46]:

train['singles'] = train['singles'].apply(literal_eval)
test['singles'] = test['singles'].apply(literal_eval)


# ## NLP

# ### Grams Creation 

# In[48]:

train['bigrams'] =  train['text'].apply(lambda x: build_grams(x, 2))
test['bigrams'] =  test['text'].apply(lambda x: build_grams(x, 2))


# In[49]:

train['trigrams'] =  train['text'].apply(lambda x: build_grams(x, 3))
test['trigrams'] =  test['text'].apply(lambda x: build_grams(x, 3))


# # LDA Files

# ## Create N-Gram Files

# In[54]:

sorts = ["bigrams", "trigrams"]

for sort in sorts:
    train_tokens = train[sort].tolist()
    test_tokens = test[sort].tolist()
    train_dict = corpora.Dictionary(train_tokens)
    train_dict.filter_extremes(no_below=1000, no_above=0.7)
    train_corpus = [train_dict.doc2bow(token, allow_update=True) for token in train_tokens]
    
    train_dict.save_as_text("train_sent_dictfile_" + sort, sort_by_word=True)
    train_dict.save('train_sent_dict_' + sort + '.dict')
    gensim.corpora.MmCorpus.serialize('train_sent_corpus_' + sort +'.mm', train_corpus)


# ## Create Unigram Files

# In[141]:

train_tokens = train['singles'].tolist()
test_tokens = test['singles'].tolist()

#assign each word an id and save in dictionary
train_dictSingle = corpora.Dictionary(train_tokens)

#no_below HAS to be an absolute number of docs and not fraction
train_dictSingle.filter_extremes(no_below=1000, no_above=0.7)

#create bag of words
train_corpusSingle = [train_dictSingle.doc2bow(token, allow_update=True) for token in train_tokens]



# In[ ]:

train_dictSingle.save_as_text("train_dictfileSingle", sort_by_word=True)
train_dictSingle.save('train_dictSingle.dict')
gensim.corpora.MmCorpus.serialize('train_corpusSingle.mm', train_corpusSingle)


# In[ ]:

sorts = ["singles", "bigrams", "trigrams"]

for sort in sorts:
    train_tokens = train[sort].tolist()
    test_tokens = test[sort].tolist()
    train_dict = corpora.Dictionary(train_tokens)
    train_dict.filter_extremes(no_below=1000, no_above=0.7)
    train_corpus = [train_dict.doc2bow(token) for token in train_tokens]
    
    #train_dict.save_as_text("train_dictfile1_" + sort, sort_by_word=True)
    train_dict.save('train_dict1_' + sort + '.dict')
    gensim.corpora.MmCorpus.serialize('train_corpus_1' + sort +'.mm', train_corpus)


# # Fit LDA

# In[ ]:

# Load dictionary, corpus, loop over different settings
topic_no = [2, 3, 4, 5, 6, 8, 10, 11]
names = ["Single", "Bigram", "Trigram"]
corpora = [train_corpus_singles, train_corpus_bigrams, train_corpus_trigrams]
dicts = [train_dict_ingle, train_dict_bigrams, train_dict_trigrams]

for name, train_corpus, train_dict in zip(names, corpora, dicts):
    for no in topic_no:
        lda = models.ldamodel.LdaModel(corpus=train_corpus, id2word=train_dict, num_topics=no, passes=20, eval_every = None)
        lda.save(name + '_' +str(no)+'.model')


# # TF-IDF Files

# ## Corpus Creation

# In[ ]:

#convert other regressors to sparse

train_dense = csr_matrix(train[['raceId', 'distanceCumulative', 'ispDecimal', 'positionNormal', 'winner']])
test_dense = csr_matrix(test[['raceId', 'distanceCumulative', 'ispDecimal', 'positionNormal', 'winner']])

sorts = ["singles", "bigrams", "trigrams"]

for sort in sorts:
    train_tokens = train[sort].tolist()
    test_tokens = test[sort].tolist()
    train_corpus = [" ".join(token) for token in train_tokens]
    test_corpus = [" ".join(token) for token in test_tokens]
    tf = TfidfVectorizer(max_df = 0.7, min_df = 1000, sublinear_tf = True)
    tfidf_train = tf.fit_transform(train_corpus)
    tfidf_test = tf.transform(test_corpus)
    
    train_weights = np.asarray(tfidf_train.mean(axis=0)).ravel().tolist()
    train_weights_df = pd.DataFrame({'term': tf.get_feature_names(), 'weight': train_weights})
    train_top_30 = train_weights_df.sort_values(by='weight', ascending=False).head(30)

    test_weights = np.asarray(tfidf_test.mean(axis=0)).ravel().tolist()
    test_weights_df = pd.DataFrame({'term': tf.get_feature_names(), 'weight': test_weights})
    test_top_30 = test_weights_df.sort_values(by='weight', ascending=False).head(30)
    
    train_top_30.to_csv("train_top_30_" + sort + '.csv")
    test_top_30.to_csv("test_top_30_" + sort + '.csv")
                       
    train_sparse = hstack((train_dense, tfidf_train), format = 'csr')
    test_sparse = hstack((test_dense, tfidf_test), format = 'csr')
                       
    io.mmwrite("train" + sort +"_sparse.mtx",train_sparse)
    io.mmwrite("test" + sort +"_sparse.mtx", test_sparse)
                       
#############################################################################################################

