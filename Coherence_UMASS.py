
# coding: utf-8

# In[1]:

"""Load train dictionary and corpus of sentence based data.
Load models.
Get coherence measures for each model to choose the K. 
"""

#LinReg
import pandas as pd
import numpy as np


#NLP
import string
import gensim
from gensim import corpora

#TM
from gensim import models
from gensim.models import tfidfmodel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary 
import pyLDAvis.gensim as gensimvis
import pyLDAvis

from ast import literal_eval
from astropy.table import Table

def topwords_in_topics(models):
    """Show top 10 words of each topic and their probabilities in each model."""
    top10 = []
    for model in models:
        topics = model.num_topics
        top10_words = model.print_topics(num_topics=topics, num_words=10)
        top10.append({topics:top10_words})
    return top10

# # Load Dicts and Corpora

# ## Train

# In[2]:

train_dict_singles = Dictionary.load('train_dict_singles.dict', mmap = None)
train_dict_bigrams = Dictionary.load('train_sent_dict_bigrams.dict', mmap = None)
train_dict_trigrams = Dictionary.load('train_sent_dict_trigrams.dict', mmap = None)

train_corpus_singles = gensim.corpora.MmCorpus('train_corpus_singles.mm')
train_corpus_bigrams = gensim.corpora.MmCorpus('train_sent_corpus_bigrams.mm')
train_corpus_trigrams = gensim.corpora.MmCorpus('train_sent_corpus_trigrams.mm')


# # Model Coherence

# ## Singles

# In[3]:

#load all models
topic_no = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
model_names = ['singles_' + str(no) + '.model' for no in topic_no]
models = [gensim.models.LdaModel.load(name) for name in model_names]


# In[4]:

f = open('umass_unigram.txt', 'w')
coh = {}
for no, model in zip(topic_no, models):
    coherence = CoherenceModel(model=model, corpus = train_corpus_singles, coherence='u_mass').get_coherence()
    coh[no] = coherence
    f.write('Topic Number: %d,  Coherence: %.4f. \n' % (no, coherence))
f.close()


# ## Bigrams

# In[5]:

model_names = ['bigrams_' + str(no) + '.model' for no in topic_no]
models = [gensim.models.LdaModel.load(name) for name in model_names]


# In[6]:

f = open('umass_bigrams.txt', 'w')
coh_bigram = {}
for no, model in zip(topic_no, models):
    coherence = CoherenceModel(model=model, corpus = train_corpus_bigrams, coherence='u_mass').get_coherence()    
    coh_bigram[no] = coherence
    f.write('Topic Number: %i,  Coherence: %.4f. \n' % (no, coherence))
f.close()


# ## Trigragrams

# In[7]:

model_names = ['trigrams_' + str(no) + '.model' for no in topic_no]
models = [gensim.models.LdaModel.load(name) for name in model_names]


# In[8]:

f = open('umass_trigrams.txt', 'w')
coh_trigram = {}
for no, model in zip(topic_no, models):
    coherence = CoherenceModel(model=model, corpus = train_corpus_trigrams, coherence='u_mass').get_coherence()
    coh_trigram[no] = coherence
    f.write('Topic Number: %i,  Coherence: %.4f. \n' % (no, coherence))
f.close()




