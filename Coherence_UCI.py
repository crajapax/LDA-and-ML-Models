
# coding: utf-8

# In[18]:

"""Load train dictionary and corpus of sentence based data.
Load models.
Get coherence measures for each model to choose the best K. 
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

def get_coherence(models):
    """U-Mass coherentation in and out of sample. 
    Interpretation: the higher the stats the better the model."""
    descriptives = []  
    #Intra Coherence
    for model in models:
        model_desc = {}
        topics = model.num_topics
        intra_stats = CoherenceModel(model=model, corpus=train_corpus, dictionary=train_dictionary, coherence='u_mass').get_coherence()
        out_stats = CoherenceModel(model=model, corpus=test_corpus, dictionary=test_dictionary, coherence='u_mass').get_coherence()
        model_desc.update({'In-Sample Fit': intra_stats, 'Out-of-Sample Fit': out_stats, 'Model': topics})
        descriptives.append(model_desc)
    return pd.DataFrame(descriptives, columns = ['Model', 'In-Sample Fit', 'Out-of-Sample Fit']) 


# # Load Dicts and Corpora

# ## Train

# In[32]:

train_dict_singles = Dictionary.load('train_dict_singles.dict', mmap = None)
train_dict_bigrams = Dictionary.load('train_sent_dict_bigrams.dict', mmap = None)
train_dict_trigrams = Dictionary.load('train_sent_dict_trigrams.dict', mmap = None)

# transform test data into semantic space
test_corpus_singles = [train_dict_singles.doc2bow(token) for token in unigrams]
test_corpus_bigrams = [train_dict_bigrams.doc2bow(token) for token in bigrams]
test_corpus_trigrams = [train_dict_trigrams.doc2bow(token) for token in trigrams]


# In[10]:

unigrams = test['singles'].apply(literal_eval).tolist()
bigrams = test['bigrams'].apply(literal_eval).tolist()
trigrams = test['trigrams'].apply(literal_eval).tolist()


# # Model Coherence

# ## Singles

# In[37]:

#load all models
topic_no = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
model_names = ['singles_' + str(no) + '.model' for no in topic_no]
models = [gensim.models.LdaModel.load(name) for name in model_names]


# In[38]:

f = open('uci_unigram.txt', 'w')
coh = {}
for no, model in zip(topic_no, models):
    coherence = CoherenceModel(model=model, texts=unigrams, dictionary=test_dict_singles, coherence='c_uci').get_coherence()
    coh[no] = coherence
    f.write('Topic Number: %d,  Coherence: %.4f. \n' % (no, coherence))
f.close()


# ## Bigrams

# In[26]:

model_names = ['bigrams_' + str(no) + '.model' for no in topic_no]
models = [gensim.models.LdaModel.load(name) for name in model_names]


# In[27]:

f = open('uci_bigrams.txt', 'w')
coh_bigram = {}
for no, model in zip(topic_no, models):
    coherence = CoherenceModel(model=model, texts=bigrams, dictionary=test_dict_bigrams, coherence='c_uci').get_coherence()
    coh_bigram[no] = coherence
    f.write('Topic Number: %i,  Coherence: %.4f. \n' % (no, coherence))
f.close()


# ## Trigragrams
f = open('uci_trigrams.txt', 'w')
model_names = ['trigrams_' + str(no) + '.model' for no in topic_no]
models = [gensim.models.LdaModel.load(name) for name in model_names]
coh_trigram = {}
for no, model in zip(topic_no, models):
    coherence = CoherenceModel(model=model, texts=trigrams, dictionary=test_dict_trigrams, coherence='c_uci').get_coherence()
    coh_trigram[no] = coherence
    f.write('Topic Number: %i,  Coherence: %.4f. \n' % (no, coherence))
f.close()

