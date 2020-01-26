
# coding: utf-8

# In[47]:

"""Load train dictionary and corpus of sentence based data.
Load processed test data.
Load models.
Transform test docs into bag-of-words-space of train data and into LDA space of chosen models."""


import gensim
from gensim import corpora
import pandas as pd
import numpy as np

#TM
from gensim import models
from gensim.models import tfidfmodel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary 
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import ast

def topwords_in_topics(models):
    """Show top 20 words of each topic and their probabilities in each model."""
    top20 = []
    for model in models:
        topics = model.num_topics
        top20_words = model.print_topics(num_topics=topics, num_words=20)
        top20.append({topics:top20_words})
    return top20



# # Load Models, Dicts and Corpora

# ## Models

# In[48]:

# choose all models with 2 and 3 topics (due to coherence measures)

topic_no = [2, 3]
sorts = ['singles_', 'bigrams_', 'trigrams_']
model_names = [sort + str(no) + '.model' for sort in sorts for no in topic_no]
models = [gensim.models.LdaModel.load(name) for name in model_names]


# ## Train

# In[49]:

train_dict_singles = Dictionary.load('train_dict_singles.dict', mmap = None)
train_dict_bigrams = Dictionary.load('train_sent_dict_bigrams.dict', mmap = None)
train_dict_trigrams = Dictionary.load('train_sent_dict_trigrams.dict', mmap = None)

train_corpus_singles = gensim.corpora.MmCorpus('train_corpus_singles.mm')
train_corpus_bigrams = gensim.corpora.MmCorpus('train_sent_corpus_bigrams.mm')
train_corpus_trigrams = gensim.corpora.MmCorpus('train_sent_corpus_trigrams.mm')


# ## Test 

# In[50]:

test = pd.read_csv('sent_test_grams.csv')


# In[51]:

test_singles = test['singles'].apply(lambda x: ast.literal_eval(x)).tolist()
test_bigrams = test['bigrams'].apply(lambda x: ast.literal_eval(x)).tolist()
test_trigrams = test['trigrams'].apply(lambda x: ast.literal_eval(x)).tolist()


# In[36]:

test_dict_singles = corpora.Dictionary(test_singles)
test_dict_bigrams = corpora.Dictionary(test_bigrams)
test_dict_trigrams = corpora.Dictionary(test_trigrams)


# # Project Test Data into Model Space

# ## Singles

# In[31]:

##Wirte word prominence to file

file = open('LDA_eval_singles.txt', 'w')
single_models = [models[0], models[1]]
word_prom = topwords_in_topics(single_models)
file.write("Top 20 words for single models with two and three topics:" + str(word_prom))
file.close()


# In[53]:

test_corpus = [train_dict_singles.doc2bow(single) for single in test_singles]


# In[56]:

for model in single_models:
    num = model.num_topics
    print(num)
    #train data
    doc_prob_matrix = model[train_corpus_singles]
    matrix = gensim.matutils.corpus2dense(doc_prob_matrix, num_terms = num)
    frame = pd.DataFrame(matrix.transpose())
    frame.to_csv("Single_trainProbs_" + str(num) + ".csv")
    
    #test data
    doc_prob_matrix_test = model[test_corpus]
    matrix_test = gensim.matutils.corpus2dense(doc_prob_matrix_test, num_terms = num)
    frame_test = pd.DataFrame(matrix_test.transpose())
    frame_test.to_csv("Single_testProbs_" + str(num) + ".csv")


# In[45]:

frame


# ## Bigrams

# In[57]:

##Write word prominence to file

file = open('LDA_eval_bigrams.txt', 'w')
bigram_models = [models[2], models[3]]
word_prom = topwords_in_topics(bigram_models)
file.write("Top 20 words for bigram models with two and three topics:" + str(word_prom))
file.close()


# In[58]:

test_corpus = [train_dict_bigrams.doc2bow(bigram) for bigram in test_bigrams]

for model in bigram_models:
    num = model.num_topics
    print(num)
    #train data
    doc_prob_matrix = model[train_corpus_bigrams]
    matrix = gensim.matutils.corpus2dense(doc_prob_matrix, num_terms = num)
    frame = pd.DataFrame(matrix.transpose())
    frame.to_csv("Bigram_trainProbs_" + str(num) + ".csv")
    
    #test data
    doc_prob_matrix_test = model[test_corpus]
    matrix_test = gensim.matutils.corpus2dense(doc_prob_matrix_test, num_terms = num)
    frame_test = pd.DataFrame(matrix_test.transpose())
    frame_test.to_csv("Bigram_testProbs_" + str(num) + ".csv")


# ## Trigrams

# In[59]:

##Wirte word prominence to file

file = open('LDA_eval_trigrams.txt', 'w')
trigram_models = [models[4], models[5]]
word_prom = topwords_in_topics(trigram_models)
file.write("Top 20 words for trigram models with two and three topics:" + str(word_prom))
file.close()


# In[60]:

test_corpus = [train_dict_trigrams.doc2bow(trigram) for trigram in test_trigrams]

for model in trigram_models:
    num = model.num_topics
    print(num)
    #train data
    doc_prob_matrix = model[train_corpus_trigrams]
    matrix = gensim.matutils.corpus2dense(doc_prob_matrix, num_terms = num)
    frame = pd.DataFrame(matrix.transpose())
    frame.to_csv("Trigram_trainProbs_" + str(num) + ".csv")
    
    #test data
    doc_prob_matrix_test = model[test_corpus]
    matrix_test = gensim.matutils.corpus2dense(doc_prob_matrix_test, num_terms = num)
    frame_test = pd.DataFrame(matrix_test.transpose())
    frame_test.to_csv("Trigram_testProbs_" + str(num) + ".csv")


# # Combine Data with Topic Probs

# In[95]:

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[72]:

topic_no = [2, 3]
sorts = ['Single_', 'Bigram_', 'Trigram_']
sets = ['trainProbs', 'testProbs']
frames = [sort + s  + '_'+ str(no) + ".csv" for sort in sorts for s in sets for no in topic_no]


# In[140]:

len(train2[0])


# In[89]:

data = [pd.read_csv(frame).drop("Unnamed: 0", axis=1) for frame in frames]


# In[92]:

sorts = ['Single_', 'Bigram_', 'Trigram_']

train2 = [data[0], data[4], data[8]]
test2 = [data[2], data[6], data[10]]
train3 = [data[1], data[5], data[9]]
test3 = [data[3], data[7], data[11]]

for sort, train in zip(sorts, train2):
    train.columns = [sort + '2_1', sort + '2_2']
for sort, test in zip(sorts, test2):
    test.columns = [sort + '2_1', sort + '2_2']
    
for sort, train in zip(sorts, train3):
    train.columns = [sort + '3_1', sort + '3_2', sort + '3_3']
for sort, test in zip(sorts, test3):
    test.columns = [sort + '3_1', sort + '3_2', sort + '3_3']


# In[121]:

tr = pd.concat([train] + train2 + train3, axis=1).drop(['Unnamed: 0', 'Unnamed: 0.1', 'horseCode', 'distanceCumulative',
                                                       'courseId','raceTypeChar', 'raceSurfaceChar', 'distance', 'going',
                                                       'text', 'winner'], axis=1)


# In[125]:

tst = pd.concat([test] + test2 + test3, axis=1).drop(['Unnamed: 0', 'Unnamed: 0.1', 'horseCode', 'distanceCumulative',
                                                       'courseId','raceTypeChar', 'raceSurfaceChar', 'distance', 'going',
                                                       'text', 'winner'], axis=1)


# In[123]:

tr.to_csv('train_topics.csv' , index = False)


# In[137]:

tst.to_csv('test_topics.csv', index = False)


# In[142]:

tst


# # Visuals

# In[144]:

model_names


# In[149]:

def visualize_model(models):
    for model in models:
        vis_data = gensimvis.prepare(model, train_corpus, train_dictionary)
        pyLDAvis.display(vis_data)


single2 = gensimvis.prepare(models[0], train_corpus_singles, train_dict_singles)
pyLDAvis.display(single2)



# In[150]:

single3 = gensimvis.prepare(models[1], train_corpus_singles, train_dict_singles)
pyLDAvis.display(single3)


# In[151]:

bigram2 = gensimvis.prepare(models[2], train_corpus_bigrams, train_dict_bigrams)
pyLDAvis.display(bigram2)


# In[152]:

bigram3 = gensimvis.prepare(models[3], train_corpus_bigrams, train_dict_bigrams)
pyLDAvis.display(bigram3)


# In[153]:

trigram2 = gensimvis.prepare(models[4], train_corpus_trigrams, train_dict_trigrams)
pyLDAvis.display(trigram2)


# In[154]:

trigram3 = gensimvis.prepare(models[5], train_corpus_trigrams, train_dict_trigrams)
pyLDAvis.display(trigram3)


# In[ ]:



