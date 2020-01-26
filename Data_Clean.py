
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

def na_analysis(data):
    missing = data.apply(lambda x: sum(x.isnull()),axis=0) 
    return missing

def median_imputation(data):
    data[column].fillna(data[column].median(), inplace=True)

def detect_outlier(numerics):
    regular = numerics[((np.abs(numerics) - numerics.mean()) / numerics.std()) < 3]
    #regular.fillna(regular.median())
    return regular

def replace_outlier(data):   
    numerics = [c for c in data if data[c].dtype.kind == 'f' or train[c].dtype.kind == 'i']
    data.loc[:, numerics] = data.loc[:, numerics].apply(lambda x: detect_outlier(x))
    data.loc[:, numerics]=data.loc[:, numerics].fillna(data.loc[:, numerics].median(), inplace = True)
    return data

def normalized_target(data):
    race_max = data.groupby('raceId')['positionOfficial'].transform('max')
    race_min = data.groupby('raceId')['positionOfficial'].transform('min')
    normalized = (data['positionOfficial'] - race_min)/(race_max-race_min)
    normalTarget =  normalized.apply(lambda x: -0.5 + x)
    return normalTarget
   
def winner_coding(data):
    data['winner'] = np.where(data['positionOfficial'] <=3, 1, 0)
    return data

def text_merging(data):
    """Apply to data frame of each individual runner."""
    for k in list(range(len(data))):
        if k == 0:
            data.iloc[k, data.columns.get_loc('postCommentJoined')] = data.iloc[k, data.columns.get_loc('postComment')]
            data.iloc[k, data.columns.get_loc('text')] = str(data.iloc[k, data.columns.get_loc('productionComment')])
        else:    
            data.iloc[k, data.columns.get_loc('postCommentJoined')] = data.loc[:k, 'postComment'].str.cat(sep = ' ')
            data.iloc[k, data.columns.get_loc('text')] = str(data.iloc[k, data.columns.get_loc('productionComment')])+ ' ' + str(data['postCommentJoined'][k-1])

    return data



# In[4]:

data = pd.read_csv("timeform comments.csv")


# ## Define Target

# In[5]:

data['positionNormal'] = normalized_target(data)
data = winner_coding(data)


# ## Missing Value Analysis

# In[18]:

na_analysis(data)


# ## Fill Missing Values in Stings with empty String

# In[6]:

data['productionComment'] = data['productionComment'].fillna('')
data['performanceComment'] = data['performanceComment'].fillna('')
data['performanceCommentPremium'] = data['performanceCommentPremium'].fillna('')


# ## Join Post Race Comments on Horse

# In[113]:

data['postComment'] = data['performanceComment'].str.cat(data['performanceCommentPremium'], sep = ' ')


# ## Data Partitioning

# In[7]:

# divide data into 80% train and 20% test
races = data['raceId'].unique()
train_races = races[:int(len(races)*0.8)].tolist()

train = data[data['raceId'].isin(train_races) == True]
test = data[data['raceId'].isin(train_races) == False]


# In[38]:

test.columns


# ## Text Merging from Prior Races

# # Train

# In[116]:

"""Append pre-race comment on horse to post_race comments on horse from prior races.
Save data frame with raceId, horseCode, text to .csv."""

runnerIds = train['horseCode'].unique().tolist()
runners = pd.DataFrame()

for runnerId in runnerIds:
    runner = train[train['horseCode'] == runnerId].reset_index()
    runner['postCommentJoined'] = pd.Series()
    runner['text'] = pd.Series()
    merged = text_merging(runner)
    runners = runners.append(merged[["raceId", "horseCode", "productionComment", "postComment", "text"]])
#runners.to_csv("merged_text.csv", index = False)
final_train = pd.merge(left = train, right = runners[["raceId", "horseCode", "text"]], how = "left", 
                 left_on = ["raceId", "horseCode"], right_on = ["raceId", "horseCode"],
                 left_index = False, right_index = False, sort = False)
final_train.to_csv("final_train.csv")


# # Test

# In[117]:

"""Append pre-race comment on horse to post_race comments on horse from prior races.
Save data frame with raceId, horseCode, text to .csv."""

runnerIds = test['horseCode'].unique().tolist()
runners = pd.DataFrame()

for runnerId in runnerIds:
    runner = test[test['horseCode'] == runnerId].reset_index()
    runner['postCommentJoined'] = pd.Series()
    runner['text'] = pd.Series()
    merged = text_merging(runner)
    runners = runners.append(merged[["raceId", "horseCode", "productionComment", "postComment", "text"]])
#runners.to_csv("merged_text.csv", index = False)
final_test = pd.merge(left = test, right = runners[["raceId", "horseCode", "text"]], how = "left", 
                 left_on = ["raceId", "horseCode"], right_on = ["raceId", "horseCode"],
                 left_index = False, right_index = False, sort = False)
final_test.to_csv("final_test.csv")


# ## Vertical Reduction 

# In[126]:

final_train= final_train.drop(['horseName', 'positionStatus','meetingDate', 'courseName', 'startTimeLocalScheduled',
                  'perspectiveComment', 'productionComment', 'analystVerdict', 'performanceComment',
                  'performanceCommentPremium', 'postComment'], axis = 1)

final_test= final_test.drop(['horseName', 'positionStatus','meetingDate', 'courseName', 'startTimeLocalScheduled',
                  'perspectiveComment', 'productionComment', 'analystVerdict', 'performanceComment',
                  'performanceCommentPremium', 'postComment'], axis = 1)


# ## Horizontal Reduction: Drop NA

# In[40]:

train = pd.read_csv("final_train.csv")
test = pd.read_csv("final_test.csv")


# In[44]:

"""Drop races where at least one text is missing.
Drop races with only one runner."""
train = train.loc[train.groupby('raceId')['text'].filter(lambda x: len(x[pd.isnull(x)]) == 0).index]
train = train.dropna(subset=['positionNormal'], how = 'all')

test = test.loc[test.groupby('raceId')['text'].filter(lambda x: len(x[pd.isnull(x)]) == 0).index]
test = test.dropna(subset=['positionNormal'], how = 'all')


# In[196]:

len(test)
len(train)


# ## Outlier Filtering for Numeric Data

# In[197]:

train = replace_outlier(train)
test = replace_outlier(test)


# ## Divide Train Data in Actual Train and Validation

# In[198]:

# divide data into 80% train and 20% test
races = train['raceId'].unique()
train_races = races[:int(len(races)*0.6)].tolist()

valid = train[train['raceId'].isin(train_races) == False]

