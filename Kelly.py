
# coding: utf-8

# In[ ]:

"""Takes CL Logit Prediction-Frame as input."""


# In[3]:

import pandas as pd
import numpy as np


# In[39]:

test = pd.read_csv('sample1_preds.csv', index = False)


# In[40]:

ror = pd.DataFrame()
names = ['track', 'LR', 'RFC', 'MNB', 'unigram2', 'unigram3', 'bigram2', 'bigram3', 'trigram2', 'trigram3']
for name in names:
    test["expected_" + name] = test[name]*(test["ispDecimal"].apply(lambda x: x+1))
    test = test.groupby("raceId").apply(lambda x: x.sort_values(["expected_" + name], ascending = False)).reset_index(drop = True)
    ids = test['raceId'].unique().tolist()
    #left = test.loc[test[name + '_optimalFraction'].isnull() == True]['raceId'].unique().tolist()

    for i in ids:
    #for i in left:
        race = test.loc[test["raceId"] == i]
        n = list(range(len(race)))
        frac = 1
        for k in n:
            if race.iloc[k]['expected_'+ name] > frac:
                sigma_k = float(race[['track_prob']][:(k+1)].sum())
                pi_k = float(race[[name]][:(k+1)].sum())
                frac = (1-pi_k)/(1-sigma_k)
            else:
                break
    
        optimalFraction = np.where((race[name] - race['track_prob']*frac)>0, race[name] - race['track_prob']*frac, 0)
        test.loc[test["raceId"] == i, name + '_optimalFraction'] = optimalFraction
        race[name + '_optimalFraction'] = optimalFraction
        #while True:
            #try:
                #stake = (race['prizeFund']*race[name + '_optimalFraction']).sum()
                #assume 1 Pound bet on each race
        stake = race[name + '_optimalFraction'].sum()
        gain = ((race[name + '_optimalFraction'] + 
             race[name + '_optimalFraction']*race['ispDecimal'])* race['winner']).sum()
        #gain = (race["base_optimalFraction"]*(race['ispDecimal']+1)*race['winner']).sum()
        wins = np.repeat(gain-stake, len(race))
        bets = np.repeat(stake, len(race))
        
        test.loc[(test["raceId"] == i), name + "_wins"] = wins
        test.loc[(test["raceId"] == i), name + "_bets"] = bets

    unique = test.drop_duplicates("raceId")
    
    rate = (unique[name +'_wins'].sum())/(unique[name + '_bets'].sum())
    ror[name] = pd.Series(rate)

