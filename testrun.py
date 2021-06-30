#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import re
import sys
import os
import ast

from tqdm import tqdm
from collections import Counter
from pathlib import Path
#from pandas import json_normalize #Deprecated? Also not used


# In[2]:


import sys
import re, numpy as np, pandas as pd
from pprint import pprint

from pandas.io.json import json_normalize

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
import nltk
from gensim.utils import simple_preprocess#, lemmatize deprecetated
from gensim.models import CoherenceModel

from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors

from nltk.corpus import stopwords
import gensim.downloader as api
import string
from sklearn.semi_supervised import LabelSpreading, LabelPropagation


# # **`Clean and Process Data `**

# In[3]:


import nltk
#nltk.download() #Dowload stopwords, only needed to update after first download


# In[4]:


#Opens the data and reorganizes it by tweets instead of by events
with open('./Trec_data/labeled_by_event.json') as f:
    js = json.load(f)
df = json_normalize(data=js['events'], record_path='tweets')
df.to_json('labeled.json', orient='records', lines=True)

df


# In[5]:


stop_words = stopwords.words('english')
stop_words.extend(["http", "https", "rt", "@", ":", "t.co", "co", "amp", "&amp;", "...", "\n", "\r"])
stop_words.extend(string.punctuation)


# In[6]:


def sent_to_words(sentences):
    for sent in sentences:
        sent = str(sent)
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        yield(sent)


# In[7]:


# Look into what shape this ./labeled.json
df = pd.read_json("labeled.json", orient='records', lines=True)
data = df.postText.values.tolist()
print(len(data))

data_words = list(sent_to_words(data))

# print(data_words[0:5]) it seems that t.co isn't in this list, so it's not being erased as a stop word
#the urls are also not being removed


# In[8]:


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[ ]:


def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #Changed en to be full name en_core_web_sm #might change to trf
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out


# In[ ]:


print(len(data_words))


# In[ ]:


data_ready = process_words(data_words)


# In[13]:


#df.to_json('data_ready.json', orient='records', lines=True)


# In[14]:


print(len(df.eventID), len(df.eventType), len(data_ready))
#eventID = df.eventID.values.tolist()
print(data_ready[:10])
#print(data_ready[:5])
#print(len(data_ready), len())

#I think you have to turn all empty lists into null

count = 0

for i in data_ready:
    if i == []:
        count += 1
        
print(count)

df['processed_text']=data_ready
df.to_json('processed.json', orient='records', lines=True)


# In[15]:


#What does this do?
#cp -a ./PR_all_Labeled.json  gdrive/My\ Drive/Code/


# In[9]:


data = pd.read_json("processed.json", orient='records',lines=True)


# # Generating Similarity Scores and Matrix
# ### **Mean and Cosine Similarity(each event with all other event-types)**
# 

# In[10]:


import pickle
#import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline
from scipy import sparse
import numpy as np
import os.path
import re
from tqdm import tqdm
import warnings
import pandas as pd


# In[18]:


# remove irrelevant tweets
data = pd.read_json("processed.json", orient='records',lines=True)
print(data.columns)

data.loc[(data.eventID == 'parisAttacks2015'),'event_type']='shooting' #Do we need
def label_ir_tweets(postCategories):

    if 'Irrelevant' in postCategories:
        return 1
    else:
        return 0
data['ir'] = [label_ir_tweets(x) for x in data['postCategories']]
data=data.query("ir == 0")
# data=data.query("label ==1")


# In[19]:


#This only removes all tweets except ovbservations
#Do not run but keep for reference
def label_observation_tweets(postCategories):
    if 'FirstPartyObservation' in postCategories or 'ThirdPartyObservation' in postCategories :
        return 1
    else:
        return 0
#data['obs'] = [label_observation_tweets(x) for x in data['categories']]
#data=data.query("obs == 1")
# data=data.query("label ==1")


# In[21]:


data['l'] = data.apply(lambda row: len(row['processed_text']), axis=1)
data= data.query("l >1")
data.drop(columns=['l'], inplace=True)
data


# In[ ]:


#This would run through the ssh oopsy, do not run
#pip install -U sentence-transformers


# In[22]:


# generate sentnece embedding
class SBERT:

    def __init__(self, lang="en"):
        from sentence_transformers import SentenceTransformer
        self.name = "SBERT"
        if lang == "fr":
            self.model = SentenceTransformer(
                "/home/bmazoyer/Dev/pytorch_bert/output/sts_fr_long_multilingual_bert-2019-10-01_15-07-03")
        elif lang == "en":
            self.model = SentenceTransformer(
                # "bert-large-nli-stsb-mean-tokens"
                "roberta-large-nli-stsb-mean-tokens"
            )
# roberta-large-nli-stsb-mean-tokens
    def compute_vectors(self, data):
        data["text"] = data.postText.str.slice(0, 500)
        vectors = np.array(self.model.encode(data.text.tolist()))
        return vectors


# In[23]:


sbert=SBERT()


# In[34]:


v=sbert.compute_vectors(data)
data['sbert_emb']=[item for item in v]


# In[223]:


v


# In[35]:


from numpy import dot
from numpy.linalg import norm


# In[36]:


#Can't handle the list of postType
def generate_similarity_matrix (frame, grouping, group_types):
    #generate similarity scores dataframe
    group_ranks=pd.DataFrame()
    for heldout_event in group_types:

        training = frame[frame[grouping] != heldout_event]
        test = frame[frame[grouping] == heldout_event]

        ref=np.mean(test["sbert_emb"], axis=0)

        grpups=training.groupby(grouping) #Might need to be changed 
        ranks={}
        ranks["reference-group"]=heldout_event
        for name, group in grpups:
           val=np.mean(group["sbert_emb"], axis=0)
           cos_sim = dot(ref, val)/(norm(ref)*norm(val))
           ranks[name]=cos_sim

        # event_ranks[heldout_event]=ranks
        #print(ranks)
        group_ranks = group_ranks.append(ranks, ignore_index=True)
    group_ranks.set_index("reference-group",inplace=True)
    #group_ranks=frame.groupby(grouping) #Does the label==1 need to change?
    return group_ranks


# In[37]:


import seaborn as sns


# In[38]:


#is this even required? and is this based off the 4 different sources?
events=[ 
'2014_Philippines_Typhoon_Hagupi',
 '2015_Cyclone_Pam',
 'albertaFloods2013',
 'albertaWildfires2019',
 'australiaBushfire2013',
 'cycloneKenneth2019',
 'fireYMM2016',
 'hurricaneFlorence2018',
 'keralaFloods2019',
 'manilaFloods2013',
 'philipinnesFloods2012',
 'queenslandFloods2013',
 'southAfricaFloods2019',
 'typhoonHagupit2014',
 'typhoonYolanda2013'
]
event_types=['typhoon', 'storm', 'wildfire', 'covid', 'flood',
       'shooting', 'earthquake', 'explosion', 'hostage', 'fire',
       'tornado']
event_types


# In[41]:


event_ranks = generate_similarity_matrix(data, 'eventType', event_types)
event_ranks = event_ranks.replace(np.nan, 1)
event_ranks.to_csv("event_ranks.csv")
event_ranks
#cp -a ./event_ranks_roberta.csv gdrive/My\ Drive/Code/


# In[147]:


critical_types=['Low', 'Medium', 'High', 'Critical']
print((critical_types))
critical_ranks = generate_similarity_matrix(data, 'postPriority', critical_types) #check what this variable is called in the dataframe
critical_ranks = critical_ranks.replace(np.nan, 1)
critical_ranks.to_csv("critical_ranks.csv")
critical_ranks


# In[159]:


for i in data['postCategories']:
    print(len(i))


# In[153]:


info_types=df['postCategories'].explode().unique() #double check that this is string based
info_types = info_types.tolist()
info_types = ['Irrelevant', 'Location', 'MultimediaShare', 'ContextualInformation', 'Weather', 'Discussion', 'Hashtags', 'News', 'Official', 'EmergingThreats', 'FirstPartyObservation', 'Factoid', 'ThirdPartyObservation', 'MovePeople', 'Sentiment', 'Advice', 'GoodsServices', 'Donations', 'ServiceAvailable', 'SearchAndRescue', 'NewSubEvent', 'Volunteer', 'CleanUp', 'InformationWanted', 'OriginalEvent']


# In[224]:


cat = []
for i in data['postCategories']:
    if len(i) == 1:
        listToStr = ' '.join(map(str, i))
        cat.append(listToStr)
    else:
        i = tuple(i)
        #listToStr = ' '.join(map(str, i))
        cat.append(i)
        
print(cat)


# In[190]:


cat = map(lambda s: s.strip(), cat)

converted_list = []


for element in cat:
    print(type(element))
    converted_list.append(element.strip())


# In[221]:


cut = []
for i in data['postCategories']:
    if len(i) == 1:
        listToStr = ' '.join(map(str, i))
        cut.append(listToStr)
    else:
        i = i[0]
        listToStr = ''.join(map(str, i))
        cut.append(listToStr)


# In[225]:


data['tupled']=[item for item in cat]


# In[227]:


data.head()


# In[226]:


#info_ranks = generate_similarity_matrix(data, 'postCategories', info_types) #check what this variable is called in the dataframe
info_ranks = generate_similarity_matrix(data, 'tupled', info_types)
info_ranks.to_csv("info_ranks.csv")
info_ranks


# In[191]:


#info_ranks = generate_similarity_matrix(data, 'postCategories', info_types) #check what this variable is called in the dataframe
info_ranks = info_ranks.replace(np.nan, 1)
info_ranks.to_csv("info_ranks.csv")
info_ranks


# In[84]:


import matplotlib.pyplot as plt
ax = plt.axes()

ax.set_title('Pairs of Priorities')
sns.heatmap(critical_ranks, ax=ax,cmap="YlGnBu", annot=True)
plt.ylabel(' ')
#sns.heatmap(dt_tweet_cnt, ax=ax2)

b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!

plt.show()

import matplotlib
matplotlib.__version__


# In[120]:


event_ranks = pd.read_csv("event_ranks.csv", index_col = 0)
#type(event_ranks)


# In[123]:


event_ranks_final = pd.read_csv("event_ranks_final.csv", index_col = 0)
event_ranks_final


# In[124]:


# Visualize similarity matrix for event type using heatmap

import matplotlib.pyplot as plt
ax = plt.axes()

ax.set_title('Pairs of Event Types')
sns.heatmap(event_ranks, ax=ax,cmap="YlGnBu", annot=True)
plt.ylabel(' ')
#sns.heatmap(dt_tweet_cnt, ax=ax2)

b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!


# In[126]:


# Visualize similarity matrix for event type using heatmap

import matplotlib.pyplot as plt
ax = plt.axes()

ax.set_title('Pairs of Event Types')
sns.heatmap(event_ranks_final, ax=ax,cmap="YlGnBu", annot=True)
plt.ylabel(' ')
#sns.heatmap(dt_tweet_cnt, ax=ax2)

b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!


# In[ ]:




