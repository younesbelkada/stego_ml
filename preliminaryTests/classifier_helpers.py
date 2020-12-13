from demoRoBERTa import *

from nltk.corpus import brown
import nltk

import string
import os, errno

import fasttext
import csv

import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

NUM_OF_ARTICLES = 200

def gen_fake_text(mod, tok, start_text, start_secret, secret_text):
  ranks = get_ranks(mod, tok, start_secret, secret_text)
  cover_text = generate_cover_text(mod, tok, start_text, ranks)

  return cover_text

def get_news(mod, tok, start_secret, secret_text, corpus_size=NUM_OF_ARTICLES, secret_text_length=5):

    news_text = brown.sents(categories=['news'])[:corpus_size]
    fake_news = []
    true_news = []

    iter_index = 1

    for sentence in news_text:
        if len(sentence) <= secret_text_length:
            continue

        sentence_true = ' '.join(sentence)

        sentence_true = sentence_true.lower()

        true_news.append(sentence_true)

        sentence_fake = ' '.join(sentence[:-secret_text_length])
        
        sentence_fake = gen_fake_text(mod, tok, sentence_fake, start_secret, secret_text)

        fake_news.append(sentence_fake)

        print(iter_index, '/', corpus_size)
        iter_index += 1
    
    return true_news, fake_news

def prepare_news(true_news, fake_news):
    punctuation_dict = {}
    for x in string.punctuation:
        punctuation_dict[ord(x)] = None

    print(true_news[1].translate(punctuation_dict))

    for i in range(len(true_news)):
        true_news[i] = true_news[i].translate(punctuation_dict)

    for i in range(len(fake_news)):
        fake_news[i] = fake_news[i].translate(punctuation_dict)

    # s.translate({ord(c): None for c in string.whitespace})
    
    for i in range(len(true_news)):
        true_news[i] = true_news[i].lower()
    
    for i in range(len(fake_news)):
        fake_news[i] = fake_news[i].lower()
        
    for i in range(len(true_news)):
        true_news[i] = true_news[i].strip()

    for i in range(len(fake_news)):
        fake_news[i] = fake_news[i].strip()
    
    return true_news, fake_news

def prepare_df(true_news, fake_news):
  all_dict = {'text': true_news + fake_news, 'label': [0]*len(true_news)+[1]*len(fake_news)}

  df = pd.DataFrame(data=all_dict, columns=['label', 'text'])
  return df

def prepare_df_for_fasttext(orig_df, train_test_ratio):
  df = orig_df.copy()

  df['label'] = df['label'].astype('str')
  df['label'] = '__label__' + df['label']

  df_fasttext_train, df_fasttext_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)

  df_fasttext_train.to_csv('train.txt', 
          index = False, 
          sep = ' ',
          header = None, 
          quoting = csv.QUOTE_NONE, 
          quotechar = "", 
  escapechar = " ")

  df_fasttext_test.to_csv('test.txt', 
          index = False, 
          sep = ' ',
          header = None, 
          quoting = csv.QUOTE_NONE, 
          quotechar = "", 
  escapechar = " ")

  return
