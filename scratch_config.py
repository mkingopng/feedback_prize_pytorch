"""

"""
# imports
import os
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from transformers import *
import sentencepiece
from transformers import AutoTokenizer, AutoModelForTokenClassification

import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda



# declare how many gpus you wish to use. kaggle only has 1, but offline, you can use more
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0,1,2,3 for four gpu

# version for saving model weights
VER = 26

# if variable is none, then notebook computes tokens otherwise notebook loads tokens from path
LOAD_TOKENS_FROM = 'py-bigbird-v26'

# if variable is none, then notebook trains a new model otherwise it loads your previously trained model
LOAD_MODEL_FROM = 'py-bigbird-v26'

# if following is none, then notebook uses internet and downloads huggingface config, tokenizer, and model
DOWNLOADED_MODEL_PATH = 'py-bigbird-v26'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'google/bigbird-roberta-base'

config = {'model_name': MODEL_NAME,
          'max_length': 1024,
          'train_batch_size': 4,
          'valid_batch_size': 4,
          'epochs': 5,
          'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
          'max_grad_norm': 10,
          'device': 'cuda' if cuda.is_available() else 'cpu'}

# this will compute val score during commit but not during submit
COMPUTE_VAL_SCORE = True
if len(os.listdir('data/test')) > 5:
    COMPUTE_VAL_SCORE = False

if DOWNLOADED_MODEL_PATH == 'model':
    os.mkdir('model')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    tokenizer.save_pretrained('model')

    config_model = AutoConfig.from_pretrained(MODEL_NAME)
    config_model.num_labels = 15
    config_model.save_pretrained('model')

    backbone = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config_model)
    backbone.save_pretrained('model')

train_df = pd.read_csv('data/train.csv')
print(train_df.shape)
train_df.head()

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, test_texts = [], []
for f in list(os.listdir('data/test')):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open('data/test/' + f, 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})
test_texts.head()

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, train_texts = [], []
for f in tqdm(list(os.listdir('data/train'))):
    test_names.append(f.replace('.txt', ''))
    train_texts.append(open('data/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})
train_text_df.head()

if not LOAD_TOKENS_FROM:
    all_entities = []
    for ii, i in enumerate(train_text_df.iterrows()):
        if ii % 100 == 0: print(ii, ', ', end='')
        total = i[1]['text'].split().__len__()
        entities = ["O"] * total
        for j in train_df[train_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]:
                entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    train_text_df['entities'] = all_entities
    train_text_df.to_csv('train_NER.csv', index=False)

else:
    from ast import literal_eval

    train_text_df = pd.read_csv(f'{LOAD_TOKENS_FROM}/train_NER.csv')
    # pandas saves lists as string, we must convert back
    train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x))

print(train_text_df.shape)
train_text_df.head()

# create dictionaries that we can use during train and infer # mk: these are the entities
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                 'I-Concluding Statement']

labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}

print(labels_to_ids)
