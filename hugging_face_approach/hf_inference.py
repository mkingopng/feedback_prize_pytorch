"""
for now this if pretty much directly transposed from the notebook. I will refactor into proper .py code.
many of the functions are used in hf_train.py and are contained in hf_functions.py. Should help to streamline this
"""
import wandb
from hf_functions import *
from hf_config import *
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import torch
import os


# if __name__ == "__main__":

# Config
batch_size = 1
min_tokens = 5  # why is this different from the train config?
tok_checkpoint = 'longformer-large-4096-hf'
model_checkpoint = 'longformer-large-4096-hf-1/pytorch_model.bin'  # I'm not surprised that this is all we need but it makes me question why the save strategy in hf_train has us saving so many other things that are apparently unnecessary

# Load data
train = pd.read_csv('../data/train.csv')
print(train.head(10))

test = pd.read_csv('../data/sample_submission.csv')
print(test.head(1))

# Setup dictionaries
classes = train.discourse_type.unique().tolist()

tags = defaultdict()
for i, c in enumerate(classes):
    tags[f'B-{c}'] = i
    tags[f'I-{c}'] = i + len(classes)
tags[f'O'] = len(classes) * 2
tags[f'Special'] = -100
l2i = dict(tags)


i2l = defaultdict()
for k, v in l2i.items():
    i2l[v] = k
i2l[-100] = 'Special'
i2l = dict(i2l)


# Helper functions
test_path = Path('../data/test')

def get_test_text(ids):
    with open(test_path/f'{ids}.txt', 'r') as file: data = file.read()
    return data

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tok_checkpoint, add_prefix_space=True)

# Load model
model = AutoModelForTokenClassification.from_pretrained(tok_checkpoint, num_labels=len(i2l)-1)
model.load_state_dict(torch.load(model_checkpoint))
model.eval()

# data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# We'll use trainer with the loaded model to run inference on test set
trainer = Trainer(
    model,
    data_collator=data_collator,
    tokenizer=tokenizer,
)


# code that will convert our predictions into prediction strings. we'll skip visualization here.
# this most likely requires some refactoring

def get_class(c):
    if c == 14:
        return 'Other'
    else:
        return i2l[c][2:]


def pred2span(pred, example, viz=False, test=False):
    example_id = example['id']
    n_tokens = len(example['input_ids'])
    classes = []
    all_span = []
    for i, c in enumerate(pred.tolist()):
        if i == n_tokens - 1:
            break
        if i == 0:
            cur_span = example['offset_mapping'][i]
            classes.append(get_class(c))
        elif i > 0 and (c == pred[i - 1] or (c - 7) == pred[i - 1]):
            cur_span[1] = example['offset_mapping'][i][1]
        else:
            all_span.append(cur_span)
            cur_span = example['offset_mapping'][i]
            classes.append(get_class(c))
    all_span.append(cur_span)

    if test:
        text = get_test_text(example_id)
    else:
        text = get_raw_text(example_id)

    # map token ids to word (whitespace) token ids
    predstrings = []
    for span in all_span:
        span_start = span[0]
        span_end = span[1]
        before = text[:span_start]
        token_start = len(before.split())
        if len(before) == 0:
            token_start = 0
        elif before[-1] != ' ':
            token_start -= 1
        num_tkns = len(text[span_start:span_end + 1].split())
        tkns = [str(x) for x in range(token_start, token_start + num_tkns)]
        predstring = ' '.join(tkns)
        predstrings.append(predstring)

    rows = []
    for c, span, predstring in zip(classes, all_span, predstrings):
        e = {
            'id': example_id,
            'discourse_type': c,
            'predictionstring': predstring,
            'discourse_start': span[0],
            'discourse_end': span[1],
            'discourse': text[span[0]:span[1] + 1]
        }
        rows.append(e)

    df = pd.DataFrame(rows)
    df['length'] = df['discourse'].apply(lambda t: len(t.split()))

    # short spans are likely to be false positives, we can choose a min number of tokens based on validation
    df = df[df.length > min_tokens].reset_index(drop=True)

    return df

# Load test data
files = os.listdir('../data/test')
ids = [x.split('.')[0] for x in files]

df_test = pd.DataFrame()
df_test['id'] = ids
df_test['text'] = df_test['id'].apply(get_test_text)

print(df_test.head(10))

test_ds = Dataset.from_pandas(df_test)
print(test_ds)

def tokenize_for_test(examples):
    o = tokenizer(examples['text'], truncation=True, return_offsets_mapping=True, max_length=4096)

    return o

tokenized_test = test_ds.map(tokenize_for_test)
print(tokenized_test)

predictions, _, _ = trainer.predict(tokenized_test)

preds = np.argmax(predictions, axis=-1)
print(predictions.shape)
print(preds.shape)


dfs = []
for i in range(len(tokenized_test)):
    dfs.append(pred2span(preds[i], tokenized_test[i], test=True))

pred_df = pd.concat(dfs, axis=0)
pred_df['class'] = pred_df['discourse_type']

sub = pred_df[['id', 'class', 'predictionstring']]

sub.to_csv('fblongformerlarge1536.csv', index=False)

