"""
bigbird is interesting because in theory it should outperform longformer by 1-3%. THis notebook doesn't ensemble like
Abishek's 2x longformers approach, but I think there is no reason why it can't be done.

There is NER post-processing in this notebook (I don't yet understand what that means), but as far as I can tell it
doesn't use spelling, punctuation and grammar correction.

I think potentially therefore there are 3x opportunities to improve this approach and beat the 2x longformer approach:
- ensemble 2x or more bigbird models
- use dictionary of corrected spelling, punctuation and grammar to generate a token
- improved NER processing
"""

import os
import gc
import numpy as np
from scipy import stats
from scipy.special import softmax
import pandas as pd
from tqdm import tqdm
import pickle

from transformers import *
from transformers import AutoTokenizer, AutoModelForTokenClassification

import torch
from torch import cuda
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score
from ast import literal_eval
import warnings
from collections import Counter
from joblib import Parallel, delayed
from multiprocessing import Manager
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from skopt.space import Real
from skopt import gp_minimize
import sys

warnings.filterwarnings('ignore', '.*__floordiv__ is deprecated.*', )

# DECLARE HOW MANY GPUS YOU WISH TO USE. KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0,1,2,3 for four gpu

# VERSION FOR SAVING MODEL WEIGHTS
VER = 1

# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
LOAD_TOKENS_FROM = f'py-bigbird-v{VER}'

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = 'whitespace'

# IF FOLLOWING IS NONE, THEN NOTEBOOK USES INTERNET AND DOWNLOADS HUGGINGFACE CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = f'py-bigbird-v{VER}'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'google/bigbird-roberta-base'

# Tune the probability threshold for sequence classifiers to maximize F1
TRAIN_SEQ_CLASSIFIERS = True

DATA_DIR = 'data'
NUM_FOLDS = 8

# A copy of my local cache for this notebook KAGGLE_CACHE = '../input/boostedcache'

CACHE = 'cache'
cacheExists = os.path.exists(CACHE)
if not cacheExists:
    os.makedirs(CACHE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {'model_name': MODEL_NAME,
          'max_length': 1024,
          'train_batch_size': 4,
          'valid_batch_size': 2,
          'epochs': 5,
          'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
          'max_grad_norm': 10,
          'device': 'cuda' if cuda.is_available() else 'cpu'}

if DOWNLOADED_MODEL_PATH == 'model':
    os.mkdir('model')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    tokenizer.save_pretrained('model')

    config_model = AutoConfig.from_pretrained(MODEL_NAME)
    config_model.NUM_LABELS = 15
    config_model.save_pretrained('model')

    backbone = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config_model)
    backbone.save_pretrained('model')

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
print(train_df.shape)
print(train_df.head())

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, test_texts = [], []
for f in list(os.listdir(os.path.join(DATA_DIR, 'test/'))):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open(os.path.join(DATA_DIR, 'test/') + f, 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})
test_texts.head()

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
    train_text_df = pd.read_csv(
        f'{LOAD_TOKENS_FROM}/train_NER.csv')  # pandas saves lists as string, we must convert back
    train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x))

print(train_text_df.shape)
train_text_df.head()

# create dictionaries that we can use during train and infer
output_labels = ['O',
                 'B-Lead',
                 'I-Lead',
                 'B-Position',
                 'I-Position',
                 'B-Claim',
                 'I-Claim',
                 'B-Counterclaim',
                 'I-Counterclaim',
                 'B-Rebuttal',
                 'I-Rebuttal',
                 'B-Evidence',
                 'I-Evidence',
                 'B-Concluding Statement',
                 'I-Concluding Statement']

labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}
disc_type_to_ids = {'Lead': (1, 2), 'Position': (3, 4), 'Claim': (5, 6), 'Counterclaim': (7, 8), 'Rebuttal': (9, 10),
                    'Evidence': (11, 12), 'Concluding Statement': (13, 14)}

print(labels_to_ids)


# Return an array that maps character index to index of word in list of split() words
def split_mapping(unsplit):
    splt = unsplit.split()
    offset_to_wordidx = np.full(len(unsplit), -1)
    txt_ptr = 0
    for split_index, full_word in enumerate(splt):
        while unsplit[txt_ptr:txt_ptr + len(full_word)] != full_word:
            txt_ptr += 1
        offset_to_wordidx[txt_ptr:txt_ptr + len(full_word)] = split_index
        txt_ptr += len(full_word)
    return offset_to_wordidx


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, get_wids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids  # for validation

    def __getitem__(self, index):
        # get text and word labels
        text = self.data.text[index]
        word_labels = self.data.entities[index] if not self.get_wids else None

        # tokenize text
        encoding = self.tokenizer(text,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        word_ids = encoding.word_ids()
        split_word_ids = np.full(len(word_ids), -1)
        offset_to_wordidx = split_mapping(text)
        offsets = encoding['offset_mapping']

        # create targets and mapping of tokens to split() words
        label_ids = []
        # Iterate in reverse to label whitespace tokens until a Begin token is encountered
        for token_idx, word_idx in reversed(list(enumerate(word_ids))):

            if word_idx is None:
                if not self.get_wids: label_ids.append(-100)
            else:
                if offsets[token_idx] != (0, 0):
                    # Choose the split word that shares the most characters with the token if any
                    split_idxs = offset_to_wordidx[offsets[token_idx][0]:offsets[token_idx][1]]
                    split_index = stats.mode(split_idxs[split_idxs != -1]).mode[0] if len(
                        np.unique(split_idxs)) > 1 else split_idxs[0]

                    if split_index != -1:
                        if not self.get_wids:
                            label_ids.append(labels_to_ids[word_labels[split_index]])
                        split_word_ids[token_idx] = split_index
                    else:
                        # Even if we don't find a word, continue labeling 'I' tokens until a 'B' token is found
                        if label_ids and label_ids[-1] != -100 and ids_to_labels[label_ids[-1]][0] == 'I':
                            split_word_ids[token_idx] = split_word_ids[token_idx + 1]
                            if not self.get_wids:
                                label_ids.append(label_ids[-1])
                        else:
                            if not self.get_wids:
                                label_ids.append(-100)
                else:
                    if not self.get_wids:
                        label_ids.append(-100)

        encoding['labels'] = list(reversed(label_ids))

        # convert to torch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids:
            item['wids'] = torch.as_tensor(split_word_ids)
        return item

    def __len__(self):
        return self.len


# choose validation indexes (that match TF notebook)
IDS = train_df.id.unique()
print('There are', len(IDS), 'train texts. We will split 90% 10% for validation.')

# train valid split 90% 10%
np.random.seed(42)
train_idx = np.random.choice(np.arange(len(IDS)), int(0.9 * len(IDS)), replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)), train_idx)

np.random.seed(None)

# create train subset and valid subset
data = train_text_df[['id', 'text', 'entities']]
train_dataset = data.loc[data['id'].isin(IDS[train_idx]), ['text', 'entities']].reset_index(drop=True)
test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH)
training_set = dataset(train_dataset, tokenizer, config['max_length'], False)
testing_set = dataset(test_dataset, tokenizer, config['max_length'], True)

# missing py-bigbird-v1/added_tokens.json. why?

# train dataset and valid dataset
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 2,
                'pin_memory': True
                }

test_params = {'batch_size': config['valid_batch_size'],
               'shuffle': False,
               'num_workers': 2,
               'pin_memory': True
               }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# test dataset
test_texts_set = dataset(test_texts, tokenizer, config['max_length'], True)
test_texts_loader = DataLoader(test_texts_set, **test_params)

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # tr_preds, tr_labels = [], []

    model.train()  # put model in training mode

    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
        labels = batch['labels'].to(config['device'], dtype=torch.long)
        with amp.autocast():
            loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=False)
        tr_loss += loss.item()
        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        if idx % 200 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss after {idx:04d} training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.NUM_LABELS)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
        # active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        # tr_labels.extend(labels)
        # tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

# create model
scaler = amp.GradScaler()
config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH + '/config.json')
model = AutoModelForTokenClassification.from_pretrained(
    DOWNLOADED_MODEL_PATH + '/pytorch_model.bin', config=config_model)
model.to(config['device'])
optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rates'][0])

# loop to train model (or load model)
if not LOAD_MODEL_FROM:
    for epoch in range(config['epochs']):

        print(f"### Training epoch: {epoch + 1}")
        for g in optimizer.param_groups:
            g['lr'] = config['learning_rates'][epoch]
        lr = optimizer.param_groups[0]['lr']
        print(f'### LR = {lr}\n')

        train(epoch)
        torch.cuda.empty_cache()
        gc.collect()

    torch.save(model.state_dict(), f'bigbird_v{VER}.pt')
else:
    model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/bigbird_v{VER}.pt', map_location=config['device']))
    print('Model loaded.')

# error related to no such file or directory. I've downloaded the whitespace/bigbird_v26.pt file from kaggle but the q
# is where is this created?

# If prob=True, returns per-word, mean class prediction probability over all tokens corresponding to each word
def inference(batch, prob=False):
    model.eval()  # put model in training mode
    ids = batch["input_ids"].to(config['device'])  # move batch to GPU and infer
    mask = batch["attention_mask"].to(config['device'])
    outputs = model(ids, attention_mask=mask, return_dict=False)
    if not prob:
        all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()
    else:
        all_preds = outputs[0].cpu().detach().numpy()

    # iterate through each text and get pred
    predictions = []
    for k, text_preds in enumerate(all_preds):
        if not prob:
            token_preds = [ids_to_labels[i] for i in text_preds]
        else:
            token_preds = text_preds
        prediction = []
        word_ids = batch['wids'][k].numpy()
        previous_word_idx = -1
        prob_buffer = []
        for idx, word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:
                if not prob:
                    prediction.append(token_preds[idx])
                else:
                    if prob_buffer:
                        prediction.append(np.mean(prob_buffer, axis=0))
                        prob_buffer = []
                    prob_buffer.append(softmax(token_preds[idx]))
                previous_word_idx = word_idx
            elif prob:
                prob_buffer.append(softmax(token_preds[idx]))
        if prob:
            prediction.append(np.mean(prob_buffer, axis=0))
        predictions.append(prediction)
    torch.cuda.empty_cache()
    return predictions


# from Rob Mulla @robikscube https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):
    """
    Calculates the overlap between prediction and ground truth and overlap percentages used for determining true
    positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    len_gt = len(set_gt)  # Length of each and intersection
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle Student Writing Competition
    Uses the steps in the evaluation page here: https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id', 'class', 'predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id', 'class'],
                           right_on=['id', 'discourse_type'],
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')
    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,and the overlap between the prediction and the
    # ground truth >= 0.5, the prediction is a match and considered a true positive. If multiple matches exist, the
    # match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])
    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id', 'predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]
    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]
    tp = len(tp_pred_ids)  # Get numbers of each type
    fp = len(fp_pred_ids)
    fn = len(unmatched_gt_ids)
    my_f1_score = tp / (tp + 0.5 * (fp + fn))  # calc microf1
    return my_f1_score


valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

print('Predicting with BigBird...')
try:
    with open('cache' + "/valid_preds.p", "rb") as validFile:
        valid_word_preds = pickle.load(validFile)
except:
    valid_word_preds = []
    for btch in testing_loader:
        valid_word_preds.extend(inference(btch, prob=True))
test_word_preds = []
for btch in test_texts_loader:
    test_word_preds.extend(inference(btch, prob=True))

with open(CACHE + "/valid_preds.p", "wb") as validFile:
    pickle.dump(valid_word_preds, validFile)
print('Done.')

uniqueValidGroups = range(len(valid_word_preds))
uniqueSubmitGroups = range(len(test_word_preds))

# Values are 98 Percentile split() length from this notebook
# https://www.kaggle.com/vuxxxx/tensorflow-longformer-ner-postprocessing
MAX_SEQ_LEN = {
    'Claim': 45,
    'Concluding Statement': 149,
    'Counterclaim': 68,
    'Evidence': 221,
    'Lead': 145,
    'Position': 47,
    'Rebuttal': 87,
}

# The minimum probability prediction for a 'B'egin class for which we will evaluate a word sequence
MIN_BEGIN_PROB = {
    'Claim': .35,
    'Concluding Statement': .35,
    'Counterclaim': .01,
    'Evidence': .35,
    'Lead': .35,
    'Position': .35,
    'Rebuttal': .01,
}


class SeqDataset(object):

    def __init__(self, features, labels, groups, wordRanges, truePos):
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.groups = np.array(groups)
        self.wordRanges = np.array(wordRanges)
        self.truePos = np.array(truePos)


def seq_dataset(disc_type, pred_indices=None, submit=False):
    word_preds = valid_word_preds if not submit else test_word_preds
    window = pred_indices if pred_indices else range(len(word_preds))
    X = []
    y = []
    true_pos = []
    word_ranges = []
    groups = []
    for text_i in tqdm(window):
        text_preds = np.array(word_preds[text_i])
        num_words = len(text_preds)
        disc_begin, disc_inside = disc_type_to_ids[disc_type]

        # The probability that a word corresponds to either a 'B'-egin or 'I'-nside token for a class
        prob_or = lambda word_preds: (1 - (1 - word_preds[:, disc_begin]) * (1 - word_preds[:, disc_inside]))

        if not submit:
            gt_idx = set()
            gt_arr = np.zeros(num_words, dtype=int)
            text_gt = valid.loc[valid.id == test_dataset.id.values[text_i]]
            disc_gt = text_gt.loc[text_gt.discourse_type == disc_type]

            # Represent the discourse instance locations in a hash set and an integer array for speed
            for row_i, row in enumerate(disc_gt.iterrows()):
                splt = row[1]['predictionstring'].split()
                start, end = int(splt[0]), int(splt[-1]) + 1
                gt_idx.add((start, end))
                gt_arr[start:end] = row_i + 1
            gt_lens = np.bincount(gt_arr)

        # Iterate over every sub-sequence in the text
        for pred_start in range(num_words):
            for pred_end in range(pred_start + 1, min(num_words + 1, pred_start + MAX_SEQ_LEN[disc_type] + 1)):

                # Generate features for a word sub-sequence
                if text_preds[pred_start, disc_begin] > MIN_BEGIN_PROB[disc_type]:
                    begin_or_inside = prob_or(text_preds[pred_start:pred_end])
                    features = [pred_end - pred_start]
                    features.extend(list(np.quantile(begin_or_inside, np.linspace(0, 1, 5))))

                    # The probability that words on either edge of the current sub-sequence belong to the class of interest
                    features.append(prob_or(text_preds[pred_start - 1:pred_start])[0] if pred_start > 0 else 0)
                    features.append(prob_or(text_preds[pred_end:pred_end + 1])[0] if pred_end < num_words else 0)

                    # The probability that the first word corresponds to a 'B'-egin token
                    features.append(text_preds[pred_start, disc_begin])
                    exact_match = (pred_start, pred_end) in gt_idx if not submit else None
                    if not submit:
                        true_pos = False
                        for match_cand, count in Counter(gt_arr[pred_start:pred_end]).most_common(2):
                            if match_cand != 0 and count / float(pred_end - pred_start) >= .5 and float(count) / \
                                    gt_lens[match_cand] >= .5:
                                true_pos = True
                    else:
                        true_pos = None
                    X.append(features)
                    y.append(exact_match)
                    true_pos.append(true_pos)
                    word_ranges.append((pred_start, pred_end))
                    groups.append(text_i)
    return SeqDataset(X, y, groups, word_ranges, true_pos)


manager = Manager()


def sequence_dataset(disc_type, submit=False):
    if not submit:
        validSeqSets[disc_type] = seq_dataset(disc_type)
    else:
        submitSeqSets[disc_type] = seq_dataset(disc_type, submit=True)


try:
    with open(CACHE + "/valid_seqds.p", "rb") as validFile:
        validSeqSets = pickle.load(validFile)
except:
    print('Making validation sequence datasets...')
    validSeqSets = manager.dict()
    Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(sequence_dataset)(disc_type, False)
        for disc_type in disc_type_to_ids
    )
    print('Done.')

print('Making submit sequence datasets...')
submitSeqSets = manager.dict()
Parallel(n_jobs=-1, backend='multiprocessing')(
    delayed(sequence_dataset)(disc_type, True)
    for disc_type in disc_type_to_ids
)
print('Done.')

with open(CACHE + "/valid_seqds.p", "wb") as validFile:
    pickle.dump(dict(validSeqSets), validFile)


warnings.filterwarnings('ignore', '.*ragged nested sequences*', )

prob_cache = {}  # Cache each fold's probability predictions for speed
clfs = []  # Each fold will add its classifier here


# Predict sub-sequences for a discourse type and set of train/test texts
def predict_strings(disc_type, probThresh, test_groups, train_ind=None, submit=False):
    string_preds = []
    validSeqDs = validSeqSets[disc_type]
    submitSeqDs = submitSeqSets[disc_type]

    # Average the probability predictions of a set of classifiers
    get_tp_prob = lambda testDs, classifiers: np.mean([clf.predict_proba(testDs.features)[:, 1] for clf in classifiers],
                                                      axis=0) if testDs.features.shape[0] > 0 else np.array([])

    if not submit:
        # Point to validation set values
        predict_df = test_dataset
        text_df = train_text_df
        groupIdx = np.isin(validSeqDs.groups, test_groups)
        testDs = SeqDataset(validSeqDs.features[groupIdx], validSeqDs.labels[groupIdx], validSeqDs.groups[groupIdx],
                            validSeqDs.wordRanges[groupIdx], validSeqDs.truePos[groupIdx])

        # Cache the classifier predictions to speed up tuning iterations
        prob_key = (disc_type, tuple(test_groups), tuple(train_ind))
        if prob_key in prob_cache:
            prob_tp = prob_cache[prob_key]
        else:
            clf = GradientBoostingClassifier()
            clf.fit(validSeqDs.features[train_ind], validSeqDs.truePos[train_ind])
            clfs.append(clf)
            prob_tp = get_tp_prob(testDs, [clf])
            prob_cache[prob_key] = prob_tp

    else:
        # Point to submission set values
        predict_df = test_texts
        text_df = test_texts
        groupIdx = np.isin(submitSeqDs.groups, test_groups)
        testDs = SeqDataset(submitSeqDs.features[groupIdx], submitSeqDs.labels[groupIdx], submitSeqDs.groups[groupIdx],
                            submitSeqDs.wordRanges[groupIdx], submitSeqDs.truePos[groupIdx])

        # Classifiers are always loaded from disc during submission
        with open(f"seqclassifiers/{disc_type}_clf.p", "rb") as clfFile:
            classifiers = pickle.load(clfFile)
        prob_tp = get_tp_prob(testDs, classifiers)

    for text_idx in test_groups:
        # The probability of true positive and (start,end) of each sub-sequence in the curent text
        prob_tp_curr = prob_tp[testDs.groups == text_idx]
        word_ranges_curr = testDs.wordRanges[testDs.groups == text_idx]

        i = 1
        split_text = text_df.loc[text_df.id == predict_df.id.values[text_idx]].iloc[0].text.split()
        full_preds = np.zeros(len(split_text))
        # Include the sub-sequence predictions in order of predicted probability
        for prob, wordRange in reversed(sorted(zip(prob_tp_curr, [tuple(wr) for wr in word_ranges_curr]))):

            # Until the predicted probability is lower than the tuned threshold
            if prob < probThresh: break

            # Add the sub-sequence if it does not intersect with previously added sub-sequences
            if not np.any(full_preds[wordRange[0]:wordRange[1]]):
                full_preds[wordRange[0]:wordRange[1]] = i
                string_preds.append((predict_df.id.values[text_idx], disc_type,
                                     ' '.join(map(str, list(range(wordRange[0], wordRange[1]))))))
                i += 1
    return string_preds


def sub_df(string_preds):
    return pd.DataFrame(string_preds, columns=['id', 'class', 'predictionstring'])


# Convert skopt's uniform distribution over the tuning threshold to a distribution that exponentially decays from 100% to 0%
def prob_thresh(x):
    return .01 * (100 - np.exp(100 * x))


# Convert back to the scalar supplied by skopt
def skopt_thresh(x):
    return np.log((x / .01 - 100.) / -1.) / 100.


# This function is called every tuning iteration. It takes the probability threshold as input and returns Macro F1
def score_fmin(arr, disc_type):
    validSeqDs = validSeqSets[disc_type]
    string_preds = []
    folds = np.array(list(GroupKFold(n_splits=NUM_FOLDS).split(validSeqDs.features, groups=validSeqDs.groups)))
    gt_indices = []
    for ind in folds[:, 1]: gt_indices.extend(ind)

    # Texts that have no samples in our dataset for this class
    unsampled_texts = np.array(
        np.array_split(list(set(uniqueValidGroups).difference(set(np.unique(validSeqDs.groups)))), NUM_FOLDS))

    gt_texts = test_dataset.id.values[np.unique(validSeqDs.groups[np.array(gt_indices, dtype=int)]).astype(int)]

    # Generate predictions from each fold of the validation set
    for fold_i, (train_ind, test_ind) in enumerate(folds):
        string_preds.extend(predict_strings(disc_type, prob_thresh(arr[0]), np.concatenate(
            (np.unique(validSeqDs.groups[test_ind]), unsampled_texts[fold_i])), train_ind))
    boost_df = sub_df(list(string_preds))
    gt_df = valid.loc[np.bitwise_and(valid['discourse_type'] == disc_type, valid.id.isin(gt_texts))].copy()
    f1 = score_feedback_comp(boost_df.copy(), gt_df)
    return -f1


def train_seq_clfs(disc_type):
    # The optimization bounds on the tuned probability threshold
    space_start = skopt_thresh(.999)
    space_end = skopt_thresh(0)
    space = [Real(space_start, space_end)]

    # Minimize F1
    score_fmin_disc = lambda arr: score_fmin(arr, disc_type)
    res_gp = gp_minimize(score_fmin_disc, space, n_calls=60, x0=[skopt_thresh(.5)])

    # Use the gaussian approximation of f(threshold) -> F1 to select the minima
    thresh_cand = np.rot90([np.linspace(0, 1, 1000)])
    cand_scores = res_gp.models[-1].predict(thresh_cand)
    best_thresh_raw = space_start + (space_end - space_start) * thresh_cand[np.argmin(cand_scores)][0]
    best_thresh = prob_thresh(best_thresh_raw)
    exp_score = -np.min(cand_scores)

    # Make predictions at the inferred function minima
    pred_thresh_score = -score_fmin_disc([best_thresh_raw])

    # And the best iteration in the optimization run
    best_iter_score = -score_fmin_disc(res_gp.x)

    # Save the trained classifiers to disc
    with open(f"{disc_type}_clf.p", "wb") as clfFile:
        pickle.dump(clfs, clfFile)

    # Save the tuning run results to file
    with open(f"{disc_type}_res.p", "wb") as resFile:
        pickle.dump(
            {
                'pred_thresh': best_thresh,  # The location of the minimum of the gaussian function inferred by skopt
                'min_thresh': prob_thresh(res_gp.x[0]),  # The threshold which produces the best score
                'pred_score': exp_score,  # The minimum of the gaussian function inferred by skopt
                'min_score': best_iter_score,  # The best score in the tuning run
                'pred_thresh_score': pred_thresh_score  # The score produced by 'pred_thresh'
            },
            resFile
        )
    print('Done training', disc_type)


if TRAIN_SEQ_CLASSIFIERS:
    print('Training sequence classifiers... (This takes a long time.)')
    Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(train_seq_clfs)(disc_type)
        for disc_type in disc_type_to_ids
    )
    print('Done training all sequence classifiers.')


thresholds = {}
for disc_type in disc_type_to_ids:
    with open(f"seqclassifiers/{disc_type}_res.p", "rb") as res_file:
        train_result = pickle.load(res_file)
    thresholds[disc_type] = train_result['pred_thresh']
sub = pd.concat([sub_df(predict_strings(disc_type, thresholds[disc_type], uniqueSubmitGroups, submit=True))
                 for disc_type in disc_type_to_ids]).reset_index(drop=True)

sub.to_csv("submission.csv", index=False)

