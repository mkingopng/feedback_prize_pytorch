"""
https://www.kaggle.com/code/wht1996/feedback-lgb-train
"""

import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import gc
import copy
import time
import random
import string
import json
import pickle
import re
import math
from numba import jit
import lightgbm as lgb
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from multiprocessing import Pool
from joblib import Parallel, delayed
from util import *
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

df = pickle.load(open('./data/data_6model_offline712_online704_ensemble.pkl', 'rb'))

train_df = pd.read_csv('./data/train.csv')
IDS = train_df.id.unique()

dic_off_map = df[['id', 'offset_mapping']].set_index('id')['offset_mapping'].to_dict()
dic_txt = df[['id', 'text']].set_index('id')['text'].to_dict()


class CONFIG:
    def __init__(self):
        self.max_length = 4096


config = CONFIG()

id2label = {0: 'Lead', 1: 'Position', 2: 'Evidence', 3: 'Claim', 4: 'Concluding Statement',
            5: 'Counterclaim', 6: 'Rebuttal', 7: 'blank'}
label2id = {v: k for k, v in id2label.items()}


def change_label(x):
    res1 = x[:, 8:].sum(axis=1)
    res2 = np.zeros((len(res1), 8))

    label_map = {0: 5, 1: 3, 2: 2, 3: 1, 4: 4, 5: 6, 6: 7, 7: 0}
    for i in range(8):
        if i == 7:
            res2[:, i] = x[:, label_map[i]]
        else:
            res2[:, i] = x[:, [label_map[i], label_map[i] + 7]].sum(axis=1)

    return res1, res2


preds1_mean = {}
preds2_mean = {}
for irow, row in df.iterrows():
    t1, t2 = change_label(row.pred)
    preds1_mean[row.id] = t1
    preds2_mean[row.id] = t2

all_predictions = []

recall_thre = {
    "Lead": 0.07,
    "Position": 0.06,
    "Evidence": 0.07,
    "Claim": 0.06,
    "Concluding Statement": 0.07,
    "Counterclaim": 0.03,
    "Rebuttal": 0.02,
}

for id in tqdm(preds1_mean):
    pred1_np = np.array(preds1_mean[id])
    pred2_np_all = np.array(preds2_mean[id])

    off_map = dic_off_map[id]
    off_map_len = len(off_map) if off_map[-1][1] != 0 else len(off_map) - 1
    max_length = min(config.max_length, off_map_len)
    for class_num in range(7):
        thre = recall_thre[id2label[class_num]]
        pred2_np = pred2_np_all[:, class_num]

        i_start = 0
        while i_start < max_length:
            i = 0
            if pred1_np[i_start] > thre and pred2_np[i_start:i_start + 10].max() > thre:  # 开头 两个阈值
                i = i_start + 1
                if i >= max_length: break
                while pred1_np[i] < (1 - thre) and pred2_np[i:i + 10].max() > thre:  # 是否结束 两个阈值
                    cond = any([
                        i + 1 == max_length,
                        pred1_np[i] > thre,
                        i + 1 < max_length and pred2_np[i] < 0.7 and pred2_np[i] - pred2_np[i + 1] > thre
                    ])
                    if i > i_start + 1 and cond:
                        all_predictions.append((id, id2label[class_num], [i_start, i]))
                    i += 1
                    if i >= max_length: break

            if i != 0:
                if i == max_length:
                    i -= 1

                all_predictions.append((id, id2label[class_num], [i_start, i]))
            i_start += 1

print(len(all_predictions))
valid_pred = pd.DataFrame(all_predictions, columns=['id', 'class', 'pos'])

predictionstring = []
for cache in tqdm(valid_pred.values):
    id = cache[0]
    pos = cache[2]
    off_map = dic_off_map[id]
    txt = dic_txt[id]
    txt_max = len(txt.split())

    start_word = len(txt[:off_map[pos[0]][0]].split())

    L = len(txt[off_map[pos[0]][0]:off_map[pos[1]][1]].split())
    end_word = min(txt_max, start_word + L) - 1

    predictionstring.append((start_word, end_word))

valid_pred['predictionstring'] = predictionstring

L_k = {
    "Evidence": 0.85,
    "Rebuttal": 0.6,
}


def deal_predictionstring(df):
    new_predictionstring = []
    new_pos_list = []
    flag_list = []
    thre = 0.75
    for id, typ, pos, (start, end) in tqdm(df.values):
        flag = 0
        L = round(max(1, (pos[1] - pos[0] + 1) * 0.25))

        pos_left = max(0, pos[0] - L)
        pos_right = min(len(preds1_mean[id]), pos[1] + 1 + L)

        if start < 10:
            left_thre = 2
        else:
            left_thre = max(preds1_mean[id][pos[0]], 1 - preds2_mean[id][pos_left:pos[0], label2id[typ]].min())

        if pos[1] >= len(preds1_mean[id]) - 10:
            right_thre = 2
        else:
            right_thre = max(preds1_mean[id][pos[1] + 1:pos_right].max(),
                             1 - preds2_mean[id][pos[1] + 1:pos_right, label2id[typ]].min())

        if left_thre > thre and right_thre > thre:

            L = math.ceil((pos[1] - pos[0] + 1) * L_k.get(typ, 0.65))

            tmp = {}
            for i in range(pos[0], pos[1]):
                if i + L > pos[1]:
                    break
                tmp[i] = np.sum(preds2_mean[id][i:i + L + 1, label2id[typ]])
            if len(tmp) == 0:
                new_pos = pos
            else:
                flag = min(left_thre, right_thre)

                new_start = max(tmp.keys(), key=lambda x: tmp[x])
                new_pos = (new_start, new_start + L)

        else:
            new_pos = pos

        off_map = dic_off_map[id]
        txt = dic_txt[id]
        txt_max = len(txt.split())

        start_word = len(txt[:off_map[new_pos[0]][0]].split())

        L = len(txt[off_map[new_pos[0]][0]:off_map[new_pos[1]][1]].split())
        end_word = min(txt_max, start_word + L) - 1

        new_predictionstring.append((start_word, end_word))
        new_pos_list.append(new_pos)
        flag_list.append(flag)

    df_new = df.copy()
    df_new['pos'] = new_pos_list
    df_new['predictionstring'] = new_predictionstring
    df_new['flag'] = flag_list

    df_new = pd.concat([df_new, df.loc[df_new[(df_new.flag >= thre) & (df_new.flag < 0.95)].index]])
    df_new = df_new.reset_index(drop=True)
    df_new['flag'].fillna(0, inplace=True)

    return df_new


valid_pred = deal_predictionstring(valid_pred)
valid_oof = train_df.copy()
tmp = valid_oof.predictionstring.map(lambda x: x.split())
tmp1 = [(int(x[0]), int(x[-1])) for x in tmp]
valid_oof['predictionstring'] = tmp1


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    try:
        start_pred, end_pred = row.predictionstring_pred
        start_gt, end_gt = row.predictionstring_gt
    except:
        return [0, 0]

    # Length of each and intersection
    len_gt = end_gt - start_gt + 1
    len_pred = end_pred - start_pred + 1
    inter = min(end_pred, end_gt) - max(start_pred, start_gt) + 1
    overlap_1 = inter / (len_gt + 1e-5)
    overlap_2 = inter / (len_pred + 1e-5)
    return [overlap_1, overlap_2]


gt_df = (
    valid_oof[["id", "discourse_type", "predictionstring"]]
        .reset_index(drop=True)
        .copy()
)
pred_df = valid_pred[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
pred_df["pred_id"] = pred_df.index
gt_df["gt_id"] = gt_df.index
# Step 1. all ground truths and predictions for a given class are compared.
joined = pred_df.merge(
    gt_df,
    left_on=["id", "class"],
    right_on=["id", "discourse_type"],
    how="outer",
    suffixes=("_pred", "_gt"),
)
joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

joined["overlaps"] = joined.apply(calc_overlap, axis=1)
joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
joined["min_overlap"] = joined[["overlap1", "overlap2"]].min(axis=1)

valid_pred['label'] = 0
valid_true_id = joined[joined.potential_TP == True]['pred_id']

valid_pred.loc[valid_true_id, 'label'] = 1

overlap = joined[['pred_id', 'min_overlap']]
overlap = overlap[~ overlap.pred_id.isna()]
overlap = overlap.groupby('pred_id')['min_overlap'].max().reset_index()

valid_pred = valid_pred.merge(overlap, left_index=True, right_on='pred_id', how='left')
valid_pred = valid_pred.drop('pred_id', axis=1)

pickle.dump(valid_pred, open('./data/recall_data.pkl', 'wb+'))

# lgb
# train

# !/usr/bin/env python
# coding: utf-8


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%m/%d %H:%M:%S")
logging.info("code begin!!!!!!!")

model_name = time.strftime('%d_%H_%M_%S_', time.localtime(time.time())) + str(random.randint(0, 1000))
logging.info(f"====== model_name: {model_name} ======")

num_jobs = 60

data_path = './data/'

train_df = pd.read_csv(data_path + 'train.csv')
IDS = train_df.id.unique()

id2label = {0: 'Lead', 1: 'Position', 2: 'Evidence', 3: 'Claim', 4: 'Concluding Statement',
            5: 'Counterclaim', 6: 'Rebuttal', 7: 'blank'}
label2id = {v: k for k, v in id2label.items()}

df_wf = pickle.load(open('./data/data_6model_offline712_online704_ensemble.pkl', 'rb'))

dic_txt_feat = pickle.load(open('./data/dic_txt_feat.pkl', 'rb'))
dic_off_map = df_wf[['id', 'offset_mapping']].set_index('id')['offset_mapping'].to_dict()
dic_txt = df_wf[['id', 'text']].set_index('id')['text'].to_dict()


def change_label(x):
    res1 = x[:, 8:].sum(axis=1)
    res2 = np.zeros((len(res1), 8))

    label_map = {0: 5, 1: 3, 2: 2, 3: 1, 4: 4, 5: 6, 6: 7, 7: 0}
    for i in range(8):
        if i == 7:
            res2[:, i] = x[:, label_map[i]]
        else:
            res2[:, i] = x[:, [label_map[i], label_map[i] + 7]].sum(axis=1)

    return res1, res2


preds1_5fold = {}
preds2_5fold = {}
for irow, row in df_wf.iterrows():
    t1, t2 = change_label(row.pred)
    preds1_5fold[row.id] = t1
    preds2_5fold[row.id] = t2

valid_pred = pickle.load(open('./data/recall_data.pkl', 'rb'))
kfold_ids = pickle.load(open('./data/kfold_ids.pkl', 'rb'))

logging.info(f'valid_pred num:{len(valid_pred)}')

preds2_5fold_type = {}
for k, t in preds2_5fold.items():
    preds2_5fold_type[k] = np.array(t).argmax(axis=-1)


@jit(nopython=True)
def feat_speedup(arr):
    r_max, r_min, r_sum = -1e5, 1e5, 0
    for x in arr:
        r_max = max(r_max, x)
        r_min = min(r_min, x)
        r_sum += x
    return r_max, r_min, r_sum, r_sum / len(arr)


np_lin = np.linspace(0, 1, 7)


@jit(nopython=True)
def sorted_quantile(array, q):
    n = len(array)
    index = (n - 1) * q
    left = int(index)
    fraction = index - left
    right = left
    right = right + int(fraction > 0)
    i, j = array[left], array[right]
    return i + (j - i) * fraction


def get_percentile(array):
    x = np.sort(array)
    n = len(x) - 1
    return x[[int(n * t) for t in np_lin[1:-1]]]


def fun_get_feat(data_sub):
    df_feat = []
    for cache in tqdm(data_sub):
        id = cache[0]
        typ = cache[1]
        start, end = cache[2]
        prediction = cache[3]

        dic = {'id': id, 'label': cache[5], 'label_rate': max(0, cache[6])}
        dic['class'] = label2id[typ]
        dic['post_flag'] = cache[4]
        #         dic['cluster'] = dic_cluster[id]

        txt = dic_txt[id]

        txt_feat = dic_txt_feat[id]
        dic['paragraph_cnt'] = txt_feat[0]
        dic['sentence_cnt'] = txt_feat[1]
        dic['paragraph_rk'] = txt_feat[2][start]
        dic['paragraph_rk_r'] = txt_feat[3][end]
        dic['sentence_rk'] = txt_feat[4][start]
        dic['sentence_rk_r'] = txt_feat[5][end]
        dic['sentence_cnt_of_paragraph'] = txt_feat[6][start]
        dic['sentence_cnt_of_paragraph2'] = txt_feat[6][end]
        dic['sentence_rk_of_paragraph'] = txt_feat[7][start]
        dic['sentence_rk_r_of_paragraph'] = txt_feat[8][end]
        dic['sub_paragraph_cnt'] = txt_feat[2][end] - txt_feat[2][start]
        dic['sub_sentence_cnt'] = txt_feat[4][end] - txt_feat[4][start]

        other_type = [t for t in range(8) if t != dic['class']]
        preds1_all = np.array(preds1_5fold[id])
        preds2_all = np.array(preds2_5fold[id])[:, label2id[typ]]
        preds4_all = np.array(preds2_5fold[id])[:, other_type].max(axis=-1)
        preds1 = preds1_all[start:end + 1]
        preds2 = preds2_all[start:end + 1]
        preds4 = preds4_all[start:end + 1]

        word_length = prediction[-1] - prediction[0] + 1
        token_length = len(dic_off_map[id])

        dic['L1'] = word_length
        dic['L2'] = end - start + 1
        dic['text_char_length'] = len(txt)
        dic['text_word_length'] = len(txt.split())
        dic['text_token_length'] = token_length

        dic['word_start'] = prediction[0]
        dic['word_end'] = prediction[-1]
        dic['token_start'] = start
        dic['token_start2'] = start / token_length
        dic['token_end'] = end
        dic['token_end2'] = token_length - end
        dic['token_end3'] = end / token_length

        dic[f'head_preds1'] = preds1[0]
        dic[f'head2_preds1'] = preds1_all[start - 1:start + 2].sum()
        if len(preds1) > 1:
            dic[f'tail_preds1'] = preds1[-1]
            dic['max_preds1'], dic['min_preds1'], dic['sum_preds1'], dic['mean_preds1'] = feat_speedup(preds1[1:])

        sort_idx = preds1[1:].argsort()[::-1]
        tmp = []
        for i in range(5):
            if i < len(sort_idx):
                dic[f'other_preds1_{i}'] = preds1[1 + sort_idx[i]]
                dic[f'other_preds1_idx_{i}'] = (1 + sort_idx[i]) / len(preds1)
                tmp.append(preds1[1 + sort_idx[i]])

        if len(tmp):
            dic[f'other_preds1_mean'] = np.mean(tmp)

        dic[f'head_preds2'] = preds2[0]
        dic[f'tail_preds2'] = preds2[-1]
        dic['max_preds2'], dic['min_preds2'], dic['sum_preds2'], dic['mean_preds2'] = feat_speedup(preds2)

        dic[f'head_preds4'] = preds4[0]
        dic[f'tail_preds4'] = preds4[-1]
        dic['max_preds4'], dic['min_preds4'], dic['sum_preds4'], dic['mean_preds4'] = feat_speedup(preds4)

        sort_idx = preds2.argsort()
        tmp = []
        for i in range(5):
            if i < len(sort_idx):
                dic[f'other_preds2_{i}'] = preds2[sort_idx[i]]
                dic[f'other_preds2_idx_{i}'] = (sort_idx[i]) / len(preds2)
                tmp.append(preds2[sort_idx[i]])
        if len(tmp):
            dic[f'other_preds2_mean'] = np.mean(tmp)

        for i, ntile in enumerate([sorted_quantile(preds2, i) for i in np_lin]):
            dic[f'preds2_trend{i}'] = ntile
        for i, ntile in enumerate(get_percentile(preds2)):
            dic[f'preds2_ntile{i}'] = ntile
        for i, ntile in enumerate([sorted_quantile(preds4, i) for i in np_lin]):
            dic[f'preds4_trend{i}'] = ntile
        for i, ntile in enumerate(get_percentile(preds4)):
            dic[f'preds4_ntile{i}'] = ntile

        for i in range(1, 4):
            if start - i >= 0:
                dic[f'before_head2_prob{i}'] = preds2_all[start - i]
                dic[f'before_other_prob{i}'] = preds4_all[start - i]
                dic[f'before_other_type{i}'] = preds2_5fold_type[id][start - i]

            if end + i < len(preds1_all):
                dic[f'after_head2_prob{i}'] = preds2_all[end + i]
                dic[f'after_other_prob{i}'] = preds4_all[end + i]
                dic[f'after_other_type{i}'] = preds2_5fold_type[id][end + i]

        for mode in ['before', 'after']:
            for iw, extend_L in enumerate([math.ceil(word_length / 2), word_length]):
                if mode == 'before':
                    if start - extend_L < 0:
                        continue
                    preds1_extend = preds1_all[start - extend_L:start]
                    preds2_extend = preds2_all[start - extend_L:start]
                else:
                    if end + extend_L >= len(preds1_all):
                        continue
                    preds1_extend = preds1_all[end + 1:end + extend_L]
                    preds2_extend = preds2_all[end + 1:end + extend_L]

                if len(preds1_extend) == 0:
                    continue
                dic[f'{mode}{iw}_head_preds1'] = preds1_extend[0]
                dic[f'{mode}{iw}_max_preds1'], dic[f'{mode}{iw}_min_preds1'], \
                dic[f'{mode}{iw}_sum_preds1'], dic[f'{mode}{iw}_mean_preds1'] = feat_speedup(preds1_extend)

                dic[f'{mode}{iw}_head_preds2'] = preds2_extend[0]
                dic[f'{mode}{iw}_max_preds2'], dic[f'{mode}{iw}_min_preds2'], \
                dic[f'{mode}{iw}_sum_preds2'], dic[f'{mode}{iw}_mean_preds2'] = feat_speedup(preds2_extend)

                dic[f'{mode}{iw}_sum_preds1_rate'] = dic[f'{mode}{iw}_sum_preds1'] / dic[f'sum_preds1']
                dic[f'{mode}{iw}_sum_preds2_rate'] = dic[f'{mode}{iw}_sum_preds2'] / dic[f'sum_preds2']
                dic[f'{mode}{iw}_max_preds1_rate'] = dic[f'{mode}{iw}_max_preds1'] / dic[f'max_preds1']
                dic[f'{mode}{iw}_max_preds2_rate'] = dic[f'{mode}{iw}_max_preds2'] / dic[f'max_preds2']

        df_feat.append(dic)

    save_path = './cache/' + '_'.join([cache[0], cache[1], str(cache[2])]) + '.pkl'
    pickle.dump(df_feat, open(save_path, 'wb+'))
    return save_path


#     return df_feat


data_splits = np.array_split(valid_pred.values, num_jobs)
results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
    delayed(fun_get_feat)(data_sub) for data_sub in data_splits
)

logging.info(f"====== load pickle ======")
df_feat = []
for path in tqdm(results):
    df_feat.extend(pickle.load(open(path, 'rb')))

df_feat = pd.DataFrame(df_feat)
logging.info(f"====== dataFrame ok ======")

params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    #           'objective': 'regression',
    #           'metric': {'l2'},
    'num_leaves': 15,
    'min_data_in_leaf': 30,
    'max_depth': 5,
    'learning_rate': 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    'min_data_in_bin': 15,
    #           "min_sum_hessian_in_leaf": 6,
    "lambda_l1": 5,
    'lambda_l2': 5,
    "random_state": 1996,
    "num_threads": num_jobs,
}

valid_pred = df_feat[['id', 'class', 'word_start', 'word_end']].copy()
valid_pred['class'] = valid_pred['class'].map(lambda x: id2label[x])
valid_pred['lgb_prob'] = -1
for fold in range(5):
    df_feat_train = df_feat[df_feat.id.isin(kfold_ids[fold][0])].copy()
    df_feat_val = df_feat[df_feat.id.isin(kfold_ids[fold][1])].copy()

    lgb_train = lgb.Dataset(df_feat_train.drop(['id', 'label', 'label_rate'], axis=1), label=df_feat_train['label'])
    lgb_val = lgb.Dataset(df_feat_val.drop(['id', 'label', 'label_rate'], axis=1), label=df_feat_val['label'])

    clf = lgb.train(params,
                    lgb_train,
                    10000,
                    valid_sets=[lgb_train, lgb_val],
                    verbose_eval=200,
                    early_stopping_rounds=100)

    lgb_preds = clf.predict(df_feat_val.drop(['id', 'label', 'label_rate'], axis=1))

    valid_pred.loc[df_feat_val.index, 'lgb_prob'] = lgb_preds

    pickle.dump(clf, open(f'./result/lgb_fold{fold}.pkl', 'wb+'))

assert len(valid_pred[valid_pred.lgb_prob == -1]) == 0

pickle.dump(valid_pred, open(f'./result/lgb_valid_pred.pkl', 'wb+'))
pickle.dump([t for t in list(df_feat.columns) if t not in ['id', 'label', 'label_rate']],
            open(f'./result/lgb_columns.pkl', 'wb+'))

inter_thresh = {
    "Lead": 0.15,
    "Position": 0.15,
    "Evidence": 0.15,
    "Claim": 0.25,
    "Concluding Statement": 0.15,
    "Counterclaim": 0.25,
    "Rebuttal": 0.25,
}


def post_choice(df):
    rtn = []
    for k, group in tqdm(df.groupby(['id', 'class'])):
        group = group.sort_values('lgb_prob', ascending=False)

        preds_range = []
        for irow, row in group.iterrows():
            start = row.word_start
            end = row.word_end
            L1 = end - start + 1
            flag = 0
            if L1 == 0:
                continue
            for pos_range in preds_range:
                L2 = pos_range[1] - pos_range[0] + 1
                intersection = (min(end, pos_range[1]) - max(start, pos_range[0]) + 1) / L1
                inter_t = inter_thresh[row['class']]
                if intersection > inter_t and (inter_t <= L1 / L2 <= 1 or inter_t <= L2 / L1 <= 1):
                    flag = 1
                    break

            if flag == 0:
                preds_range.append((start, end, row.lgb_prob))

                predictionstring = ' '.join(list(map(str, range(int(row.word_start), int(row.word_end) + 1))))
                rtn.append((row.id, row['class'], predictionstring, row.lgb_prob))
    rtn = pd.DataFrame(rtn, columns=['id', 'class', 'predictionstring', 'lgb_prob'])
    return rtn


valid_pred_choice = post_choice(valid_pred)

proba_thresh = {
    "Lead": 0.45,
    "Position": 0.4,
    "Evidence": 0.45,
    "Claim": 0.35,
    "Concluding Statement": 0.5,
    "Counterclaim": 0.3,
    "Rebuttal": 0.3,
}

train_oof = train_df.copy()
res = {}
for k, v in proba_thresh.items():
    sub = valid_pred_choice[(valid_pred_choice.lgb_prob > v) & (valid_pred_choice['class'] == k)]
    score_now = score_feedback_comp(sub, train_oof)[1][k]['f1']
    sub = valid_pred_choice[(valid_pred_choice.lgb_prob > v - 0.05) & (valid_pred_choice['class'] == k)]
    score_now1 = score_feedback_comp(sub, train_oof)[1][k]['f1']
    sub = valid_pred_choice[(valid_pred_choice.lgb_prob > v + 0.05) & (valid_pred_choice['class'] == k)]
    score_now2 = score_feedback_comp(sub, train_oof)[1][k]['f1']

    if max(score_now, score_now1, score_now2) == score_now:
        res[k] = (v, score_now)

    elif max(score_now, score_now1, score_now2) == score_now1:
        best_score = score_now1
        score_now3 = best_score
        i = 2
        while score_now3 >= best_score:
            sub = valid_pred_choice[(valid_pred_choice.lgb_prob > v - 0.05 * i) & (valid_pred_choice['class'] == k)]
            score_now3 = score_feedback_comp(sub, train_oof)[1][k]['f1']
            best_score = max(best_score, score_now3)
            i += 1
            if v - 0.05 * i <= 0:
                break
        res[k] = (v - 0.05 * (i - 2), best_score)

    elif max(score_now, score_now1, score_now2) == score_now2:
        best_score = score_now2
        score_now3 = best_score
        i = 2
        while score_now3 >= best_score:
            sub = valid_pred_choice[(valid_pred_choice.lgb_prob > v + 0.05 * i) & (valid_pred_choice['class'] == k)]
            score_now3 = score_feedback_comp(sub, train_oof)[1][k]['f1']
            best_score = max(best_score, score_now3)
            i += 1
            if v + 0.05 * i >= 1:
                break
        res[k] = (v + 0.05 * (i - 2), best_score)

for k, v in res.items():
    logging.info(f"{k}:{v}")
logging.info(f"====== final score: {np.mean([v[1] for v in res.values()])} ======")
