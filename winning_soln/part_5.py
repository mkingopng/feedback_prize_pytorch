"""

"""

import numpy as np
import pandas as pd
import pickle
import os
import math
import re
from numba import jit
from tqdm import tqdm

data_path = 'data'

train_df = pd.read_csv('data/train.csv')
IDS = train_df.id.unique()

kfold_ids = pickle.load(open('data/feedback-two-stage-data/kfold_ids.pkl', 'rb'))

id2label = {0: 'Lead', 1: 'Position', 2: 'Evidence', 3: 'Claim', 4: 'Concluding Statement',
            5: 'Counterclaim', 6: 'Rebuttal', 7: 'blank'}
label2id = {v: k for k, v in id2label.items()}


class CONFIG:
    def __init__(self):
        self.max_length = 4096


config = CONFIG()

data_pred = pickle.load(open('data/feedback-two-stage-data/feedback-lb704.pkl', 'rb'))
data_pred = pd.DataFrame(data_pred, columns=['id', 'text', 'input_ids', 'attention_mask', 'token_label',
                                             'offset_mapping', 'kfold', 'pred'])

dic_off_map = data_pred[['id', 'offset_mapping']].set_index('id')['offset_mapping'].to_dict()
dic_txt = data_pred[['id', 'text']].set_index('id')['text'].to_dict()

"""
ori class order
{'O': 0,
 'I-Claim': 1,
 'I-Evidence': 2,
 'I-Position': 3,
 'I-Concluding Statement': 4,
 'I-Lead': 5,
 'I-Counterclaim': 6,
 'I-Rebuttal': 7,
 'B-Claim': 8,
 'B-Evidence': 9,
 'B-Position': 10,
 'B-Concluding Statement': 11,
 'B-Lead': 12,
 'B-Counterclaim': 13,
 'B-Rebuttal': 14}
"""


def change_label(x):
    """
    change N*15 preds to N*1 + N*8 preds
    """
    res1 = x[:, 8:].sum(axis=1)
    res2 = np.zeros((len(res1), 8))

    # change order, If it is in the same order as id2label, delete it
    label_map = {0: 5, 1: 3, 2: 2, 3: 1, 4: 4, 5: 6, 6: 7, 7: 0}
    for i in range(8):
        if i == 7:
            res2[:, i] = x[:, label_map[i]]
        else:
            res2[:, i] = x[:, [label_map[i], label_map[i] + 7]].sum(axis=1)

    return res1, res2


preds1_mean = {}
preds2_mean = {}
for irow, row in data_pred.iterrows():
    t1, t2 = change_label(row.pred)
    preds1_mean[row.id] = t1.astype('float64')
    preds2_mean[row.id] = t2.astype('float64')

recall_thre = {
    "Lead": 0.07,
    "Position": 0.06,
    "Evidence": 0.07,
    "Claim": 0.06,
    "Concluding Statement": 0.07,
    "Counterclaim": 0.03,
    "Rebuttal": 0.02,
}

L_k = {
    "Evidence": 0.85,
    "Rebuttal": 0.6,
}


def deal_predictionstring(df):
    """
    select sample with high boundary threshold and
    choice 65% length with the highest probability of the current class as a new sample
    """
    new_predictionstring = []
    new_pos_list = []
    flag_list = []
    thre = 0.8
    for id, typ, pos, (start, end) in df.values:
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

        new_predictionstring.append([start_word, end_word])
        new_pos_list.append(new_pos)
        flag_list.append(flag)

    df_new = df.copy()
    df_new['pos'] = new_pos_list
    df_new['predictionstring'] = new_predictionstring
    df_new['flag'] = flag_list

    df_new = pd.concat([df_new, df.loc[df_new[(df_new.flag >= 0.8) & (df_new.flag < 0.95)].index]])
    df_new = df_new.reset_index(drop=True)
    df_new['flag'].fillna(0, inplace=True)

    return df_new


def get_recall(id):
    all_predictions = []

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
            if pred1_np[i_start] > thre and pred2_np[i_start:i_start + 10].max() > thre:
                i = i_start + 1
                if i >= max_length:
                    break
                while pred1_np[i] < (1 - thre) and pred2_np[i:i + 10].max() > thre:
                    cond = any([
                        i + 1 == max_length,
                        pred1_np[i] > thre,
                        i + 1 < max_length and pred2_np[i] < 0.6 and pred2_np[i] - pred2_np[i + 1] > thre
                    ])
                    if i > i_start + 1 and cond:
                        all_predictions.append((id, id2label[class_num], [i_start, i]))
                    i += 1
                    if i >= max_length:
                        break

            if i != 0:
                if i == max_length:
                    i -= 1

                all_predictions.append((id, id2label[class_num], [i_start, i]))
            i_start += 1

    df_recall = pd.DataFrame(all_predictions, columns=['id', 'class', 'pos'])

    predictionstring = []
    for cache in df_recall.values:
        id = cache[0]
        pos = cache[2]
        off_map = dic_off_map[id]
        txt = dic_txt[id]
        txt_max = len(txt.split())

        start_word = len(txt[:off_map[pos[0]][0]].split())

        L = len(txt[off_map[pos[0]][0]:off_map[pos[1]][1]].split())
        end_word = min(txt_max, start_word + L) - 1

        predictionstring.append([start_word, end_word])

    df_recall['predictionstring'] = predictionstring

    return deal_predictionstring(df_recall)


#     return df_recall

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


def tuple_map(offset_mapping, threshold):
    paragraph_rk = []
    rk = 0
    last = 1
    for token_index in offset_mapping:
        if len(threshold) == 0:
            paragraph_rk.append(1)
        elif token_index[1] <= threshold[rk][1]:
            last = max(rk + 1, last)
            paragraph_rk.append(last)
        else:
            last = max(rk + 2, last)
            paragraph_rk.append(last)
            if rk + 1 < len(threshold) - 1:
                rk += 1

    return paragraph_rk


def get_pos_feat(text, offset_mapping):
    paragraph_cnt = len(text.split('\n\n')) + 1

    paragraph_th = [m.span() for m in re.finditer('\n\n', text)]
    paragraph_rk = tuple_map(offset_mapping, paragraph_th)

    paragraph_rk_r = [paragraph_cnt - rk + 1 if rk != 0 else 0 for rk in paragraph_rk]

    sentence_th = []
    for i, v in enumerate([m.span() for m in re.finditer('\n\n|\.|,|\?|\!', text)]):
        if i == 0:
            sentence_th.append(list(v))
        else:
            if v[0] == sentence_th[-1][-1]:
                sentence_th[-1][-1] = v[-1]
            else:
                sentence_th.append(list(v))
    sentence_cnt = len(sentence_th) + 1

    sentence_rk = tuple_map(offset_mapping, sentence_th)
    sentence_rk_r = [sentence_cnt - rk + 1 if rk != 0 else 0 for rk in sentence_rk]

    last_garagraph_cnt = 0
    sentence_rk_of_paragraph = []
    for i in range(len(offset_mapping)):
        sentence_rk_of_paragraph.append(sentence_rk[i] - last_garagraph_cnt)
        if i + 1 == len(offset_mapping) or paragraph_rk[i] != paragraph_rk[i + 1]:
            last_garagraph_cnt = sentence_rk[i]

    sentence_cnt_of_paragraph = []
    last_max = None
    for i in range(1, len(offset_mapping) + 1):
        if i == 1 or paragraph_rk[-i] != paragraph_rk[-i + 1]:
            last_max = sentence_rk_of_paragraph[-i]
        sentence_cnt_of_paragraph.append(last_max)
    sentence_cnt_of_paragraph = sentence_cnt_of_paragraph[::-1]

    sentence_rk_r_of_paragraph = [s_cnt - rk + 1 if rk != 0 else 0 for s_cnt, rk in
                                  zip(sentence_cnt_of_paragraph, sentence_rk_of_paragraph)]

    return paragraph_cnt, sentence_cnt, paragraph_rk, paragraph_rk_r, sentence_rk, sentence_rk_r, \
           sentence_cnt_of_paragraph, sentence_rk_of_paragraph, sentence_rk_r_of_paragraph


lgb_columns = pickle.load(open('../input/feedback-two-stage-data/lgb_columns.pkl', 'rb'))


def fun_get_feat(id):
    df_feat = []

    data_sub = get_recall(id)
    txt = dic_txt[id]
    off_map = dic_off_map[id]
    txt_feat = get_pos_feat(txt, off_map)

    preds1_all = preds1_mean[id]
    preds_type = preds2_mean[id].argmax(axis=-1)

    text_char_length = len(txt)
    text_word_length = len(txt.split())
    text_token_length = len(off_map)
    for cache in data_sub.values:
        id = cache[0]
        typ = cache[1]
        start, end = cache[2]
        prediction = cache[3]

        dic = {k: np.nan for k in lgb_columns}
        #         dic={'id': id}
        dic['id'] = id
        dic['pos'] = cache[2]
        dic['class'] = label2id[typ]
        dic['post_flag'] = cache[4]

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
        preds2_all = preds2_mean[id][:, label2id[typ]]
        preds4_all = preds2_mean[id][:, other_type].max(axis=-1)
        preds1 = preds1_all[start:end + 1]
        preds2 = preds2_all[start:end + 1]
        preds4 = preds4_all[start:end + 1]

        word_length = prediction[-1] - prediction[0] + 1

        dic['L1'] = word_length
        dic['L2'] = end - start + 1
        dic['text_char_length'] = text_char_length
        dic['text_word_length'] = text_word_length
        dic['text_token_length'] = text_token_length

        dic['word_start'] = prediction[0]
        dic['word_end'] = prediction[-1]
        dic['token_start'] = start
        dic['token_start2'] = start / text_token_length
        dic['token_end'] = end
        dic['token_end2'] = text_token_length - end
        dic['token_end3'] = end / text_token_length

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
                dic[f'before_other_type{i}'] = preds_type[start - i]

            if end + i < len(preds1_all):
                dic[f'after_head2_prob{i}'] = preds2_all[end + i]
                dic[f'after_other_prob{i}'] = preds4_all[end + i]
                dic[f'after_other_type{i}'] = preds_type[end + i]

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

    return df_feat


proba_thresh = {
    "Lead": 0.45,
    "Position": 0.4,
    "Evidence": 0.45,
    "Claim": 0.35,
    "Concluding Statement": 0.5,
    "Counterclaim": 0.35,
    "Rebuttal": 0.3,
}

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
    for k, group in df.groupby(['id', 'class']):
        group = group.sort_values('lgb_prob', ascending=False)

        preds_range = []
        for irow, row in group.iterrows():
            start = row.word_start
            end = row.word_end
            L1 = end - start + 1
            flag = 0
            for pos_range in preds_range:
                L2 = pos_range[1] - pos_range[0] + 1
                intersection = (min(end, pos_range[1]) - max(start, pos_range[0]) + 1) / L1
                inter_t = inter_thresh[row['class']]
                if intersection > inter_t and (inter_t <= L1 / L2 <= 1 or inter_t <= L2 / L1 <= 1):
                    flag = 1
                    break

            if flag == 0:
                preds_range.append((start, end, row.lgb_prob))
                rtn.append((row.id, row['class'], row.pos, row.word_start, row.word_end, row.lgb_prob))
    rtn = pd.DataFrame(rtn, columns=['id', 'class', 'pos', 'start', 'end', 'lgb_prob'])
    return rtn


fold = 0
lgb_model = pickle.load(open(f'../input/feedback-two-stage-data/lgb_fold{fold}.pkl', 'rb'))

sub = pd.DataFrame()
for id in tqdm(kfold_ids[fold][1]):
    df_feat = pd.DataFrame(fun_get_feat(id))

    lgb_preds = lgb_model.predict(df_feat.drop(['id', 'pos'], axis=1))

    df_final = df_feat[['id', 'class', 'pos', 'word_start', 'word_end']].copy()
    df_final['lgb_prob'] = lgb_preds
    df_final['class'] = df_final['class'].map(lambda x: id2label[x])

    df_final['thre'] = df_final['class'].map(lambda x: proba_thresh[x])
    df_final = df_final[df_final.lgb_prob >= df_final.thre]
    df_final = post_choice(df_final)

    sub = pd.concat([sub, df_final])


def get_predictionstring(df):
    predictionstring = []
    for cache in df.values:
        predictionstring.append(' '.join(list(map(str, range(cache[3], cache[4] + 1)))))
    return predictionstring


sub['predictionstring'] = get_predictionstring(sub)


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp_micro(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = (
        gt_df[["id", "discourse_type", "predictionstring"]]
            .reset_index(drop=True)
            .copy()
    )
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
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

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
            .sort_values("max_overlap", ascending=False)
            .groupby(["id", "predictionstring_gt"])
            .first()["pred_id"]
            .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    f1_score = TP / (TP + 0.5 * (FP + FN))
    precise_score = TP / (TP + FP)
    recall_score = TP / (TP + FN)

    return {'f1': f1_score, 'precise': precise_score, 'recall': recall_score}


def score_feedback_comp(pred_df, gt_df, return_class_scores=True):
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = (
            pred_df.loc[pred_df["class"] == discourse_type]
                .reset_index(drop=True)
                .copy()
        )
        class_scores[discourse_type] = score_feedback_comp_micro(pred_subset, gt_subset)
    f1 = np.mean([v['f1'] for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


train_oof = train_df[train_df.id.isin(kfold_ids[fold][1])]
score_feedback_comp(sub, train_oof)
