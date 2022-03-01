"""

"""
import hf_config
from hf_config import *
import numpy as np
import pandas as pd
import os
from pathlib import Path
import spacy
from spacy import displacy
from pylab import cm, matplotlib
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification


def create_l2i(tags, classes):
    """

    :param tags:
    :param classes:
    :return:
    """
    for i, c in enumerate(classes):
        tags[f'B-{c}'] = i
        tags[f'I-{c}'] = i + len(classes)
    tags[f'O'] = len(classes) * 2
    tags[f'Special'] = -100
    l2i = dict(tags)
    return l2i


def create_i2l(l2i):
    """

    :param l2i:
    :return:
    """
    i2l = defaultdict()
    for k, v in l2i.items():
        i2l[v] = k
    i2l[-100] = 'Special'
    i2l = dict(i2l)
    return i2l


def create_n_labels(i2l):
    """

    :param i2l:
    :return:
    """
    n_labels = len(i2l) - 1  # not accounting for -100
    return n_labels


def get_raw_text(ids):
    """

    :param ids:
    :return:
    """
    with open(Config.TRAIN_DATA/f'{ids}.txt', 'r') as file:
        data = file.read()
    return data


def dataset(df):
    """

    :param df:
    :return:
    """
    ds = Dataset.from_pandas(df)
    datasets = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
    return datasets


def set_tokenizer(model_checkpoint):
    """

    :param model_checkpoint:
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    return tokenizer


def fix_beginnings(labels):
    """

    :param labels:
    :return:
    """
    for i in range(1, len(labels)):
        curr_lab = labels[i]
        prev_lab = labels[i-1]
        if curr_lab in range(7,14):
            if prev_lab != curr_lab and prev_lab != curr_lab - 7:
                labels[i] = curr_lab -7
    return labels


# tokenize and add labels
def tokenize_and_align_labels(examples, max_length, tokenizer, l2i, stride):
    """

    :param examples:
    :param max_length:
    :param tokenizer:
    :param l2i:
    :param stride:
    :return:
    """
    o = tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        return_offsets_mapping=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True
        )
    # Since one example might give us several features if it has a long context, we need a map from a feature to its
    # corresponding example. This key gives us just that.
    sample_mapping = o["overflow_to_sample_mapping"]
    # The offset mappings will give us a map from token to character position in the original context. This will help us
    # compute the start_positions and end_positions.
    offset_mapping = o["offset_mapping"]
    o["labels"] = []
    for i in range(len(offset_mapping)):
        sample_index = sample_mapping[i]
        labels = [l2i['O'] for i in range(len(o['input_ids'][i]))]
        for label_start, label_end, label in \
                list(zip(examples['starts'][sample_index], examples['ends'][sample_index],
                         examples['classlist'][sample_index])):
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]
                if token_start == label_start:
                    labels[j] = l2i[f'B-{label}']
                if token_start > label_start and token_end <= label_end:
                    labels[j] = l2i[f'I-{label}']
        for k, input_id in enumerate(o['input_ids'][i]):
            if input_id in [0, 1, 2]:
                labels[k] = -100
        labels = fix_beginnings(labels)
        o["labels"].append(labels)
    return o


def tokenize_datasets(datasets, batch_size):
    """

    :param datasets:
    :param batch_size:
    :return:
    """
    datasets.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=batch_size,
        remove_columns=datasets["train"].column_names
    )


def compute_metrics(p, i2l, metric):
    """

    :param p:
    :param i2l:
    :param metric:
    :return:
    """
    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [i2l[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [i2l[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def tokenize_for_validation(examples, tokenizer, l2i):
    """
    The offset mappings will give us a map from token to character position in the original context. This will
    help us compute the start_positions and end_positions.
    :param examples:
    :param tokenizer:
    :param l2i:
    :return:
    """
    o_mapping = tokenizer(examples['text'], truncation=True, return_offsets_mapping=True, max_length=4096)
    offset_mapping = o_mapping["offset_mapping"]
    o_mapping["labels"] = []
    for i in range(len(offset_mapping)):
        labels = [l2i['O'] for i in range(len(o_mapping['input_ids'][i]))]
        for label_start, label_end, label in \
                list(zip(examples['starts'][i], examples['ends'][i], examples['classlist'][i])):
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]
                if token_start == label_start:
                    labels[j] = l2i[f'B-{label}']
                if token_start > label_start and token_end <= label_end:
                    labels[j] = l2i[f'I-{label}']
        for k, input_id in enumerate(o_mapping['input_ids'][i]):
            if input_id in [0, 1, 2]:
                labels[k] = -100
        labels = fix_beginnings(labels)
        o_mapping["labels"].append(labels)
    return o_mapping


def ground_truth_for_validation(tokenized_val):
    """
    ground truth for validation
    :param tokenized_val:
    :return:
    """
    ground_truth_list = []
    for example in tokenized_val['test']:
        for c, p in list(zip(example['classlist'], example['predictionstrings'])):
            ground_truth_list.append({
                'id': example['id'],
                'discourse_type': c,
                'predictionstring': p,
            })
    ground_truth_df = pd.DataFrame(ground_truth_list)
    return ground_truth_df


def visualize(df, text, colors, train_df):
    """
    visualization with displacy
    :param df:
    :param text:
    :return:
    """
    ents = []
    example = df['id'].loc[0]
    for i, row in df.iterrows():
        ents.append(
            {'start': int(row['discourse_start']),
             'end': int(row['discourse_end']),
             'label': row['discourse_type']}
        )
    doc2 = {
        "text": text,
        "ents": ents,
        "title": example
    }
    options = {"ents": train_df.discourse_type.unique().tolist() + ['Other'], "colors": colors}
    displacy.render(doc2, style="ent", options=options, manual=True, jupyter=False)


# code that will convert our predictions into prediction strings, and visualize it at the same time
# this most likely requires some refactoring


def get_class(c, i2l):
    """

    :param c:
    :param i2l:
    :return:
    """
    if c == 14:
        return 'Other'
    else:
        return i2l[c][2:]


def pred2span(text, all_span, classes, predstrings, example_id, min_tokens, pred, example, viz=False, test=False):
    """

    :param text:
    :param all_span:
    :param classes:
    :param predstrings:
    :param example_id:
    :param min_tokens:
    :param pred:
    :param example:
    :param viz:
    :param test:
    :return:
    """
    # map token ids to word (whitespace) token ids
    predstrings = []
    for span in all_span:  # mk: problem with all_span
        span_start = span[0]
        span_end = span[1]
        before = text[:span_start]  # mk: problem with text
        token_start = len(before.split())
        if len(before) == 0:
            token_start = 0
        elif before[-1] != ' ':
            token_start -= 1
        num_tkns = len(text[span_start:span_end + 1].split())  # mk: problem with text
        tkns = [str(x) for x in range(token_start, token_start + num_tkns)]
        predstring = ' '.join(tkns)
        predstrings.append(predstring)

    rows = []
    for c, span, predstring in zip(classes, all_span, predstrings):  # mk: problems with classes, all_span, predstrings
        e = {
            'id': example_id,  # mk: example_id
            'discourse_type': c,
            'predictionstring': predstring,
            'discourse_start': span[0],
            'discourse_end': span[1],
            'discourse': text[span[0]:span[1] + 1]  # mk: text
        }
        rows.append(e)
    df = pd.DataFrame(rows)
    df['length'] = df['discourse'].apply(lambda t: len(t.split()))

    # short spans are likely to be false positives, we can choose a min number of tokens based on validation
    df = df[df.length > min_tokens].reset_index(drop=True)  # mk: problem here with min_tokens
    if viz:
        visualize(df, text, colors=hf_config.COLORS, train_df=Parameters.TRAIN_DF)  # mk: problem here with circular logic
    return df


def calc_overlap(row):
    """
    Calculates the overlap between prediction and ground truth and overlap percentages used for determining true
    positives.
    source: https://www.kaggle.com/robikscube/student-writing-competition-twitch#Competition-Metric-Code
    :param row:
    :return:
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
    A function that scores for the kaggle Student Writing Competition
    Uses the steps in the evaluation page here: https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    :param pred_df:
    :param gt_df:
    :return:
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

    # 2. If the overlap between the ground truth and prediction is >= 0.5, and the overlap between the prediction and
    # the ground truth >= 0.5, the prediction is a match and considered a true positive. If multiple matches exist, the
    # match with the highest pair of overlaps is taken.
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

    # 3. Any unmatched ground truths are false negatives and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]
    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    tp = len(tp_pred_ids)
    fp = len(fp_pred_ids)
    fn = len(unmatched_gt_ids)

    # calc microf1
    my_f1_score = tp / (tp + 0.5 * (fp + fn))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    """

    :param pred_df:
    :param gt_df:
    :param return_class_scores:
    :return:
    """
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = (
            pred_df.loc[pred_df["class"] == discourse_type]
            .reset_index(drop=True)
            .copy()
        )
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1