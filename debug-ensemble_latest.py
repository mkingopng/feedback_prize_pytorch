"""

"""
import gc
import numpy as np
import pandas as pd
import os
import transformers
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle
import optuna
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import glob

gc.enable()

# todo: refactor variable names to be more meaningful
#  turn model part into a function
#  separate functions from main script
#  separate variables into a config file
#  consider having separate variables for each model as they were trained on different parameters
#  the trial part can be accelerated using GPU. Its currently running on CPU
#  currently only outputting 5x weights. I've set n_folds incorrectly somehow. Need to change to generate weights for
#  all folds in all models, which sum to 1

# config
DATA_ROOT = "data"
MODEL_STORE = os.path.join('model_stores')
BATCH_SIZE = 8
NUM_WORKERS = 0
RANDOM_STATE = 42
MAX_LENGTH = 4096

pd.options.display.max_columns = 50
pd.options.display.width = 500
train_df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))


def apply_stratified_kfold_to_train_data():
    """

    :return:
    """
    df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))

    # WKNOTE: pd.get_dummies - pandas's one-hot encoder
    dfx = pd.get_dummies(df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()
    cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
    dfx = dfx[cols].copy()
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    labels = [c for c in dfx.columns if c != "id"]
    dfx_labels = dfx[labels]
    dfx["kfold"] = -1
    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        print(len(trn_), len(val_))
        dfx.loc[val_, "kfold"] = fold
    df = df.merge(dfx[["id", "kfold"]], on="id", how="left")
    print(df.kfold.value_counts())
    print(df.pivot_table(index='kfold', values='id', aggfunc=lambda x: len(np.unique(x))))
    print(df.pivot_table(index='kfold', columns='discourse_type', values='id', aggfunc=len))
    df.to_csv("train_folds.csv", index=False)
    return df


# load train data
if os.path.exists('train_folds.csv'):
    train_df = pd.read_csv('train_folds.csv')
else:
    train_df = apply_stratified_kfold_to_train_data()

# prepare label, as the label values when the model was trained
target_id_map = {
    "B-Lead": 0,
    "I-Lead": 1,
    "B-Position": 2,
    "I-Position": 3,
    "B-Evidence": 4,
    "I-Evidence": 5,
    "B-Claim": 6,
    "I-Claim": 7,
    "B-Concluding Statement": 8,
    "I-Concluding Statement": 9,
    "B-Counterclaim": 10,
    "I-Counterclaim": 11,
    "B-Rebuttal": 12,
    "I-Rebuttal": 13,
    "O": 14,
    "PAD": -100,
}

id_target_map = {v: k for k, v in target_id_map.items()}


def prepare_samples(df, is_train: bool, tkz: transformers.models.longformer.tokenization_longformer_fast.LongformerTokenizerFast):
    """

    :param df:
    :param is_train:
    :param tkz:
    :return:
    """
    # prepare test data so that they can be processed by the model
    samples = []
    ids = df['id'].unique()
    train_or_test = 'train' if is_train else 'test'
    for idx in ids:
        filename = os.path.join(DATA_ROOT, train_or_test, f'{idx}.txt')
        with open(filename, 'r') as f:
            text = f.read()

        encoded_text = tkz.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)

        # WKNOTE: same effect as `tkz(text, add_special_tokens=False, return_offsets_mapping=True)`

        input_ids = encoded_text["input_ids"]
        offset_mapping = encoded_text["offset_mapping"]
        sample = {
            'id': idx,
            'input_ids': input_ids,
            'text': text,
            'offset_mapping': offset_mapping
        }
        samples.append(sample)
    return samples


class Collate(object):
    def __init__(self, tkz):
        """

        :param tkz:
        """
        self.tokenizer = tkz

    def __call__(self, batch):
        """

        :param batch:
        :return:
        """
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        return output


class FeedbackDataset:
    def __init__(self, samples, max_len, tokenizer):
        """

        :param samples:
        :param max_len:
        :param tokenizer:
        """
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        input_ids = self.samples[idx]["input_ids"]

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "ids": input_ids,
            "mask": attention_mask,
        }


class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_labels):
        """

        :param model_name:
        :param num_labels:
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained(model_name)
        hidden_dropout_prob: float = 0.18
        layer_norm_eps: float = 17589e-7
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
        self.transformer = AutoModel.from_config(config)
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, ids, mask):
        """

        :param ids:
        :param mask:
        :return:
        """
        transformer_out = self.transformer(ids, mask)
        sequence_output = transformer_out.last_hidden_state
        logits = self.output(sequence_output)
        logits = torch.softmax(logits, dim=-1)
        return logits, 0, {}


def jn(pst, start, end):
    """

    :param pst:
    :param start:
    :param end:
    :return:
    """
    return " ".join([str(x) for x in pst[start:end]])


def link_evidence(oof):
    """

    :param oof:
    :return:
    """
    thresh = 1
    idu = oof['id'].unique()
    # idc = idu[1]
    eoof = oof[oof['class'] == "Evidence"]
    neoof = oof[oof['class'] != "Evidence"]
    for thresh2 in range(26, 27, 1):
        retval = []
        for idv in idu:
            for c in ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement',
                      'Counterclaim', 'Rebuttal']:
                q = eoof[(eoof['id'] == idv) & (eoof['class'] == c)]
                if len(q) == 0:
                    continue
                pst = []
                for i, r in q.iterrows():
                    pst = pst + [-1] + [int(x) for x in r['predictionstring'].split()]
                start = 1
                end = 1
                for i in range(2, len(pst)):
                    cur = pst[i]
                    end = i
                    if (cur == -1 and c != 'Evidence') or ((cur == -1) and (
                            (pst[i + 1] > pst[end - 1] + thresh) or (pst[i + 1] - pst[start] > thresh2))):
                        retval.append((idv, c, jn(pst, start, end)))
                        start = i + 1
                v = (idv, c, jn(pst, start, end + 1))
                retval.append(v)
        roof = pd.DataFrame(retval, columns=['id', 'class', 'predictionstring'])
        roof = roof.merge(neoof, how='outer')
        return roof


tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_STORE, 'longformer-large-4096/'))
test_df = pd.read_csv(os.path.join(DATA_ROOT, 'sample_submission.csv'))

# for debug
# train_df = train_df[train_df['id'].isin(train_df['id'].unique()[:20])].copy()

# a sample is a dict of 'id', 'input_ids', 'text', 'offset_mapping'
test_samples = prepare_samples(test_df, is_train=False, tkz=tokenizer)
train_samples = prepare_samples(train_df, is_train=True, tkz=tokenizer)
collate = Collate(tkz=tokenizer)

# mk: turn this into a function call
# mk: model 1
path_to_saved_model = os.path.join('model_stores', 'fblongformerlarge1536')
checkpoints = glob.glob(os.path.join(path_to_saved_model, '*.bin'))
pred_folds = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for ch in checkpoints:
    ch_parts = ch.split(os.path.sep)
    pred_folder = os.path.join('preds', ch_parts[1])
    os.makedirs(pred_folder, exist_ok=True)
    pred_path = os.path.join(pred_folder, f'{os.path.splitext(ch_parts[-1])[0]}.dat')
    if os.path.exists(pred_path):
        with open(pred_path, 'rb') as f:
            pred_iter = pickle.load(f)
    else:
        test_dataset = FeedbackDataset(test_samples, MAX_LENGTH, tokenizer)
        train_dataset = FeedbackDataset(train_samples, MAX_LENGTH, tokenizer)
        model = FeedbackModel(model_name=os.path.join('model_stores', 'longformer-large-4096'), num_labels=len(target_id_map) - 1)
        model_path = os.path.join(ch)
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict)  # this loads the nn.Module and match all the parameters in model.transformer
        model.to(device)
        test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate)
        train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate)
        iterator = iter(train_data_loader)
        a_data = next(iterator)  # WKNOTE: get a sample from an iterable object
        model.eval()
        pred_iter = []
        with torch.no_grad():
            for sample in tqdm(train_data_loader, desc='Predicting. '):
                sample_gpu = {k: sample[k].to(device) for k in sample}
                pred = model(**sample_gpu)
                pred_prob = pred[0].cpu().detach().numpy().tolist()
                pred_prob = [np.array(l) for l in pred_prob]  # to list of ndarray
                pred_iter.extend(pred_prob)
            del sample_gpu
            torch.cuda.empty_cache()
            gc.collect()
        with open(pred_path, 'wb') as f:
            pickle.dump(pred_iter, f)
    pred_folds.append(pred_iter)

# mk: model 2
path_to_saved_model = os.path.join('model_stores', 'tez-fb-large')
checkpoints = glob.glob(os.path.join(path_to_saved_model, '*.bin'))
for ch in checkpoints:
    ch_parts = ch.split(os.path.sep)
    pred_folder = os.path.join('preds', ch_parts[1])
    os.makedirs(pred_folder, exist_ok=True)
    pred_path = os.path.join(pred_folder, f'{os.path.splitext(ch_parts[-1])[0]}.dat')
    if os.path.exists(pred_path):
        with open(pred_path, 'rb') as f:
            pred_iter = pickle.load(f)
    else:
        test_dataset = FeedbackDataset(test_samples, MAX_LENGTH, tokenizer)
        train_dataset = FeedbackDataset(train_samples, MAX_LENGTH, tokenizer)
        model = FeedbackModel(model_name=os.path.join('model_stores', 'longformer-large-4096'), num_labels=len(target_id_map) - 1)
        model_path = os.path.join(ch)
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict)  # this loads the nn.Module and match all the parameters in model.transformer
        model.to(device)
        test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate)
        train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate)
        iterator = iter(train_data_loader)
        a_data = next(iterator)  # WKNOTE: get a sample from an iterable object
        model.eval()
        pred_iter = []
        with torch.no_grad():
            for sample in tqdm(train_data_loader, desc='Predicting. '):
                sample_gpu = {k: sample[k].to(device) for k in sample}
                pred = model(**sample_gpu)
                pred_prob = pred[0].cpu().detach().numpy().tolist()
                pred_prob = [np.array(l) for l in pred_prob]  # to list of ndarray
                pred_iter.extend(pred_prob)
            del sample_gpu
            torch.cuda.empty_cache()
            gc.collect()
        with open(pred_path, 'wb') as f:
            pickle.dump(pred_iter, f)
    pred_folds.append(pred_iter)

# mk: model 3
path_to_saved_model = os.path.join('model_stores', 'visionary_cherry')
checkpoints = glob.glob(os.path.join(path_to_saved_model, '*.bin'))
pred_folds = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for ch in checkpoints:
    ch_parts = ch.split(os.path.sep)
    pred_folder = os.path.join('preds', ch_parts[1])
    os.makedirs(pred_folder, exist_ok=True)
    pred_path = os.path.join(pred_folder, f'{os.path.splitext(ch_parts[-1])[0]}.dat')
    if os.path.exists(pred_path):
        with open(pred_path, 'rb') as f:
            pred_iter = pickle.load(f)
    else:
        test_dataset = FeedbackDataset(test_samples, MAX_LENGTH, tokenizer)
        train_dataset = FeedbackDataset(train_samples, MAX_LENGTH, tokenizer)
        model = FeedbackModel(model_name=os.path.join('model_stores', 'longformer'), num_labels=len(target_id_map) - 1)
        model_path = os.path.join(ch)
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict)  # this loads the nn.Module and match all the parameters in model.transformer
        model.to(device)
        test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate)
        train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate)
        iterator = iter(train_data_loader)
        a_data = next(iterator)  # WKNOTE: get a sample from an iterable object
        model.eval()
        pred_iter = []
        with torch.no_grad():
            for sample in tqdm(train_data_loader, desc='Predicting. '):
                sample_gpu = {k: sample[k].to(device) for k in sample}
                pred = model(**sample_gpu)
                pred_prob = pred[0].cpu().detach().numpy().tolist()
                pred_prob = [np.array(l) for l in pred_prob]  # to list of ndarray
                pred_iter.extend(pred_prob)
            del sample_gpu
            torch.cuda.empty_cache()
            gc.collect()
        with open(pred_path, 'wb') as f:
            pickle.dump(pred_iter, f)
    pred_folds.append(pred_iter)

# mk: model 4
path_to_saved_model = os.path.join('model_stores', 'curious_oath')
checkpoints = glob.glob(os.path.join(path_to_saved_model, '*.bin'))
pred_folds = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for ch in checkpoints:
    ch_parts = ch.split(os.path.sep)
    pred_folder = os.path.join('preds', ch_parts[1])
    os.makedirs(pred_folder, exist_ok=True)
    pred_path = os.path.join(pred_folder, f'{os.path.splitext(ch_parts[-1])[0]}.dat')
    if os.path.exists(pred_path):
        with open(pred_path, 'rb') as f:
            pred_iter = pickle.load(f)
    else:
        test_dataset = FeedbackDataset(test_samples, MAX_LENGTH, tokenizer)
        train_dataset = FeedbackDataset(train_samples, MAX_LENGTH, tokenizer)
        model = FeedbackModel(model_name=os.path.join('model_stores', 'longformer'), num_labels=len(target_id_map) - 1)
        model_path = os.path.join(ch)
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict)  # this loads the nn.Module and match all the parameters in model.transformer
        model.to(device)
        test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate)
        train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate)
        iterator = iter(train_data_loader)
        a_data = next(iterator)  # WKNOTE: get a sample from an iterable object
        model.eval()
        pred_iter = []
        with torch.no_grad():
            for sample in tqdm(train_data_loader, desc='Predicting. '):
                sample_gpu = {k: sample[k].to(device) for k in sample}
                pred = model(**sample_gpu)
                pred_prob = pred[0].cpu().detach().numpy().tolist()
                pred_prob = [np.array(l) for l in pred_prob]  # to list of ndarray
                pred_iter.extend(pred_prob)
            del sample_gpu
            torch.cuda.empty_cache()
            gc.collect()
        with open(pred_path, 'wb') as f:
            pickle.dump(pred_iter, f)
    pred_folds.append(pred_iter)


def build_feedback_prediction_string(preds, sample_pred_scores, sample_id, sample_text, offset_mapping, proba_thresh, min_thresh):
    """

    :param preds:
    :param sample_pred_scores:
    :param sample_id:
    :param sample_text:
    :param offset_mapping:
    :param proba_thresh:
    :param min_thresh:
    :return:
    """
    if len(preds) < len(offset_mapping):
        preds = preds + ["O"] * (len(offset_mapping) - len(preds))
        sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

    # loop through each sub-token and construct the LABELLED discourse segments
    idx = 0
    discourse_labels_details = []
    while idx < len(offset_mapping):
        start, _ = offset_mapping[idx]
        if preds[idx] != "O":
            label = preds[idx][2:]
        else:
            label = "O"
        phrase_scores = []
        phrase_scores.append(sample_pred_scores[idx])
        idx += 1
        while idx < len(offset_mapping):
            if label == "O":
                matching_label = "O"
            else:
                matching_label = f"I-{label}"
            if preds[idx] == matching_label:
                _, end = offset_mapping[idx]
                phrase_scores.append(sample_pred_scores[idx])
                idx += 1
            else:
                break
        if "end" in locals():
            discourse = sample_text[start:end]
            discourse_labels_details.append((discourse, start, end, label, phrase_scores))

    temp_df = []
    for phrase_idx, (discourse, start, end, label, phrase_scores) in enumerate(discourse_labels_details):
        word_start = len(sample_text[:start].split())
        word_end = word_start + len(sample_text[start:end].split())
        word_end = min(word_end, len(sample_text.split()))
        ps = " ".join([str(x) for x in range(word_start, word_end)])
        if label != "O":
            if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
                if len(ps.split()) >= min_thresh[label]:
                    temp_df.append((sample_id, label, ps))
    temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])
    return temp_df


# the following 3 funcs are from https://www.kaggle.com/cpmpml/faster-metric-computation
def calc_overlap3(set_prediction, set_ground_truth):
    """
    Calculates if the overlap between prediction and ground truth is enough fora potential True positive
    :param set_prediction:
    :param set_ground_truth:
    :return:
    """
    # Length of each and intersection
    try:
        len_ground_truth = len(set_ground_truth)
        len_prediction = len(set_prediction)
        inter = len(set_ground_truth & set_prediction)
        overlap_1 = inter / len_ground_truth
        overlap_2 = inter / len_prediction
        return overlap_1 >= 0.5 and overlap_2 >= 0.5
    except:  # at least one of the input is NaN
        return False


def score_feedback_comp_micro3(predictions_df, ground_truth_df, discourse_type):
    """
    A function that scores for the kaggle Student Writing Competition Uses the steps in the evaluation page here:
    https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    :param predictions_df:
    :param ground_truth_df:
    :param discourse_type:
    :return:
    """
    ground_truth_df = ground_truth_df.loc[ground_truth_df['discourse_type'] == discourse_type, ['id', 'predictionstring']].reset_index(drop=True)
    predictions_df = predictions_df.loc[predictions_df['class'] == discourse_type, ['id', 'predictionstring']].reset_index(drop=True)
    predictions_df['pred_id'] = predictions_df.index
    ground_truth_df['gt_id'] = ground_truth_df.index
    predictions_df['predictionstring'] = [set(prediction.split(' ')) for prediction in predictions_df['predictionstring']]
    ground_truth_df['predictionstring'] = [set(prediction.split(' ')) for prediction in ground_truth_df['predictionstring']]

    """
    Step 1. all ground truths and predictions for a given class are compared.
    """
    joined = predictions_df.merge(ground_truth_df, left_on='id', right_on='id', how='outer', suffixes=('_pred', '_gt'))
    overlaps = [calc_overlap3(*args) for args in zip(joined.predictionstring_pred, joined.predictionstring_gt)]

    """
    2. If the overlap between the ground truth and prediction is >= 0.5, and the overlap between the prediction and 
    the ground truth >= 0.5,the prediction is a match and considered a true positive. If multiple matches exist, the 
    match with the highest pair of overlaps is taken. We don't need to compute the match to compute the score
    """
    true_positive = joined.loc[overlaps]['gt_id'].nunique()

    """    
    3. Any unmatched ground truths are false negatives and any unmatched predictions are false positives.
    """
    true_positive_and_false_positive = len(predictions_df)
    true_positive_and_false_negative = len(ground_truth_df)

    """
    4. calc micro_f1
    """
    my_f1_score = 2 * true_positive / (true_positive_and_false_positive + true_positive_and_false_negative)
    return my_f1_score


def score_feedback_comp3(predictions_df, ground_truth_df, return_class_scores=False):
    """

    :param predictions_df:
    :param ground_truth_df:
    :param return_class_scores:
    :return:
    """
    class_scores = {}
    for discourse_type in ground_truth_df.discourse_type.unique():
        class_score = score_feedback_comp_micro3(predictions_df, ground_truth_df, discourse_type)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


# average for a model


def ensemble_and_post_processing_loss_func(trial: optuna.trial.Trial):
    """

    :param trial:
    :return:
    """
    n_folds = len(pred_folds)  # mk: i think this is incorrectly set to only look at 5 folds from one model. It looks like n_folds is the output from the 'functions' from rows 343 - 464... I am guessing this is just doing the study for the last model or the first model. not all the models.

    # WKNOTE: set up a bunch of weights that follow a dirichlet distribution
    x = []
    for i in range(n_folds):
        x.append(- np.log(trial.suggest_float(f'x_{i}', 0, 1)))
    weights = []
    for i in range(n_folds):
        weights.append(x[i] / sum(x))
    for i in range(n_folds):
        trial.set_user_attr(f'w_{i}', weights[i])
    print(weights)
    raw_preds = []
    for fold, pred_fold in enumerate(pred_folds):
        sample_idx = 0
        for pred in pred_fold:
            weighted_pred = pred * weights[fold]
            if fold == 0:
                raw_preds.append(weighted_pred)
            else:
                raw_preds[sample_idx] += weighted_pred
                sample_idx += 1

    final_preds = []
    final_scores = []

    for rp in raw_preds:
        pred_classes = np.argmax(rp, axis=1)
        pred_scores = np.max(rp, axis=1)
        final_preds.append(pred_classes.tolist())
        final_scores.append(pred_scores.tolist())

    for j in range(len(train_samples)):
        tt = [id_target_map[p] for p in final_preds[j][1:]]
        tt_score = final_scores[j][1:]
        train_samples[j]['preds'] = tt
        train_samples[j]['pred_scores'] = tt_score

    proba_thresh = {
        "Lead": trial.suggest_uniform('proba_thres_lead', 0.1, 0.9),
        "Position": trial.suggest_uniform('proba_thres_position', 0.1, 0.9),
        "Evidence": trial.suggest_uniform('proba_thres_evidence', 0.1, 0.9),
        "Claim": trial.suggest_uniform('proba_thres_claim', 0.1, 0.9),
        "Concluding Statement": trial.suggest_uniform('proba_thres_conclusion', 0.1, 0.9),
        "Counterclaim": trial.suggest_uniform('proba_thres_counter', 0.1, 0.9),
        "Rebuttal": trial.suggest_uniform('proba_thres_rebuttal', 0.1, 0.9),
    }

    min_thresh = {
        "Lead": trial.suggest_int('min_lead', 1, 20),
        "Position": trial.suggest_int('min_position', 1, 20),
        "Evidence": trial.suggest_int('min_evidence', 1, 20),
        "Claim": trial.suggest_int('min_claim', 1, 20),
        "Concluding Statement": trial.suggest_int('min_conclusion', 1, 20),
        "Counterclaim": trial.suggest_int('min_counter', 1, 20),
        "Rebuttal": trial.suggest_int('min_rebuttal', 1, 20)
    }

    submission = []

    for sample_idx, sample in enumerate(train_samples):
        preds = sample["preds"]
        offset_mapping = sample["offset_mapping"]
        sample_id = sample["id"]
        sample_text = sample["text"]
        sample_pred_scores = sample["pred_scores"]
        sample_preds = []

        temp_df = build_feedback_prediction_string(preds, sample_pred_scores, sample_id, sample_text, offset_mapping, proba_thresh, min_thresh)
        submission.append(temp_df)

    submission = pd.concat(submission).reset_index(drop=True)
    submission = link_evidence(submission)

    train_pred_df = submission.copy()
    train_groud_truth_df = train_df[train_df['id'].isin(train_pred_df['id'].unique())].copy()

    # micro_lead = score_feedback_comp_micro3(train_pref_df, train_groud_truth_df, 'Lead')
    macro_f1 = score_feedback_comp3(train_pred_df, train_groud_truth_df)
    return macro_f1


study_db_path = 'ensemble.db'
study = optuna.create_study(
    direction='maximize', study_name='ensemble_fblongformer1536',
    storage=f'sqlite:///{study_db_path}', load_if_exists=True
)
study.optimize(ensemble_and_post_processing_loss_func, n_trials=2000)
best_params = study.best_params
print(f'the best model params are found on Trial #{study.best_trial.number}')
print(best_params)
