"""
observations:
    - i don't see any Kfolds
    - no early stopping
    - no probability thresholds or min_thresh
    - there is a list of 15 categories instead of dictionaries
    - there are a lot less parameters and hyperparameters specified
    - the learning rates are very different
    -
    -
"""
import os
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
import copy
from joblib import Parallel, delayed
import argparse
import random
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_transformers

from sklearn import metrics
from sklearn.metrics import accuracy_score

from transformers import AdamW
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from transformers import LongformerConfig, LongformerModel

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import wandb

from scratch_config import *


def split(df):
    """
    split using multi-stratification. Presently scikit-learn provides several cross validators with stratification.
    However, these cross validators do not offer the ability to stratify multilabel data. This iterative-stratification
    project offers implementations of MultilabelStratifiedKFold, MultilabelRepeatedStratifiedKFold, and
    MultilabelStratifiedShuffleSplit with a base algorithm for stratifying multilabel data.
    https://github.com/trent-b/iterative-stratification. This is largely unchanged from Abishek's code.
    :param df: the training data to be split
    :return: multilabel stratified Kfolds
    """
    df = Config.TRAIN_DF
    dfx = pd.get_dummies(df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()  #
    cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
    dfx = dfx[cols]
    mskf = MultilabelStratifiedKFold(n_splits=Params.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED)
    labels = [c for c in dfx.columns if c != "id"]
    dfx_labels = dfx[labels]  #
    dfx["kfold"] = -1  #
    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        print(len(trn_), len(val_))  #
        dfx.loc[val_, "kfold"] = fold  #
    df = df.merge(dfx[["id", "kfold"]], on="id", how="left")
    print(df.kfold.value_counts())  #
    df.to_csv(f"{Params.N_FOLDS}_train_folds.csv", index=False)
    return df


def NER_labels(train_text_df):
    # convert train text to NER labels
    if not Config.LOAD_TOKENS_FROM:
        all_entities = []
        for ii, i in enumerate(train_text_df.iterrows()):
            if ii % 100 == 0:
                print(ii, ', ', end='')
            total = i[1]['text'].split().__len__()
            entities = ["O"] * total
            for j in Parameters.TRAIN_DF[Parameters.TRAIN_DF['id'] == i[1]['id']].iterrows():
                discourse = j[1]['discourse_type']
                list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
                entities[list_ix[0]] = f"B-{discourse}"
                for k in list_ix[1:]:
                    entities[k] = f"I-{discourse}"
            all_entities.append(entities)
        train_text_df['entities'] = all_entities
        train_text_df.to_csv('train_NER.csv', index=False)

    else:
        train_text_df = pd.read_csv(f'{Config.LOAD_TOKENS_FROM}/train_NER.csv')
        # pandas saves lists as string, we must convert back
        train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x))

def _prepare_training_data_helper(args, tokenizer, df, train_ids):
    training_samples = []
    for idx in tqdm(train_ids):
        filename = os.path.join(args.input, "train", f'{idx}.txt')
        with open(filename, "r") as f:
            text = f.read()
        encoded_text = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = encoded_text["input_ids"]
        input_labels = copy.deepcopy(input_ids)
        offset_mapping = encoded_text["offset_mapping"]
        for k in range(len(input_labels)):
            input_labels[k] = "O"
        sample = {"id": idx, "input_ids": input_ids, "text": text, "offset_mapping": offset_mapping}
        temp_df = df[df["id"] == idx]
        for _, row in temp_df.iterrows():
            text_labels = [0] * len(text)
            discourse_start = int(row["discourse_start"])
            discourse_end = int(row["discourse_end"])
            prediction_label = row["discourse_type"]
            text_labels[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
            target_idx = []
            for map_idx, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)
            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = "B-" + prediction_label
            pred_end = "I-" + prediction_label
            input_labels[targets_start] = pred_start
            input_labels[targets_start + 1: targets_end + 1] = [pred_end] * (targets_end - targets_start)
        sample["input_ids"] = input_ids
        sample["input_labels"] = input_labels
        training_samples.append(sample)
    return training_samples


def prepare_training_data(df, tokenizer, args, num_jobs):
    training_samples = []
    train_ids = df["id"].unique()
    train_ids_splits = np.array_split(train_ids, num_jobs)
    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(args, tokenizer, df, idx) for idx in train_ids_splits)
    for result in results:
        training_samples.extend(result)
    return training_samples


def calc_overlap(row):
    """
    Calculates the overlap between prediction and ground truth and overlap percentages used for determining true
    positives.
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


def score_feedback_comp_micro(pred_df, gt_df):  # mk: this is very similar to code below
    """
    A function that scores for the kaggle Student Writing Competition. Uses the steps in the evaluation page here:
    https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation This code is from Rob Mulla's Kaggle kernel.
    """
    gt_df = gt_df[["id", "discourse_type", "predictionstring"]].reset_index(drop=True).copy()
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index

    # Step 1. all ground truths and predictions for tez given class are compared.
    joined = pred_df.merge(gt_df, left_on=["id", "class"], right_on=["id", "discourse_type"], how="outer",
                           suffixes=("_pred", "_gt"))
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")
    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5, and the overlap between the prediction and
    # the ground truth >= 0.5,  the prediction is tez match and considered tez true positive. If multiple matches exist,
    # the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])
    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = joined.query("potential_TP").sort_values("max_overlap", ascending=False).groupby(
        ["id", "predictionstring_gt"]).first()["pred_id"].values

    # 3. Any unmatched ground truths are false negatives and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]
    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    tp = len(tp_pred_ids)
    fp = len(fp_pred_ids)
    fn = len(unmatched_gt_ids)

    # calc micro f1
    my_f1_score = tp / (tp + 0.5 * (fp + fn))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = pred_df.loc[pred_df["class"] == discourse_type].reset_index(drop=True).copy()
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1  # this is our main score to track


class FeedbackDatasetValid:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids += [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        return {"ids": input_ids, "mask": attention_mask}


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
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

#
# class EarlyStopping(Callback):  # fix_me: tez related function
#     def __init__(self, model_path, valid_df, valid_samples, batch_size, tokenizer, patience=HyperParams.PATIENCE,
#                  mode="max",
#                  delta=HyperParams.DELTA, save_weights_only=True):
#         self.patience = patience
#         self.counter = 0
#         self.mode = mode
#         self.best_score = None
#         self.early_stop = False
#         self.delta = delta
#         self.save_weights_only = save_weights_only
#         self.model_path = model_path
#         self.valid_samples = valid_samples
#         self.batch_size = batch_size
#         self.valid_df = valid_df
#         self.tokenizer = tokenizer
#
#         if self.mode == "min":
#             self.val_score = np.Inf
#         else:
#             self.val_score = -np.Inf
#
#     def on_epoch_end(self, model):
#         model.eval()
#         valid_dataset = FeedbackDatasetValid(self.valid_samples, 4096, self.tokenizer)
#         collate = Collate(self.tokenizer)
#
#         preds_iter = model.predict(valid_dataset, batch_size=self.batch_size, n_jobs=-1, collate_fn=collate)
#
#         final_preds = []
#         final_scores = []
#         for preds in preds_iter:
#             pred_class = np.argmax(preds, axis=2)
#             pred_scrs = np.max(preds, axis=2)
#             for pred, pred_scr in zip(pred_class, pred_scrs):
#                 final_preds.append(pred.tolist())
#                 final_scores.append(pred_scr.tolist())
#
#         for j in range(len(self.valid_samples)):
#             tt = [TargetIdMap.id_target_map[p] for p in final_preds[j][1:]]
#             tt_score = final_scores[j][1:]
#             self.valid_samples[j]["preds"] = tt
#             self.valid_samples[j]["pred_scores"] = tt_score
#
#         submission = []
#         min_thresh = Params.MIN_THRESH
#         proba_thresh = Params.PROBA_THRESH
#
#         for _, sample in enumerate(self.valid_samples):
#             preds = sample["preds"]
#             offset_mapping = sample["offset_mapping"]
#             sample_id = sample["id"]
#             sample_text = sample["text"]
#             sample_pred_scores = sample["pred_scores"]
#
#             # pad preds to same length as offset_mapping
#             if len(preds) < len(offset_mapping):
#                 preds += ["O"] * (len(offset_mapping) - len(preds))
#                 sample_pred_scores += [0] * (len(offset_mapping) - len(sample_pred_scores))
#
#             idx = 0
#             phrase_preds = []
#             while idx < len(offset_mapping):
#                 start, _ = offset_mapping[idx]
#                 if preds[idx] != "O":
#                     label = preds[idx][2:]
#                 else:
#                     label = "O"
#                 phrase_scores = [sample_pred_scores[idx]]
#                 idx += 1
#                 while idx < len(offset_mapping):
#                     if label == "O":
#                         matching_label = "O"
#                     else:
#                         matching_label = f"I-{label}"
#                     if preds[idx] == matching_label:
#                         _, end = offset_mapping[idx]
#                         phrase_scores.append(sample_pred_scores[idx])
#                         idx += 1
#                     else:
#                         break
#                 if "end" in locals():
#                     phrase = sample_text[start:end]
#                     phrase_preds.append((phrase, start, end, label, phrase_scores))
#
#             temp_df = []
#             for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
#                 word_start = len(sample_text[:start].split())
#                 word_end = word_start + len(sample_text[start:end].split())
#                 word_end = min(word_end, len(sample_text.split()))
#                 ps = " ".join([str(x) for x in range(word_start, word_end)])
#                 if label != "O":
#                     if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
#                         temp_df.append((sample_id, label, ps))
#             temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])
#             submission.append(temp_df)
#
#         submission = pd.concat(submission).reset_index(drop=True)
#         submission["len"] = submission.predictionstring.apply(lambda x: len(x.split()))
#
#         def threshold(df):
#             df = df.copy()
#             for key, value in min_thresh.items():
#                 index = df.loc[df["class"] == key].query(f"len<{value}").index
#                 df.drop(index, inplace=True)
#             return df
#
#         submission = threshold(submission)
#
#         # drop len
#         submission = submission.drop(columns=["len"])
#
#         scr = score_feedback_comp(submission, self.valid_df, return_class_scores=True)
#         print(scr)  # this is the part that prints
#         model.train()
#
#         epoch_score = scr[0]
#         if self.mode == "min":
#             score = -1.0 * epoch_score
#         else:
#             score = np.copy(epoch_score)
#
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(epoch_score, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
#             if self.counter >= self.patience:
#                 model.model_state = enums.ModelState.END  # fix_me: tez related function
#         else:
#             self.best_score = score
#             self.save_checkpoint(epoch_score, model)
#             self.counter = 0
#
#     def save_checkpoint(self, epoch_score, model):
#         if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
#             print("Validation score improved ({} --> {}). Saving model!".format(self.val_score, epoch_score))
#             model.save(self.model_path, weights_only=self.save_weights_only)
#         self.val_score = epoch_score


def seed_everything(seed: int):
    random.seed(seed)  #
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  #
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed(seed)  #
    torch.backends.cudnn.deterministic = True  # ensures the algorithm is deterministic
    torch.backends.cudnn.benchmark = True  # might produce performance improvements if you do not need reproducibility


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=Params.FOLD, required=False)
    parser.add_argument("--model", type=str, default=Config.MODEL_NAME, required=False)
    parser.add_argument("--lr", type=float, default=HyperParams.LR, required=False)
    parser.add_argument("--output", type=str, default=Params.OUTPUT_DIR, required=False)
    parser.add_argument("--input", type=str, default=Params.DATA_DIR, required=False)
    parser.add_argument("--max_len", type=int, default=HyperParams.MAX_LENGTH, required=False)
    parser.add_argument("--batch_size", type=int, default=HyperParams.BATCH_SIZE, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=HyperParams.BATCH_SIZE, required=False)
    parser.add_argument("--epochs", type=int, default=Params.N_EPOCH, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=HyperParams.ACCUMULATION_STEPS, required=False)
    return parser.parse_args()


class FeedbackDataset:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        input_labels = [TargetIdMap.target_id_map[x] for x in input_labels]
        other_label_id = TargetIdMap.target_id_map["O"]
        padding_label_id = TargetIdMap.target_id_map["PAD"]

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_labels = [other_label_id] + input_labels

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            input_labels = input_labels[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids += [self.tokenizer.sep_token_id]
        input_labels += [other_label_id]
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            if self.tokenizer.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                input_labels = input_labels + [padding_label_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            else:
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                input_labels = [padding_label_id] * padding_length + input_labels
                attention_mask = [0] * padding_length + attention_mask

        return {"ids": torch.tensor(input_ids, dtype=torch.long),
                "mask": torch.tensor(attention_mask, dtype=torch.long),
                "targets": torch.tensor(input_labels, dtype=torch.long)}


class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_train_steps, learning_rate, num_labels, steps_per_epoch):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.steps_per_epoch = steps_per_epoch
        self.step_scheduler_after = "batch"

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = HyperParams.LAYER_NORM_EPS
        config = AutoConfig.from_pretrained(model_name)
        config.update({"output_hidden_states": True, "hidden_dropout_prob": hidden_dropout_prob,
                       "layer_norm_eps": layer_norm_eps, "add_pooling_layer": False, "num_labels": self.num_labels})

        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout()
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": HyperParams.WEIGHT_DECAY_1},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": HyperParams.WEIGHT_DECAY_2},
        ]
        optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)
        return optimizer

    def fetch_scheduler(self):
        sch = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.num_train_steps),
            num_training_steps=self.num_train_steps,
            num_cycles=1,
            last_epoch=-1,
        )
        return sch

    def loss(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()
        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1)
        outputs = active_logits.argmax(dim=-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)
        loss = loss_fct(active_logits, true_labels)
        return loss

    def monitor_metrics(self, outputs, targets, attention_mask):
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1).cpu().numpy()
        outputs = active_logits.argmax(dim=-1).cpu().numpy()
        idxs = np.where(active_loss == 1)[0]
        f1_score = metrics.f1_score(true_labels[idxs], outputs[idxs], average="macro")
        return {"f1": f1_score}

    def forward(self, ids, mask, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.transformer(ids, mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        logits = torch.softmax(logits, dim=-1)
        loss = 0

        if targets is not None:
            loss1 = self.loss(logits1, targets, attention_mask=mask)
            loss2 = self.loss(logits2, targets, attention_mask=mask)
            loss3 = self.loss(logits3, targets, attention_mask=mask)
            loss4 = self.loss(logits4, targets, attention_mask=mask)
            loss5 = self.loss(logits5, targets, attention_mask=mask)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            f1_1 = self.monitor_metrics(logits1, targets, attention_mask=mask)["f1"]
            f1_2 = self.monitor_metrics(logits2, targets, attention_mask=mask)["f1"]
            f1_3 = self.monitor_metrics(logits3, targets, attention_mask=mask)["f1"]
            f1_4 = self.monitor_metrics(logits4, targets, attention_mask=mask)["f1"]
            f1_5 = self.monitor_metrics(logits5, targets, attention_mask=mask)["f1"]
            f1 = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5) / 5
            metric = {"f1": f1}
            return logits, loss, metric

        return logits, loss, {}


######################################################################################################################

# functions from cdeotte's code

# Define the dataset function
# Below is our PyTorch dataset function. It always outputs tokens and attention. During training it also provides
# labels. And during inference it also provides word ids to help convert token predictions into word predictions.

# Note that we use `text.split()` and `is_split_into_words=True` when we convert train text to labeled train tokens.
# This is how the HugglingFace tutorial does it. However, this removes characters like `\n` new paragraph. If you want
# your model to see new paragraphs, then we need to map words to tokens ourselves using `return_offsets_mapping=True`.
# See my TensorFlow notebook [here][1] for an example.

# Some of the following code comes from the example at HuggingFace [here][2]. However I think the code at that link is
# wrong. The HuggingFace original code is [here][3]. With the flag `LABEL_ALL` we can either label just the first
# subword token (when one word has more than one subword token). Or we can label all the subword tokens (with the word's
# label). In this notebook version, we label all the tokens. There is a Kaggle discussion [here][4]

# [1]: https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-617
# [2]: https://huggingface.co/docs/transformers/custom_datasets#tok_ner
# [3]: https://github.com/huggingface/transformers/blob/86b40073e9aee6959c8c85fcba89e47b432c4f4d/examples/pytorch/token-classification/run_ner.py#L371
# [4]: https://www.kaggle.com/c/feedback-prize-2021/discussion/296713

LABEL_ALL_SUBTOKENS = True


class dataset(Dataset):  # mk: this is similar to the dataset class defined above.
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
        encoding = self.tokenizer(text.split(),
                                  is_split_into_words=True,
                                  # return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)
        word_ids = encoding.word_ids()

        # create targets
        if not self.get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(Config.labels_to_ids[word_labels[word_idx]])
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append(Config.labels_to_ids[word_labels[word_idx]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            encoding['labels'] = label_ids

        # convert to torch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids:
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)

        return item

    def __len__(self):
        return self.len


# Create Train and Validation Dataloaders
# We will use the same train and validation subsets as my TensorFlow notebook [here][1]. Then we can compare results.
# And/or experiment with ensembling the validation fold predictions.

# [1]: https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-617

# choose validation indexes (that match my TF notebook)
IDS = Config.train_df.id.unique()
print('There are', len(IDS), 'train texts. We will split 90% 10% for validation.')

# train valid split 90% 10%
np.random.seed(42)
train_idx = np.random.choice(np.arange(len(IDS)), int(0.9 * len(IDS)), replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)), train_idx)
np.random.seed(None)

# create train subset and valid subset
data = Config.train_text_df[['id', 'text', 'entities']]
train_dataset = data.loc[data['id'].isin(IDS[train_idx]), ['text', 'entities']].reset_index(drop=True)
test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

tokenizer = AutoTokenizer.from_pretrained(Config.DOWNLOADED_MODEL_PATH)
training_set = dataset(train_dataset, tokenizer, Config.config['max_length'], False)
testing_set = dataset(test_dataset, tokenizer, Config.config['max_length'], True)

# train dataset and valid dataset
train_params = {
    'batch_size': Config.config['train_batch_size'],
    'shuffle': True,
    'num_workers': 2,
    'pin_memory': True
}

test_params = {
    'batch_size': Config.config['valid_batch_size'],
    'shuffle': False,
    'num_workers': 2,
    'pin_memory': True
}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# test dataset
test_texts_set = dataset(Config.test_texts, tokenizer, Config.config['max_length'], True)
test_texts_loader = DataLoader(test_texts_set, **test_params)


# # Train Model
# The PyTorch train function is taken from Raghavendrakotala's great notebook [here][1]. I assume it uses a masked loss
# which avoids computing loss when target is `-100`. If not, we need to update this.

# In Kaggle notebooks, we will train our model for 5 epochs `batch_size=4` with Adam optimizer and learning rates
# `LR = [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7]`. The loaded model was trained offline with `batch_size=8` and
# `LR = [5e-5, 5e-5, 5e-6, 5e-6, 5e-7]`. (Note the learning rate changes `e-5`, `e-6`, and `e-7`). Using `batch_size=4`
# will probably achieve a better validation score than `batch_size=8`, but I haven't tried yet.

# [1]: https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
def train(epoch):  # mk: there appears to be no directly comparable function above but there must be something functionally equivalent
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # tr_preds, tr_labels = [], []

    # put model in training mode
    model.train()

    for idx, batch in enumerate(training_loader):

        ids = batch['input_ids'].to(Config.config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(Config.config['device'], dtype=torch.long)
        labels = batch['labels'].to(Config.config['device'], dtype=torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=False)

        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 200 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss after {idx:04d} training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
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
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=Config.config['max_grad_norm'])

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


# create model  # mk: there appears to be nothing directly eqivalent but there must be something functionally equivalent
config_model = AutoConfig.from_pretrained(Config.DOWNLOADED_MODEL_PATH + '/config.json')

model = AutoModelForTokenClassification.from_pretrained(Config.DOWNLOADED_MODEL_PATH + '/pytorch_model.bin', config=config_model)

model.to(Config.config['device'])

optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.config['learning_rates'][0])

# loop to train model (or load model)  # mk: i think this belongs in scratch_training.py
if not Config.LOAD_MODEL_FROM:
    for epoch in range(Config.config['epochs']):
        print(f"### Training epoch: {epoch + 1}")
        for g in optimizer.param_groups:
            g['lr'] = Config.config['learning_rates'][epoch]
        lr = optimizer.param_groups[0]['lr']
        print(f'### LR = {lr}\n')
        train(epoch)
        torch.cuda.empty_cache()
        gc.collect()

    torch.save(model.state_dict(), f'longformer-base-4096_v{Config.VER}.pt')
else:
    model.load_state_dict(torch.load(f'{Config.LOAD_MODEL_FROM}/longformer-base-4096_v{Config.VER}.pt'))
    print('Model loaded.')


# Inference and Validation Code
# We will infer in batches using our data loader which is faster than inferring one text at a time with a for-loop.
# The metric code is taken from Rob Mulla's great notebook [here][2]. Our model achieves validation F1 score 0.615!

# During inference our model will make predictions for each subword token. Some single words consist of multiple subword
# tokens. In the code below, we use a word's first subword token prediction as the label for the entire word. We can try
# other approaches, like averaging all subword predictions or taking `B` labels before `I` labels etc.

# [1]: https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
# [2]: https://www.kaggle.com/robikscube/student-writing-competition-twitch


def inference(batch):  # mk: there appears to be no direct equivalent but there must be a functional equivalent
    # move batch to GPU and infer
    ids = batch["input_ids"].to(Config.config['device'])
    mask = batch["attention_mask"].to(Config.config['device'])
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()

    # iterate through each text and get pred
    predictions = []
    for k, text_preds in enumerate(all_preds):
        token_preds = [Config.ids_to_labels[i] for i in text_preds]

        prediction = []
        word_ids = batch['wids'][k].numpy()
        previous_word_idx = -1
        for idx, word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:
                prediction.append(token_preds[idx])
                previous_word_idx = word_idx
        predictions.append(prediction)

    return predictions


# https://www.kaggle.com/zzy990106/pytorch-ner-infer
# code has been modified from original
def get_predictions(df=test_dataset, loader=testing_loader):
    # put model in training mode
    model.eval()

    # get word label predictions
    y_pred2 = []
    for batch in loader:
        labels = inference(batch)
        y_pred2.extend(labels)

    final_preds2 = []
    for i in range(len(df)):

        idx = df.id.values[i]
        # pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
        pred = y_pred2[i]  # Leave "B" and "I"
        preds = []
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O':
                j += 1
            else:
                cls = cls.replace('B', 'I')  # spans start with B
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1

            if cls != 'O' and cls != '' and end - j > 7:
                final_preds2.append((idx, cls.replace('I-', ''),
                                     ' '.join(map(str, list(range(j, end))))))

            j = end

    oof = pd.DataFrame(final_preds2)

    oof.columns = ['id', 'class', 'predictionstring']

    return oof


# from Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):  # mk: this appears to identical to above
    """
    Calculates the overlap between prediction and ground truth and overlap percentages used for determining true
    positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):  # mk: this is super similar to above
    """
    A function that scores for the kaggle Student Writing Competition
    Uses the steps in the evaluation page here: https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'predictionstring']].reset_index(drop=True).copy()

    pred_df = pred_df[['id', 'class', 'predictionstring']].reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=['id', 'class'],
        right_on=['id', 'discourse_type'],
        how='outer',
        suffixes=('_pred', '_gt')
    )

    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')

    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5, and the overlap between the prediction and
    # the ground truth >= 0.5, the prediction is a match and considered a true positive. If multiple matches exist, the
    # match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])

    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)

    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP').sort_values('max_overlap', ascending=False).groupby(
        ['id', 'predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()

    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    tp = len(tp_pred_ids)
    fp = len(fp_pred_ids)
    fn = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = tp / (tp + 0.5 * (fp + fn))
    return my_f1_score


if Config.COMPUTE_VAL_SCORE:  # note this doesn't run during submit
    # valid targets
    valid = Config.train_df.loc[Config.train_df['id'].isin(IDS[valid_idx])]

    # oof predictions
    oof = get_predictions(test_dataset, testing_loader)

    # compute f1 score
    f1s = []
    CLASSES = oof['class'].unique()
    print()
    for c in CLASSES:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = valid.loc[valid['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(c, f1)
        f1s.append(f1)
    print()
    print('Overall', np.mean(f1s))
    print()

# Infer Test Data and Write Submission CSV. We will now infer the test data and write submission CSV.
sub = get_predictions(Config.test_texts, test_texts_loader)
sub.head()
sub.to_csv("submission.csv", index=False)





