"""

"""
import copy
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
import random
import warnings
from ast import literal_eval

import tez
from tez import enums
from tez.callbacks import Callback

import torch
import torch.nn as nn
from sklearn import metrics
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from at_config import *


def split(df):  # mk: i need this
    """
    split using multi-stratification. Presently scikit-learn provides several cross validators with stratification.
    However, these cross validators do not offer the ability to stratify multilabel data. This iterative-stratification
    project offers implementations of MultilabelStratifiedKFold, MultilabelRepeatedStratifiedKFold, and
    MultilabelStratifiedShuffleSplit with a base algorithm for stratifying multilabel data.
    https://github.com/trent-b/iterative-stratification. This is largely unchanged from Abishek's code.
    :param df: the training data to be split
    :return: multilabel stratified Kfolds
    """
    df = Parameters.TRAIN_DF
    dfx = pd.get_dummies(df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()  #
    cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
    dfx = dfx[cols]
    mskf = MultilabelStratifiedKFold(n_splits=HyperParameters.N_FOLDS, shuffle=True, random_state=42)  #
    labels = [c for c in dfx.columns if c != "id"]
    dfx_labels = dfx[labels]  #
    dfx["kfold"] = -1  #
    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        print(len(trn_), len(val_))  #
        dfx.loc[val_, "kfold"] = fold  #
    df = df.merge(dfx[["id", "kfold"]], on="id", how="left")  #
    print(df.kfold.value_counts())  #
    df.to_csv(f"{HyperParameters.N_FOLDS}_train_folds.csv", index=False)  #
    return df


def _prepare_training_data_helper(args, tokenizer, df, train_ids):
    """

    :param args:
    :param tokenizer:
    :param df:
    :param train_ids:
    :return:
    """
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


def prepare_training_data(df, tokenizer, args, num_jobs):  # mk: do i need this?
    """

    :param df:
    :param tokenizer:
    :param args:
    :param num_jobs:
    :return:
    """
    training_samples = []
    train_ids = df["id"].unique()
    train_ids_splits = np.array_split(train_ids, num_jobs)
    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(args, tokenizer, df, idx) for idx in train_ids_splits)
    for result in results:
        training_samples.extend(result)
    return training_samples


def calc_overlap(row):  # mk: this is included in the ScoreFeedbackCompetition class. feeds score_comp_micro
    """
    Calculates the overlap between prediction and ground truth and overlap percentages used for determining true
    positives. This code is from Rob Mulla's @robikscube notebook,
    https://www.kaggle.com/robikscube/student-writing-competition-twitch
    :param row:
    :return:
    """
    set_pred = set(row.predictionstring_pred.kfolds_split(" "))
    set_gt = set(row.predictionstring_gt.kfolds_split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp_micro(pred_df, gt_df):
    """
    A function that scores for the kaggle Student Writing Competition. Uses the steps in the evaluation page here:
    https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation This code is from Rob Mulla's Kaggle kernel.
    :param pred_df:
    :param gt_df:
    :return:
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
    tp_pred_ids = joined.query("potential_TP").sort_values("max_overlap", ascending=False).groupby(["id", "predictionstring_gt"]).first()["pred_id"].values

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
    """
    A function that scores for the kaggle Student Writing Competition. Uses the steps in the evaluation page here:
    https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    :param pred_df:
    :param gt_df:
    :param return_class_scores:
    :return:
    """
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = pred_df.loc[pred_df["class"] == discourse_type].reset_index(drop=True).copy()
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


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
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        return {"ids": input_ids, "mask": attention_mask}


class Collate:
    def __init__(self, tokenizer):
        """

        :param tokenizer:
        """
        self.tokenizer = tokenizer

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


class EarlyStopping(Callback):
    def __init__(self, model_path, valid_df, valid_samples, batch_size, tokenizer, patience=5, mode="max", delta=0.001, save_weights_only=True):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_weights_only = save_weights_only
        self.model_path = model_path
        self.valid_samples = valid_samples
        self.batch_size = batch_size
        self.valid_df = valid_df
        self.tokenizer = tokenizer

        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def on_epoch_end(self, model):
        model.eval()
        valid_dataset = FeedbackDatasetValid(self.valid_samples, 4096, self.tokenizer)
        collate = Collate(self.tokenizer)

        preds_iter = model.predict(
            valid_dataset,
            batch_size=self.batch_size,
            n_jobs=-1,
            collate_fn=collate
        )

        final_preds = []
        final_scores = []
        for preds in preds_iter:
            pred_class = np.argmax(preds, axis=2)
            pred_scrs = np.max(preds, axis=2)
            for pred, pred_scr in zip(pred_class, pred_scrs):
                final_preds.append(pred.tolist())
                final_scores.append(pred_scr.tolist())

        for j in range(len(self.valid_samples)):
            tt = [Targets.id_target_map[p] for p in final_preds[j][1:]]
            tt_score = final_scores[j][1:]
            self.valid_samples[j]["preds"] = tt
            self.valid_samples[j]["pred_scores"] = tt_score

        submission = []

        for _, sample in enumerate(self.valid_samples):
            preds = sample["preds"]
            offset_mapping = sample["offset_mapping"]
            sample_id = sample["id"]
            sample_text = sample["text"]
            sample_pred_scores = sample["pred_scores"]

            # pad preds to same length as offset_mapping
            if len(preds) < len(offset_mapping):
                preds = preds + ["O"] * (len(offset_mapping) - len(preds))
                sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

            idx = 0
            phrase_preds = []
            while idx < len(offset_mapping):
                start, _ = offset_mapping[idx]
                if preds[idx] != "O":
                    label = preds[idx][2:]
                else:
                    label = "O"
                phrase_scores = [sample_pred_scores[idx]]
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
                    phrase = sample_text[start:end]
                    phrase_preds.append((phrase, start, end, label, phrase_scores))

            temp_df = []
            for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
                word_start = len(sample_text[:start].kfolds_split())
                word_end = word_start + len(sample_text[start:end].kfolds_split())
                word_end = min(word_end, len(sample_text.kfolds_split()))
                ps = " ".join([str(x) for x in range(word_start, word_end)])
                if label != "O":
                    if sum(phrase_scores) / len(phrase_scores) >= Targets.proba_thresh[label]:
                        temp_df.append((sample_id, label, ps))

            temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])

            submission.append(temp_df)

        submission = pd.concat(submission).reset_index(drop=True)
        submission["len"] = submission.predictionstring.apply(lambda x: len(x.kfolds_split()))

        def threshold(df):
            df = df.copy()
            for key, value in Targets.min_thresh.items():
                index = df.loc[df["class"] == key].query(f"len<{value}").index
                df.drop(index, inplace=True)
            return df

        submission = threshold(submission)

        # drop len
        submission = submission.drop(columns=["len"])

        scr = score_feedback_comp(submission, self.valid_df, return_class_scores=True)
        print(scr)
        model.train()

        epoch_score = scr[0]
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                model.model_state = enums.ModelState.END
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
            self.counter = 0


    def save_checkpoint(self, epoch_score, model):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print("Validation score improved ({} --> {}). Saving model!".format(self.val_score, epoch_score))
            model.save(self.model_path, weights_only=self.save_weights_only)
        self.val_score = epoch_score


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=HyperParameters.FOLD, required=False)
    parser.add_argument("--model", type=str, default=Parameters.MODEL_NAME, required=False)
    parser.add_argument("--lr", type=float, default=HyperParameters.LR, required=False)
    parser.add_argument("--output", type=str, default=Parameters.OUTPUT_DIR, required=False)
    parser.add_argument("--input", type=str, default=Parameters.DATA_DIR, required=False)
    parser.add_argument("--max_len", type=int, default=HyperParameters.MAX_LENGTH, required=False)
    parser.add_argument("--batch_size", type=int, default=HyperParameters.BATCH_SIZE, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=HyperParameters.BATCH_SIZE, required=False)
    parser.add_argument("--epochs", type=int, default=HyperParameters.N_EPOCH, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=HyperParameters.ACCUMULATION_STEPS, required=False)
    return parser.parse_args()


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
        """

        :return:
        """
        return self.length

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        input_ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        input_labels = [Targets.target_id_map[x] for x in input_labels]
        other_label_id = Targets.target_id_map["O"]
        padding_label_id = Targets.target_id_map["PAD"]

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_labels = [other_label_id] + input_labels

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            input_labels = input_labels[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        input_labels = input_labels + [other_label_id]
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_len - len(input_ids)  # in some notebooks its commented out from here
        if padding_length > 0:
            if self.tokenizer.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                input_labels = input_labels + [padding_label_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            else:
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                input_labels = [padding_label_id] * padding_length + input_labels
                attention_mask = [0] * padding_length + attention_mask  # to here

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(input_labels, dtype=torch.long),
        }


class FeedbackModel(tez.Model):
    def __init__(self, model_name, num_train_steps, learning_rate, num_labels, steps_per_epoch):
        """
        model class. Currently, uses tez. need to refactor to use base pytorch & huggingface
        :param model_name:
        :param num_train_steps:
        :param learning_rate:
        :param num_labels:
        :param steps_per_epoch:
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.steps_per_epoch = steps_per_epoch
        self.step_scheduler_after = "batch"

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def fetch_optimizer(self):
        """
        function to get and configure AdamW optimizer
        :return:
        """
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.learning_rate)
        return opt

    def fetch_scheduler(self):
        """

        :return:
        """
        sch = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.num_train_steps),
            num_training_steps=self.num_train_steps,
            num_cycles=1,
            last_epoch=-1,
        )
        return sch

    def loss(self, outputs, targets, attention_mask):
        """
        function calculate cross entropy loss
        :param outputs:
        :param targets:
        :param attention_mask:
        :return:
        """
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
        """

        :param outputs:
        :param targets:
        :param attention_mask:
        :return:
        """
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1).cpu().numpy()
        outputs = active_logits.argmax(dim=-1).cpu().numpy()
        idxs = np.where(active_loss == 1)[0]
        f1_score = metrics.f1_score(true_labels[idxs], outputs[idxs], average="macro")
        return {"f1": f1_score}

    def forward(self, ids, mask, token_type_ids=None, targets=None):
        """

        :param ids:
        :param mask:
        :param token_type_ids:
        :param targets:
        :return:
        """
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


def _prepare_test_data_helper(args, tokenizer, ids):
    """
    for inference
    :param args:
    :param tokenizer:
    :param ids:
    :return:
    """
    test_samples = []
    for idx in ids:
        filename = os.path.join(args.input_path, "test", idx + ".txt")
        with open(filename, "r") as f:
            text = f.read()

        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded_text["input_ids"]
        offset_mapping = encoded_text["offset_mapping"]

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }

        test_samples.append(sample)
    return test_samples


def prepare_test_data(df, tokenizer, args):
    """
    for inference
    :param df:
    :param tokenizer:
    :param args:
    :return:
    """
    test_samples = []
    ids = df["id"].unique()
    ids_splits = np.array_split(ids, 4)

    results = Parallel(n_jobs=4, backend="multiprocessing")(
        delayed(_prepare_test_data_helper)(args, tokenizer, idx) for idx in ids_splits
    )
    for result in results:
        test_samples.extend(result)

    return test_samples


def jn(pst, start, end):
    """
    for inference
    :param pst:
    :param start:
    :param end:
    :return:
    """
    return " ".join([str(x) for x in pst[start:end]])


def link_evidence(oof):
    """
    for inference. This is from gezi's code
    :param oof:
    :return:
    """
    thresh = 1
    idu = oof['id'].unique()
    idc = idu[1]
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
                    pst = pst + [-1] + [int(x) for x in r['predictionstring'].kfolds_split()]
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
