"""
https://www.kaggle.com/wht1996/feedback-nn-train
"""
import copy
import numpy as np
import pandas as pd
import math
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
import os
import time
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
import gc
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def calc_overlap(row):
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))

    return inter / max(len_gt, len_pred)


def get_f1_score(test_pred, test_df, log=print, silent=True):
    if not silent:
        log('test_pred.shape:', test_pred.shape, '\ttest_df.shape:', test_df.shape, )
        log('pred class:\n', test_pred['class'].value_counts())
        log('true class:\n', test_df['discourse_type'].value_counts())
    f1s = []

    for c in sorted(test_pred['class'].unique()):
        pred_df = test_pred.loc[test_pred['class'] == c].copy()
        gt_df = test_df.loc[test_df['discourse_type'] == c].copy()

        gt_df = gt_df[['id', 'discourse_type', 'predictionstring']].reset_index(drop=True).copy()
        pred_df = pred_df[['id', 'class', 'predictionstring']].reset_index(drop=True).copy()
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

        #     print(joined.head())
        joined['min_overlaps'] = joined.apply(calc_overlap, axis=1)
        joined['potential_TP'] = (joined['min_overlaps'] >= 0.5)

        matched_pred_ids = joined.query('potential_TP')['pred_id'].unique()
        fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in matched_pred_ids]

        matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
        fn_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

        # Get numbers of each type
        TP = len(matched_gt_ids)
        FP = len(fp_pred_ids)
        FN = len(fn_gt_ids)
        # calc microf1
        f1_score = TP / (TP + 0.5 * (FP + FN))
        if not silent:
            log(f'{c:<20} f1 score:\t{f1_score}')
        f1s.append(f1_score)
    log('\nOverall f1 score \t', np.mean(f1s))

    return np.mean(f1s)


class Log:
    def __init__(self, log_path, time_key=True):
        self.path = log_path
        if time_key:
            self.path = self.path.replace('.'
                                          , '{}.'.format(time.strftime('_%Y%m%d%H%M%S', time.localtime(time.time()))))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=open(self.path, 'a+'))
        print('log path:', self.path)
        print('****************begin*********************', file=open(self.path, 'a+'))

    def __call__(self, *content):
        t1 = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print(*content)
        print(t1, content, file=open(self.path, 'a+'))

    def clean(self):
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=open(self.path, 'w'))
        print('****************begin*********************', file=open(self.path, 'a+'))


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int \
                                           ((
                                                    self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int \
                        (self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=6, mode="max", max_epoch=1e6, min_epoch=0, at_last_score=None):
        self.patience = patience
        self.mode = mode
        self.max_epoch = max_epoch
        self.min_epoch = min_epoch
        self.at_last_score = at_last_score if at_last_score is not None else -np.Inf
        self.epoch = 0
        self.early_stop = False
        self.best_model = None
        self.best_epoch = 0
        self.model_path = None
        self.best_score = -np.Inf if self.mode == "max" else np.Inf

    def __call__(self, epoch_score, model=None, model_path=None):
        self.model_path = model_path
        self.epoch += 1

        score = -epoch_score if self.mode == "min" else epoch_score

        if score <= self.best_score:
            counter = self.epoch - self.best_epoch
            print('EarlyStopping counter: {} out of {}'.format(counter, self.patience))
            if (counter >= self.patience) and (self.best_score > self.at_last_score) and (self.epoch >= self.min_epoch):
                self.early_stop = True
                self._save_checkpoint()
        else:
            self.best_score = score
            self.best_epoch = self.epoch
            self.best_model = copy.deepcopy(model).cpu()

        if self.max_epoch <= self.epoch:
            self.early_stop = True
            self._save_checkpoint()

    def _save_checkpoint(self):
        if self.model_path is not None and self.best_model is not None:
            torch.save(self.best_model.state_dict(), self.model_path.replace('_score', '_ ' + str(self.best_score)))
            print('model saved at: ', self.model_path.replace('_score', '_ ' + str(self.best_score)))


def mapping_to_ids(mapping, text):
    mapping[0] = 0 if mapping[0] == 1 else mapping[0]
    start = len(text[:mapping[0]].split())
    end = len(text[:mapping[1]].split())
    return [str(i) for i in range(start, end)]


def is_head(token):
    return len(token.split()) > 0 and token not in '.,;?!'


def get_offset_mapping(text, tokens):
    offset_mapping = []
    start_index = 0
    for t in tokens:
        while start_index < len(text) and text[start_index] in ['\xa0', '\n', ' ']:
            start_index += 1
        if t in ('[CLS]', '[SEP]'):
            mapping = (0, 0)
        elif t == '[UNK]':
            mapping = (start_index, start_index + 1)
        else:
            if t[0] == '▁':
                t = t[1:]
            t_len = len(t)
            if t_len == 0:
                mapping = (start_index, start_index + 1)
            else:
                sample = False
                for i in [0, 1, 2, 3, 4, 5, -1, 0]:
                    if t.lower() == text[start_index + i: start_index + t_len + i].lower():
                        sample = True
                        break
                if sample:
                    mapping = (start_index + i, start_index + t_len + i)
                elif t[0] == '<' and t[-1] == '>':
                    mapping = (start_index, start_index + 1)
                else:
                    mapping = (start_index + i, start_index + t_len + i)

        start_index = mapping[1]

        offset_mapping.append(mapping)
    if len(offset_mapping) != len(tokens):
        raise ValueError('offset_mapping error！')
    if abs(offset_mapping[-2][1] - len(text)) > 2:
        raise ValueError('offset_mapping error！')
    return offset_mapping


def encode(text, tokenizer, data, labels_to_ids):
    if 'deberta-v' in tokenizer.name_or_path:
        encoding = tokenizer.encode_plus(text)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        encoding['offset_mapping'] = get_offset_mapping(text, tokens)
    else:
        encoding = tokenizer.encode_plus(text,
                                         return_offsets_mapping=True,
                                         )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    offset_mapping = encoding['offset_mapping']

    unmask_length = sum(attention_mask)
    token_label = [0] * unmask_length
    if 'discourse_start' in data.columns:
        data = data.sort_values('discourse_start').reset_index(drop=True)
        char_label = list(zip(data.discourse_start.astype(int), data.discourse_end.astype(int), data.discourse_type))

        is_first = True
        for i, mapping in enumerate(offset_mapping):
            while len(char_label) > 0 and mapping[1] > char_label[0][1]:
                char_label.pop(0)
                is_first = True
            if len(char_label) == 0:
                break
            if mapping[1] >= char_label[0][0] and mapping != (0, 0):
                if is_first:
                    if is_head(text[mapping[0]:mapping[1]]):
                        token_label[i] = labels_to_ids['B- ' + char_label[0][2]]
                        is_first = False
                else:
                    token_label[i] = labels_to_ids['I- ' + char_label[0][2]]

    return input_ids, attention_mask, token_label, offset_mapping


def get_feat_helper(args, tokenizer, df, train_ids):
    training_samples = []
    for idx in train_ids:
        filename = args.text_path + idx + ".txt"
        with open(filename, "r") as f:
            text = f.read().rstrip()
        input_ids, attention_mask, token_label, offset_mapping = \
            encode(text, tokenizer, df[df['id'] == idx], args.labels_to_ids)
        training_samples.append({'id': idx, 'text': text, 'input_ids': input_ids,
                                 'attention_mask': attention_mask, 'token_label': token_label,
                                 'offset_mapping': offset_mapping})
    return training_samples


from joblib import Parallel, delayed


def get_feat(df, tokenizer, args, data_key):
    data_path = args.cache_path + 'feat_{}.pkl'.format(data_key)
    if os.path.exists(data_path) & (args.load_feat):
        data = pickle.load(open(data_path, '+rb'))
    else:
        num_jobs = 8
        data = []
        train_ids = df["id"].unique()

        train_ids_splits = np.array_split(train_ids, num_jobs)
        results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
            delayed(get_feat_helper)(args, tokenizer, df, idx) for idx in train_ids_splits
        )
        for result in results:
            data.extend(result)

        data = pd.DataFrame(sorted(data, key=lambda x: len(x['input_ids'])))
        pickle.dump(data, open(data_path, '+wb'))
    return data


class dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        id_ = self.data.id[index]
        input_ids = self.data.input_ids[index]
        attention_mask = self.data.attention_mask[index]
        token_label = self.data.token_label[index]

        item = {'id': id_,
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(token_label, dtype=torch.long),
                }

        return item


def feat_padding(input_ids, attention_mask, token_label, batch_length, padding_dict, padding_side):
    random_seed = None
    if padding_side == 'right':
        random_seed = 0
    elif padding_side == 'left':
        random_seed = 1
    else:
        random_seed = np.random.rand()

    mask_index = attention_mask.nonzero().reshape(-1)
    input_ids = input_ids.index_select(0, mask_index)
    token_label = token_label.index_select(0, mask_index)
    attention_mask = attention_mask.index_select(0, mask_index)
    ids_length = len(input_ids)

    if ids_length > batch_length:
        if random_seed <= 0.33:
            input_ids = input_ids[:batch_length]
            attention_mask = attention_mask[:batch_length]
            token_label = token_label[:batch_length]
        elif random_seed >= 0.66:
            input_ids = input_ids[-batch_length:]
            attention_mask = attention_mask[-batch_length:]
            token_label = token_label[-batch_length:]
        else:
            sub_length = ids_length - batch_length
            strat_idx = np.random.randint(sub_length + 1)
            input_ids = input_ids[strat_idx:strat_idx + batch_length]
            attention_mask = attention_mask[strat_idx:strat_idx + batch_length]
            token_label = token_label[strat_idx:strat_idx + batch_length]

    if ids_length < batch_length:
        add_length = batch_length - ids_length
        if random_seed <= 0.33:
            input_ids = F.pad(input_ids, (0, add_length), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (0, add_length), "constant", padding_dict['attention_mask'])
            token_label = F.pad(token_label, (0, add_length), "constant", padding_dict['input_ids'])
        elif random_seed >= 0.66:
            input_ids = F.pad(input_ids, (add_length, 0), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length, 0), "constant", padding_dict['attention_mask'])
            token_label = F.pad(token_label, (add_length, 0), "constant", padding_dict['input_ids'])
        else:
            add_length1 = np.random.randint(add_length + 1)
            add_length2 = add_length - add_length1
            input_ids = F.pad(input_ids, (add_length1, add_length2), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length1, add_length2), "constant",
                                   padding_dict['attention_mask'])
            token_label = F.pad(token_label, (add_length1, add_length2), "constant", padding_dict['input_ids'])

    return input_ids, attention_mask, token_label


class Collate:
    def __init__(self, model_length=None, max_length=None, padding_side='right', padding_dict={}):
        self.model_length = model_length
        self.max_length = max_length
        self.padding_side = padding_side
        self.padding_dict = padding_dict

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["labels"] = [sample["labels"] for sample in batch]

        # calculate max token length of this batch
        batch_length = None
        if self.model_length is not None:
            batch_length = self.model_length
        else:
            batch_length = max([len(ids) for ids in output["input_ids"]])
            if self.max_length is not None:
                batch_length = min(batch_length, self.max_length)

        for i in range(len(output["input_ids"])):
            output_fill = feat_padding(output["input_ids"][i], output["attention_mask"][i], output["labels"][i],
                                       batch_length, self.padding_dict, padding_side=self.padding_side)
            output["input_ids"][i], output["attention_mask"][i], output["labels"][i] = output_fill

        # convert to tensors
        output["input_ids"] = torch.stack(output["input_ids"])
        output["attention_mask"] = torch.stack(output["attention_mask"])
        output["labels"] = torch.stack(output["labels"])

        return output


def test(model, test_loader, test_feat, test_df, args, log):
    model.eval()
    te_loss, te_accuracy = [], []
    test_pred = []
    scaler = torch.cuda.amp.GradScaler()
    with torch.no_grad():
        for batch in test_loader:

            #             with torch.cuda.amp.autocast():
            #                 loss, te_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            #                 loss = loss.mean()
            if args.test_padding_side in ['right', 'left']:
                loss, te_logits = model_predict(model, batch, model_length=args.model_length
                                                , max_length=args.max_length,
                                                padding_dict=args.padding_dict, padding_side=args.test_padding_side)
            elif args.test_padding_side == 'double':
                loss1, te_logits1 = model_predict(model, batch, model_length=args.model_length
                                                  , max_length=args.max_length,
                                                  padding_dict=args.padding_dict, padding_side='right')
                loss2, te_logits2 = model_predict(model, batch, model_length=args.model_length
                                                  , max_length=args.max_length,
                                                  padding_dict=args.padding_dict, padding_side='left')
                loss = (loss1 + loss2) / 2
                te_logits = [(l1 + l2) / 2 for l1, l2 in zip(te_logits1, te_logits2)]
            # compute training accuracy
            for i in range(len(te_logits)):
                pred = te_logits[i].argmax(axis=-1)
                token_label = batch['labels'][i][batch['attention_mask'][i] > 0].cpu().numpy().reshape(-1)
                te_accuracy.append(accuracy_score(pred, token_label))
            te_loss.append(loss.item())
            test_pred.extend(te_logits)
    te_accuracy = np.mean(te_accuracy)
    te_loss = np.mean(te_loss)
    gc.collect()
    torch.cuda.empty_cache()
    model.train()

    test_feat['pred'] = test_pred
    segment_param = {
        "Lead": {'min_proba': [0.47, 0.41], 'begin_proba': 1.00, 'min_sep': 40, 'min_length': 5},
        "Position": {'min_proba': [0.45, 0.40], 'begin_proba': 0.90, 'min_sep': 21, 'min_length': 3},
        "Evidence": {'min_proba': [0.50, 0.40], 'begin_proba': 0.56, 'min_sep': 2, 'min_length': 21},
        "Claim": {'min_proba': [0.40, 0.30], 'begin_proba': 0.30, 'min_sep': 10, 'min_length': 1},
        "Concluding Statement": {'min_proba': [0.58, 0.25], 'begin_proba': 0.93, 'min_sep': 50, 'min_length': 5},
        "Counterclaim": {'min_proba': [0.45, 0.25], 'begin_proba': 0.70, 'min_sep': 35, 'min_length': 6},
        "Rebuttal": {'min_proba': [0.37, 0.34], 'begin_proba': 0.70, 'min_sep': 45, 'min_length': 5},
    }

    test_predictionstring = after_deal(test_feat, args.labels_to_ids, segment_param, log)
    f1_score = get_f1_score(test_predictionstring, test_df, log)

    return te_loss, te_accuracy, f1_score, test_pred


def train(model, train_loader, test_loader, test_feat, test_df, args, model_path, log):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                              first_cycle_steps=args.num_train_steps, cycle_mult=1,
                                              max_lr=args.max_lr, min_lr=args.min_lr,
                                              warmup_steps=args.num_train_steps * 0.2,
                                              gamma=1., last_epoch=-1
                                              )
    es = EarlyStopping(patience=4, max_epoch=args.epochs)
    t0 = time.time()
    scaler = torch.cuda.amp.GradScaler()
    awp = AWP(model,
              optimizer,
              adv_lr=args.adv_lr,
              adv_eps=args.adv_eps,
              start_epoch=args.num_train_steps / args.epochs,
              scaler=scaler
              )
    step = 0
    f1_score = 0
    for epoch in range(100):

        tr_loss, tr_accuracy = [], []
        nb_tr_steps = 0

        model.train()
        for idx, batch in enumerate(train_loader):
            step += 1
            input_ids = batch['input_ids'].to(args.device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(args.device, dtype=torch.long)
            labels = batch['labels'].to(args.device, dtype=torch.long)

            with torch.cuda.amp.autocast():
                loss, tr_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss.mean()

            optimizer.zero_grad()
            #             loss.backward()
            #             optimizer.step()
            scaler.scale(loss).backward()
            if f1_score > 0.64:
                awp.attack_backward(input_ids, labels, attention_mask, step)

                # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=10
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tr_loss.append(loss.item())
            nb_tr_steps += 1

            # compute training accuracy
            for i in range(len(tr_logits)):
                pred = tr_logits[i][attention_mask[i] > 0].detach().cpu().numpy().argmax(axis=-1)
                token_label = labels[i][attention_mask[i] > 0].cpu().numpy().reshape(-1)
                tr_accuracy.append(accuracy_score(pred, token_label))

            if idx % 200 == 0:
                del input_ids, attention_mask, labels, loss, tr_logits
                gc.collect()
                torch.cuda.empty_cache()

                tr_loss_ = np.mean(tr_accuracy)
                tr_accuracy_ = np.mean(tr_accuracy)

                log(f"step: \t{idx:04d}, train loss: \t{tr_loss_:.4f}, train acc: \t{tr_accuracy_:.4f}, time: \t{int(time.time() - t0)}s")

        gc.collect()
        torch.cuda.empty_cache()

        tr_loss_ = np.mean(tr_loss)
        tr_accuracy_ = np.mean(tr_accuracy)
        te_loss, te_accuracy, f1_score, test_pred = test(model, test_loader, test_feat, test_df, args, log)
        log(f"epoch: \t{epoch:04d}, train loss: \t{tr_loss_:.4f}, train acc: \t{tr_accuracy_:.4f}, test loss: \t{te_loss:.4f}, test acc: \t{te_accuracy:.4f}, test f1: \t{f1_score:.4f}, time: \t{int(time.time() - t0)}s")
        es(f1_score, model, model_path=model_path)
        if es.early_stop:
            break

    return es.best_model.to(next(model.parameters()).device), test_pred


class AWP:
    def __init__(
            self,
            model,
            optimizer,
            adv_param="weight",
            adv_lr=1,
            adv_eps=0.2,
            start_epoch=0,
            adv_step=1,
            scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, x, y, attention_mask, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                adv_loss, tr_logits = self.model(input_ids=x, attention_mask=attention_mask, labels=y)
                adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self, ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def mapping_to_ids(mapping, text):
    word_start = len(text[:mapping[0]].split())
    word_end = word_start + len(text[mapping[0]:mapping[1]].split())
    word_end = min(word_end, len(text.split()))
    output = " ".join([str(x) for x in range(word_start, word_end)])
    return output


sentence_split = [';', ',', '.', '?', '!', '"']
null_symbol = ['\xa0', '\n', '\x93', ' ']


def get_sentence_split(text, pred_head=[]):
    start = []
    end = True
    for i, t in enumerate(text):
        if t == ' ':
            continue
        if i in pred_head:
            start.append(i)
            end = False
        if (t in sentence_split and text[i - 1] not in null_symbol) or \
                (t in null_symbol):
            end = True
            if (t in null_symbol) and (text[i - 1] not in null_symbol):
                start.append(i)
        elif end:
            start.append(i)
            end = False
        else:
            pass
    result = []
    for start, end in zip(start, start[1:] + [len(text)]):
        #         length = len(text[start:end].strip())
        result.append([start, end])
    return result


def word_decode(pred, b_pred, offset_mapping, text, **kwargs):
    pred_head = []
    for i, (p, b_p, mapping) in enumerate(zip(pred, b_pred, offset_mapping)):
        if mapping == (0, 0):
            continue
        if b_p >= kwargs['begin_proba'] or (abs(p - pred[i - 1]) > 0.1):
            pred_head.append(mapping[0])
    sentence = get_sentence_split(text, pred_head)

    sentence_offset_mapping = [[]]
    sentence_pred = [[]]
    sentence_b_pred = [[]]
    sentence_idx = [[]]
    i_sentence = 0
    for i, (p, b_p, mapping) in enumerate(zip(pred, b_pred, offset_mapping)):
        if mapping == (0, 0) or mapping[0] == mapping[1]:
            continue
        if i_sentence == len(sentence):
            print(text)
            print(mapping)
        if mapping[1] <= sentence[i_sentence][1]:
            sentence_offset_mapping[i_sentence].append(mapping)
            sentence_pred[i_sentence].append(p)
            sentence_b_pred[i_sentence].append(b_p)
            sentence_idx[i_sentence].append(i)
        else:
            i_sentence += 1
            sentence_offset_mapping.append([mapping])
            sentence_pred.append([p])
            sentence_b_pred.append([b_p])
            sentence_idx.append([i])
    sentence_offset_mapping2 = []
    sentence_pred2 = []
    sentence_b_pred2 = []
    sentence_idx2 = []
    for t, mapping_, pred_, b_pred_, idx_ in zip(sentence, sentence_offset_mapping, sentence_pred, sentence_b_pred
            , sentence_idx):
        #         print(np.round(pred_,2),'\t',text[t[0]:t[1]])
        if np.mean(pred_) > kwargs['min_proba'][0]:
            sentence_offset_mapping2.append([mapping_[0][0], mapping_[-1][1]])
            sentence_pred2.append(pred_)
            sentence_b_pred2.append(b_pred_)
            sentence_idx2.append(idx_)

    sentence_offset_mapping3 = []
    sentence_idx3 = []
    for i, mapping_ in enumerate(sentence_offset_mapping2):
        sep_start = sentence_idx2[i - 1][-1] + 1
        sep_end = sentence_idx2[i][0]
        sep_length = sep_end - sep_start
        if i == 0:
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        elif sentence_b_pred2[i][0] > kwargs['begin_proba']:
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        elif sep_length >= kwargs['min_sep']:
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        elif sep_length / kwargs['min_sep'] - np.mean(pred[sep_start:sep_end]) / kwargs['min_proba'][1] > 0:
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        else:
            sentence_offset_mapping3[-1][1] = mapping_[1]
            sentence_idx3[-1].extend(sentence_idx2[i])

    sentence_offset_mapping4 = []
    for i, mapping in enumerate(sentence_offset_mapping3):
        word_length = len(text[mapping[0]: mapping[1]].split())
        if word_length <= kwargs['min_length']:
            continue
        #         if word_length >= kwargs['min_length'][1]:
        #             continue
        #         if sum(pred[sentence_idx3[i][0] : sentence_idx3[i][-1]+1]) <= kwargs['min_proba'][0]*kwargs['min_length'][1]*1.2:   # 整体概率大于 p也算数
        #             continue
        sentence_offset_mapping4.append(mapping)
    result = [mapping_to_ids(mapping, text) for mapping in sentence_offset_mapping4]
    return result


def after_deal_helper(train_ids_sub, labels_to_ids, segment_param):
    y_pred = []
    for i, row in train_ids_sub:
        attention_mask = np.array(row.attention_mask)
        offset_mapping = [mapping for mapping, mask in zip(row.offset_mapping, attention_mask) if mask > 0]
        #         token_label    = np.array(row.token_label)[attention_mask>0]
        #         b_pred = row.pred[:,8:].sum(axis=1)
        for discourse in [
            'Claim',
            'Evidence',
            'Position',
            'Concluding Statement',
            'Lead',
            'Counterclaim',
            'Rebuttal'
        ]:
            if row.pred.shape[1] != 9:
                b_pred = row.pred[:, labels_to_ids['B- ' + discourse]].copy()
                pred = row.pred[:, labels_to_ids['I- ' + discourse]].copy()
                pred = pred + b_pred
            else:
                pred = row.pred[:, labels_to_ids['I- ' + discourse]].copy()
                b_pred = row.pred[:, 8].copy()
                b_pred = b_pred * pred
            pred_type = word_decode(pred, b_pred, offset_mapping, row.text,
                                    **segment_param[discourse],
                                    )
            for pred_type_ in pred_type:
                y_pred.append({'id': row['id'], 'class': discourse, 'predictionstring': pred_type_})

    return y_pred


def after_deal(data_pred, labels_to_ids, segment_param, log):
    num_jobs = 16
    y_pred = []
    train_ids = list(data_pred.sort_values('id').iterrows())
    train_ids_splits = np.array_split(train_ids, num_jobs)
    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(after_deal_helper)(train_ids_sub, labels_to_ids, segment_param) for train_ids_sub in train_ids_splits
    )
    for result in results:
        y_pred.extend(result)
    y_pred = pd.DataFrame(y_pred)
    print('y_pred.shape:', y_pred.shape)

    return y_pred


def model_predict(model, batch, model_length, max_length, padding_dict, padding_side='right', duplicate_cnt=100):
    batch_length = batch["input_ids"].shape[1]
    model_length = min(batch_length, max_length) if model_length is None else model_length
    batch_length = batch_length if batch_length >= model_length else model_length
    new_batch = []
    for i in range(len(batch["input_ids"])):
        new_batch.append(feat_padding(batch["input_ids"][i], batch["attention_mask"][i], batch["labels"][i],
                                      batch_length, padding_dict=padding_dict, padding_side=padding_side))
    batch["input_ids"] = torch.stack([c[0] for c in new_batch])
    batch["attention_mask"] = torch.stack([c[1] for c in new_batch])
    batch["labels"] = torch.stack([c[2] for c in new_batch])

    ids_length = batch['input_ids'].shape[1]

    loops = int(np.ceil((ids_length - duplicate_cnt) / (model_length - duplicate_cnt)))
    loops = max(loops, 1)
    loops_start = [i * (model_length - duplicate_cnt) for i in range(loops)]
    loops_end = [i * (model_length - duplicate_cnt) + model_length for i in range(loops)]
    if loops > 1:
        loops_start[-1] = ids_length - model_length
        loops_end[-1] = ids_length

    if padding_side == 'left':
        loops_start = [ids_length - idx for idx in loops_end][::-1]
        loops_end = [idx + model_length for idx in loops_start]

    losses = []
    preds = None
    for i, (start, end) in enumerate(zip(loops_start, loops_end)):
        device = next(model.parameters()).device
        input_ids = batch["input_ids"][:, start:end].to(device, dtype=torch.long)
        attention_mask = batch["attention_mask"][:, start:end].to(device, dtype=torch.long)
        labels = batch["labels"][:, start:end].to(device, dtype=torch.long)

        model.eval()
        with torch.cuda.amp.autocast():
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss.mean()

        if i == 0:
            preds = logits
        else:
            if i == (loops - 1):
                if i == 1:
                    inter_length = loops_end[0] - loops_start[-1]
                    weight = torch.floor_divide(torch.arange(inter_length), inter_length - 1).reshape(1, -1, 1).to(
                        device)
                    intersection = preds[:, start:] * (1 - weight) + logits[:, :inter_length] * (weight)
                    preds = torch.cat([preds[:, :start], intersection, logits[:, inter_length:]], dim=1)
                else:
                    preds = torch.cat([preds[:, :start], logits], dim=1)
            else:
                preds = torch.cat([preds, logits[:, duplicate_cnt:]], dim=1)
        losses.append(loss)

    pred_list = []
    for p, m in zip(preds, batch["attention_mask"]):
        m_index = m.nonzero().reshape(-1).to(device)
        pred_list.append(p.index_select(0, m_index).cpu().numpy().astype('float64'))

    return torch.tensor(losses).mean(), pred_list


def sorted_quantile(array, q):
    array = np.array(array)
    n = len(array)
    index = (n - 1) * q
    left = np.floor(index).astype(int)
    fraction = index - left
    right = left
    right = right + (fraction > 0).astype(int)
    i, j = array[left], array[right]
    return i + (j - i) * fraction


def get_label(text, mapping, predictionstring):
    word_start = len(text[:mapping[0]].split())
    word_end = word_start + len(text[mapping[0]:mapping[1]].split())
    word_end = min(word_end, len(text.split()))
    pred_idx = list(range(word_start, word_end))
    pred_cnt = len(pred_idx)
    for true_idx in predictionstring:
        if len(pred_idx) == 0 or true_idx[0] > pred_idx[-1] or true_idx[-1] < pred_idx[0]:
            continue
        inter_cnt = len(set(pred_idx) & set(true_idx))
        true_cnt = len(true_idx)
        inter_rate = min(inter_cnt / pred_cnt, pred_cnt / true_cnt)
        if inter_rate > 0.5:
            return 1, inter_rate, [true_idx[0], true_idx[-1]], [word_start, word_end]
    return 0, 0, [-1, -1], [word_start, word_end]


import re


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


import copy
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, AutoConfig


class TextModel(nn.Module):
    def __init__(self, model_name=None, num_labels=1):
        super(TextModel, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)  # 768
        self.drop_out = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, num_labels)

        if 'deberta-v2-xxlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False)  # 冻结24/48
        if 'deberta-v2-xlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:12].requires_grad_(False)  # 冻结12/24

    def forward(self, input_ids, attention_mask, labels=None):
        if 'gpt' in self.model.name_or_path:
            emb = self.model(input_ids)[0]
        else:
            emb = self.model(input_ids, attention_mask)[0]

        preds1 = self.output(self.dropout1(emb))
        preds2 = self.output(self.dropout2(emb))
        preds3 = self.output(self.dropout3(emb))
        preds4 = self.output(self.dropout4(emb))
        preds5 = self.output(self.dropout5(emb))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5

        logits = torch.softmax(preds, dim=-1)
        if labels is not None:
            loss = self.get_loss(preds, labels, attention_mask)
            return loss, logits
        else:
            return logits

    def get_loss(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fct(active_logits, true_labels)

        return loss


import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
CUDA_LAUNCH_BLOCKING = 1
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pdb
import torch
from torch import nn
from torch import cuda

import warnings

warnings.filterwarnings('ignore')

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_path', type=str, default="log.txt")
    parser.add_argument('--data_path', type=str, default="data/")
    parser.add_argument('--text_path', type=str, default="data/train/")
    parser.add_argument('--cache_path', type=str, default="cache/")
    parser.add_argument('--model_name', type=str, default="longformer-base-4096/")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--min_lr', type=float, default=0.000001)
    parser.add_argument('--max_lr', type=float, default=0.00001)
    parser.add_argument('--adv_lr', type=float, default=0.0000)
    parser.add_argument('--adv_eps', type=float, default=0.001)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_feat', action='store_true')
    parser.add_argument('--key_string', type=str, default='')

    args = parser.parse_args()

    if args.debug:
        args.epochs = 2

    args.key_string = args.model_name.split('/')[-2] + \
                      '_v2_15class_adv' + \
                      (f'_adv{args.adv_lr}' if args.adv_lr > 0 else '') + \
                      f'_fold{args.fold}' + \
                      ('_debug' if args.debug else '')

    log = Log(f'log/{args.key_string}.log', time_key=False)
    log('args:{}'.format(str(args)))
    return args, log


# Function to seed everything
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    args, log = get_args()
    seed_everything(args.seed)

    discourse_type = ['Claim', 'Evidence', 'Position', 'Concluding Statement', 'Lead', 'Counterclaim', 'Rebuttal']
    i_discourse_type = ['I-' + i for i in discourse_type]
    b_discourse_type = ['B-' + i for i in discourse_type]
    args.labels_to_ids = {k: v for v, k in enumerate(['O'] + i_discourse_type + b_discourse_type)}
    args.ids_to_labels = {k: v for v, k in args.labels_to_ids.items()}

    df = pd.read_csv(os.path.join(args.data_path, "train_folds.csv"))

    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    test_df = df[df["kfold"] == args.fold].reset_index(drop=True)

    if args.debug:
        sample_id = train_df['id'].drop_duplicates().sample(frac=0.1).values
        train_df = train_df[train_df['id'].isin(sample_id)].reset_index(drop=True)
        sample_id = test_df['id'].drop_duplicates().sample(frac=0.1).values
        test_df = test_df[test_df['id'].isin(sample_id)].reset_index(drop=True)
    log('train_df.shape:', train_df.shape, '\t', 'test_df.shape:', test_df.shape)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_feat = get_feat(train_df, tokenizer, args, 'train_feat' + args.key_string)
    test_feat = get_feat(test_df, tokenizer, args, 'test_feat' + args.key_string)
    #     train_feat = prepare_training_data(train_df, tokenizer, args, 8)
    #     test_feat = prepare_training_data(test_df, tokenizer, args, 8)
    log("train_feat.shap: {}".format(train_feat.shape), '\t', "test_feat.shape: {}".format(test_feat.shape))

    train_params = {'batch_size': args.train_batch_size,
                    'shuffle': True, 'num_workers': 2, 'pin_memory': True,
                    'collate_fn': Collate(args.max_length)
                    }
    test_params = {'batch_size': args.valid_batch_size,
                   'shuffle': False, 'num_workers': 2, 'pin_memory': True,
                   'collate_fn': Collate(4096)
                   }
    train_loader = DataLoader(dataset(train_feat), **train_params)
    test_loader = DataLoader(dataset(test_feat), **test_params)
    args.num_train_steps = len(train_feat) * args.epochs / args.train_batch_size

    # CREATE MODEL
    model = TextModel(args.model_name, num_labels=len(args.labels_to_ids))
    model = torch.nn.DataParallel(model)
    model.to(args.device)

    model_path = f'{args.cache_path + args.key_string}.pt'
    if args.load_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        log(f'Model loaded from {model_path}')
        te_loss, te_accuracy, f1_score, test_pred = test(model, test_loader, test_feat, test_df, args, log)
    else:
        model, test_pred = train(model, train_loader, test_loader, test_feat, test_df, args, model_path, log)
        torch.save(model.state_dict(), "model.bin")
