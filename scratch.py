"""
what can i poach from earlier work?

starting with chris deottes earaly notebook on longformer. Interesting that even without close examination, there is a
lot of similar code to abishek's. clearly I'm not the only one borrowing code. Its obviously standard practice among the
GMs.

The nice part is that this is base pytorch and doesn't use tez. Good place to start the new branch
"""
import os
from torch import cuda
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score
from ast import literal_eval

# declare how many gpus you wish to use. kaggle only has 1, but offline, you can use more
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0,1,2,3 for four gpu

# version for saving model weights
VER = 1

# if variable is none, then notebook computes tokens otherwise notebook loads tokens from path
LOAD_TOKENS_FROM = "longformer-large-4096"

# if variable is none, then notebook trains a new model otherwise it loads your previously trained model
LOAD_MODEL_FROM = "longformer-large-4096"

# if following is none, then notebook uses internet and downloads huggingface  config, tokenizer, and model
DOWNLOADED_MODEL_PATH = "longformer-large-4096"

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = "longformer-large-4096"
MODEL_NAME = "longformer-large-4096"

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

if DOWNLOADED_MODEL_PATH == 'longformer-large-4096':
    os.mkdir('longformer-large-4096')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    tokenizer.save_pretrained('longformer-large-4096')

    config_model = AutoConfig.from_pretrained(MODEL_NAME)
    config_model.num_labels = 15
    config_model.save_pretrained('longformer-large-4096')

    backbone = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config_model)
    backbone.save_pretrained('longformer-large-4096')

train_df = pd.read_csv('data/train.csv')
print(train_df.shape)
print(train_df.head())

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, test_texts = [], []
for f in list(os.listdir('data/test')):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open('data/test/' + f, 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})
print(test_texts.head())

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, train_texts = [], []
for f in tqdm(list(os.listdir('data/train'))):
    test_names.append(f.replace('.txt', ''))
    train_texts.append(open('data/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})
print(train_text_df.head())

# convert train text to NER labels
if not LOAD_TOKENS_FROM:
    all_entities = []
    for ii, i in enumerate(train_text_df.iterrows()):
        if ii % 100 == 0:
            print(ii, ', ', end='')
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
    train_text_df = pd.read_csv(f'{LOAD_TOKENS_FROM}/train_NER.csv')
    # pandas saves lists as string, we must convert back
    train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x))

print(train_text_df.shape)
train_text_df.head()

# create dictionaries that we can use during train and infer
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                 'I-Concluding Statement']

labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}
print(labels_to_ids)

# Define the dataset function
# Below is our PyTorch dataset function. It always outputs tokens and attention. During training it also provides
# labels. And during inference it also provides word ids to help convert token predictions into word predictions.

# Note that we use `text.split()` and `is_split_into_words=True` when we convert train text to labeled train tokens.
# This is how the HugglingFace tutorial does it. However, this removes characters like `\n` new paragraph. If you want
# your model to see new paragraphs, then we need to map words to tokens ourselves using `return_offsets_mapping=True`.
# See my TensorFlow notebook [here][1] for an example.

# Some of the following code comes from the example at HuggingFace [here][2]. However I think the code at that link is
# wrong. The HuggingFace original code is [here][3]. With the flag `LABEL_ALL` we can either label just the first
# subword token (when one word has more than one subword token). Or we can label all the subword tokens (with the
# word's label). In this notebook version, we label all the tokens. There is a Kaggle discussion [here][4]

# [1]: https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-617
# [2]: https://huggingface.co/docs/transformers/custom_datasets#tok_ner
# [3]: https://github.com/huggingface/transformers/blob/86b40073e9aee6959c8c85fcba89e47b432c4f4d/examples/pytorch/token-classification/run_ner.py#L371
# [4]: https://www.kaggle.com/c/feedback-prize-2021/discussion/296713

LABEL_ALL_SUBTOKENS = True


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
                    label_ids.append(labels_to_ids[word_labels[word_idx]])
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append(labels_to_ids[word_labels[word_idx]])
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

# Train Model
# The PyTorch train function is taken from Raghavendrakotala's great notebook [here][1]. I assume it uses a masked loss
# which avoids computing loss when target is `-100`. If not, we need to update this.

# In Kaggle notebooks, we will train our model for 5 epochs `batch_size=4` with Adam optimizer and learning rates
# `LR = [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7]`. The loaded model was trained offline with `batch_size=8` and
# `LR = [5e-5, 5e-5, 5e-6, 5e-6, 5e-7]`. (Note the learning rate changes `e-5`, `e-6`, and `e-7`). Using `batch_size=4`
# will probably achieve a better validation score than `batch_size=8`, but I haven't tried yet.

# [1]: https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533


def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # tr_preds, tr_labels = [], []

    # put model in training mode
    model.train()

    for idx, batch in enumerate(training_loader):

        ids = batch['input_ids'].to(config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
        labels = batch['labels'].to(config['device'], dtype=torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                return_dict=False)
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
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config['max_grad_norm'])

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


# create model
config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH+'/config.json')

model = AutoModelForTokenClassification.from_pretrained(DOWNLOADED_MODEL_PATH+'/pytorch_model.bin', config=config_model)

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

    torch.save(model.state_dict(), f'longformer_v{VER}.pt')
else:
    model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/longformer_v{VER}.pt'))
    print('Model loaded.')

# Inference and Validation Code
# We will infer in batches using our data loader which is faster than inferring one text at a time with a for-loop. The
# metric code is taken from Rob Mulla's great notebook [here][2]. Our model achieves validation F1 score 0.615!

# During inference our model will make predictions for each subword token. Some single words consist of multiple subword
# tokens. In the code below, we use a word's first subword token prediction as the label for the entire word. We can try
# other approaches, like averaging all subword predictions or taking `B` labels before `I` labels etc.

# [1]: https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
# [2]: https://www.kaggle.com/robikscube/student-writing-competition-twitch


def inference(batch):
    # move batch to gpu and infer
    ids = batch["input_ids"].to(config['device'])
    mask = batch["attention_mask"].to(config['device'])
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()

    # interate through each text and get pred
    predictions = []
    for k, text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

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
def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
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


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle Student Writing Competition
    Uses the steps in the evaluation page here: https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
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

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5, and the overlap between the prediction and
    # the ground truth >= 0.5, the prediction is a match and considered a true positive. If multiple matches exist, the
    # match with the highest pair of overlaps is taken.

    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])

    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)

    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)

    tp_pred_ids = joined.query('potential_TP').sort_values('max_overlap', ascending=False).groupby(['id', 'predictionstring_gt']).first()['pred_id'].values

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


if COMPUTE_VAL_SCORE:  # note this doesn't run during submit
    # valid targets
    valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

    # OOF predictions
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

# Infer Test Data and Write Submission CSV. We will now infer the test data and write submission CSV
sub = get_predictions(test_texts, test_texts_loader)
print(sub.head())
sub.to_csv("submission.csv", index=False)


