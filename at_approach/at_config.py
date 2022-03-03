"""

"""
import pandas as pd
import os
from transformers import AutoConfig, AutoModel, AutoTokenizer


class Parameters:
    VER = 0
    EXPERIMENT_NAME = f'feedback_prize_exp_{VER}'  # increment for each experiment
    MODEL_SAVENAME = f'longformer_{EXPERIMENT_NAME}'  # I've read bigbird gives 1-3% better performance. Consider.
    MODEL_NAME = 'longformer-base-4096-hf'  # this is longformer-base-4096
    DATA_DIR = 'data'
    LOAD_TOKENS_FROM = 'longformer-4096-hf'
    TRAIN_DF = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    FOLDS_DF = pd.read_csv('5_train_folds.csv')
    pre_data_dir = os.path.join('preprocessed')  # what is this used for? Can't find it in the code. delete?
    MODEL_DIR = os.path.join(f'model/{EXPERIMENT_NAME}')
    OUTPUT_DIR = os.path.join(f'output/{EXPERIMENT_NAME}')
    is_debug = False
    if is_debug:
        debug_sample = 1000
        verbose_steps = 16
        N_EPOCH = 1
        N_FOLDS = 2


class HyperParameters:
    FOLD = 0  # the fold you want to train
    N_EPOCH = 20  # following Abishek's code. seems too high. I guess it doesn't matter because of early stopping
    N_FOLDS = 5  # maybe not much benefit to going beyond 5
    VERBOSE_STEPS = 500  # what does this do?
    RANDOM_SEED = 42
    MAX_LENGTH = 1024  # 4096 is model max. Some points it specifies 4096, others 1024.
    BATCH_SIZE = 1  # 1 -8 depending on model and max_length
    LR = 3e-5  # based on the docs, this is the default LR. How to optimize this? 3e-5, 2e-5?
    NUM_LABELS = 15
    NUM_JOBS = 12
    LABEL_SUBTOKENS = True
    OUTPUT_HIDDEN_STATES = True
    HIDDEN_DROPOUT_PROB = 0.1
    LAYER_NORM_EPS = 1e-7  # try 1e-8
    ADD_POOLING_LAYER = False
    ACCUMULATION_STEPS = 1
    DELTA = 0.001
    PATIENCE = 5


class Targets:
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

    proba_thresh = {
        "Lead": 0.7,  # 0.689,
        "Position": 0.55,  # 0.539,
        "Evidence": 0.65,  # 0.639,
        "Claim": 0.55,  # 0.539,
        "Concluding Statement": 0.7,  # 0.589
        "Counterclaim": 0.5,  # 0.539
        "Rebuttal": 0.55,  # 0.539
    }

    min_thresh = {
        "Lead": 9,
        "Position": 5,
        "Evidence": 14,
        "Claim": 3,
        "Concluding Statement": 11,
        "Counterclaim": 6,
        "Rebuttal": 4,
    }


class Args1:
    input_path = "../data"
    model = "longformer"  # change this to match the model used for training
    tez_model = "fblongformerlarge1536"  # change this to one of the other trained models to test
    output = "."
    batch_size = 12
    max_len = 4096


class Args2:
    input_path = "../data"
    model = "longformer-large-4056"  # change this to match the model used for training
    tez_model = "tez-fb-large"  # change this to one of the other trained models to test
    output = "."
    batch_size = 12
    max_len = 4096




"""
https://seeve.medium.com/what-is-natural-language-preprocessing-and-named-entity-recognition-how-to-do-natural-language-2b1d3140985e

Architecture:
- get base 2x base longformer to train and inference so I can score on kaggle. expect 0.68
- can i get a better result using 2x large longformer instead of base longformer?
- try 2x bigbird base ensemble instead of 2x longformer base?
- what about bigbird large?
- what about bigbird pegasus large?
- longformer + bigbird => bigformer?


Parameters
- probability thresholds? some small changes give a lift. how to systematically determine the optimal probability thresholds?
- NER preprocessing?
- max_length?
- tokenizer = can i used a different tokenizer to improve my performance?
- can i use a dictionary to augment the tokenizer to deal with spelling, punctuation and grammar errors?
- NER post processing & link-evidence simplified?


Hyperparameters: https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/10
- number of folds
- number of epochs -> i think the early stopping callback takes care of this
- batch size -> this is largely determined by GPU memory
- learning rate
- max length
- AdamW properties

the process:
- define a model
- define a range of all possible values for all hyper-parameters
- define a method for sampling hyper-parameters
- define an evaluative criteria to judge the model
- https://www.jeremyjordan.me/hyperparameter-tuning/

- goal is >0.70 and medal.

https://mccormickml.com/2019/07/22/BERT-fine-tuning/
"""
