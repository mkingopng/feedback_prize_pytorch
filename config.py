"""

"""
import pandas as pd
import os
from transformers import AutoConfig, AutoModel, AutoTokenizer


# config
class Config:
    EXPERIMENT_NAME = 'feedback_prize_exp_52-2'  # increment for each experiment
    MODEL_SAVENAME = f'longformer_{EXPERIMENT_NAME}'  # I've read bigbird gives 1-3% better performance. Consider.
    MODEL_NAME = 'longformer-base-4096'
    DATA_DIR = 'data'
    TRAIN_DF = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    FOLDS_DF = pd.read_csv('5_train_folds.csv')  # should be fixed now
    FOLD = 0  # the fold you want to train
    pre_data_dir = os.path.join('preprocessed')  # what is this used for? Can't find it in the code. delete?
    MODEL_DIR = os.path.join(f'model/{EXPERIMENT_NAME}')
    OUTPUT_DIR = os.path.join(f'output/{EXPERIMENT_NAME}')
    is_debug = False
    N_EPOCH = 20  # following Abishek's code. seems too high. I guess it doesn't matter because of early stopping
    N_FOLDS = 5  # maybe not much benefit to going beyond 5
    verbose_steps = 500  # what does this do?
    RANDOM_SEED = 42
    MAX_LENGTH = 1024  # 4096 is max model. Some points it specifies 4096, others 1024
    BATCH_SIZE = 8  # 51-0 indicated that batch size 1 was too small.
    LR = 3e-5  # based on the docs, this is the default LR. How to optimize this? 3e-5, 2e-5?
    NUM_LABELS = 15
    NUM_JOBS = 12
    label_subtokens = True
    output_hidden_states = True
    hidden_dropout_prob = 0.1
    layer_norm_eps = 1e-7   # eps is the epsilon parameter, a very small number to prevent any division by zero. default is 1e-8
    add_pooling_layer = False
    if is_debug:
        debug_sample = 1000
        verbose_steps = 16
        N_EPOCH = 1
        N_FOLDS = 2


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


class Args1:
    input_path = "data"
    model = "longformer"
    tez_model = "fblongformerlarge1536/"
    output = "."
    batch_size = 12
    max_len = 4096


class Args2:
    input_path = "data"
    model = "longformer"
    tez_model = "tez-fb-large"
    output = "."
    batch_size = 12
    max_len = 4096


proba_thresh = {
    "Lead": 0.7,  # 0.689
    "Position": 0.55,  # 0.539
    "Evidence": 0.65,  # 0.639
    "Claim": 0.55,  # 0.539
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
