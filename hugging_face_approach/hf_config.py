"""

"""
import pandas as pd
import os
from collections import defaultdict
import torch
from pathlib import Path


class Config:
    EXP_NUM = 1  # increment for each experiment
    TASK = 'ner'  #
    MODEL_CHECKPOINT = 'longformer-base-4096-hf'  # try using longformer large
    MIN_TOKENS = 6  # mk: need to optimize this w/ wandb sweeps
    MODEL_PATH = f'{MODEL_CHECKPOINT.split("/")[-1]}-{EXP_NUM}'  #
    DATA_DIR = '../data'  #
    TRAIN_DATA = os.path.join(DATA_DIR, 'train')  #
    SAMPLE = False  # set True for debugging
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use GPU
    TRAIN_DF = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))  #
    CLASSES = TRAIN_DF.discourse_type.unique().tolist()  #
    TAGS = defaultdict()  #
    path = Path('../data/train')


class Parameters:
    E = [0, 7, 7, 7, 1, 1, 8, 8, 8, 9, 9, 9, 14, 4, 4, 4]  # mk: this name sucks. find something more meaningful. This looks similar to the labels that are stored as a list or dictionary in other code
    BATCH_SIZE = 20000  # mk: how is this different from the other batch size variable?


class TrainingHyperParameters:
    BATCH_SIZE = 8  # mk: based on GPU memory & model choice. 8 is ok for base, 1 for large
    GRAD_ACC = 8  # mk: need to optimize this w/ wandb sweeps
    LEARNING_RATE = 5e-5  # mk: need to optimize this w/ wandb sweeps
    WEIGHT_DECAY = 0.01  # mk: need to optimize this w/ wandb sweeps
    WARMUP_RATIO = 0.1  # mk: need to optimize this w/ wandb sweeps
    N_EPOCHS = 5  # mk: increase this to 20. no early stopping. only save checkpoint when new 'best' achieved
    MAX_LENGTH = 1024  # mk: need to optimize this w/ wandb sweeps
    N_FOLDS = 5  # mk: include this using at-approach methods
    STRIDE = 128  # mk: need to optimize this w/ wandb sweeps
    SEED = 42


COLORS = {
    'Lead': '#8000ff',
    'Position': '#2b7ff6',
    'Evidence': '#2adddd',
    'Claim': '#80ffb4',
    'Concluding Statement': 'd4dd80',
    'Counterclaim': '#ff8042',
    'Rebuttal': '#ff0000',
    'Other': '#007f00'
}  # mk: for visualization
