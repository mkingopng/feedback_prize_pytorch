"""

"""
import pandas as pd
import os
from collections import defaultdict
import torch


class Config:
    EXP_NUM = 1  # increment for each experiment
    TASK = 'ner'  #
    MODEL_CHECKPOINT = 'longformer-base-4096-hf'  # try using longformer large
    STRIDE = 128  #
    MIN_TOKENS = 6  #
    MODEL_PATH = f'{MODEL_CHECKPOINT.split("/")[-1]}-{EXP_NUM}'  #
    DATA_DIR = 'data'  #
    TRAIN_DATA = os.path.join('data', 'TRAIN_DF')  #
    SAMPLE = False  # set True for debugging
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use GPU


class Parameters:
    TRAIN_DF = pd.read_csv('TRAIN_DF.csv')  #
    CLASSES = TRAIN_DF.discourse_type.unique().tolist()  #
    TAGS = defaultdict()  #
    E = [0, 7, 7, 7, 1, 1, 8, 8, 8, 9, 9, 9, 14, 4, 4, 4]  #
    BATCH_SIZE = 20000  # how is this different from the other batch size variable?


class TrainingHyperParameters:
    BATCH_SIZE = 8  # based on GPU memory
    GRAD_ACC = 8  #
    LEARNING_RATE = 5e-5  # need to optimize this
    WEIGHT_DECAY = 0.01  #
    WARMUP_RATIO = 0.1  #
    N_EPOCHS = 5  #
    MAX_LENGTH = 1024  #


COLORS = {
    'Lead': '#8000ff',
    'Position': '#2b7ff6',
    'Evidence': '#2adddd',
    'Claim': '#80ffb4',
    'Concluding Statement': 'd4dd80',
    'Counterclaim': '#ff8042',
    'Rebuttal': '#ff0000',
    'Other': '#007f00'
}
