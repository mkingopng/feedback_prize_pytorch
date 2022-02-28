"""

"""
import pandas as pd
import os
from collections import defaultdict


class Config:
    EXP_NUM = 4
    TASK = 'ner'
    MODEL_CHECKPOINT = 'longformer-base-4096'
    STRIDE = 128
    MIN_TOKENS = 6
    MODEL_PATH = f'{MODEL_CHECKPOINT.split("/")[-1]}-{EXP_NUM}'
    DATA_DIR = 'data'
    TRAIN_DATA = os.path.join('data', 'train')
    SAMPLE = False  # set True for debugging


class Parameters:
    TRAIN_DF = pd.read_csv('train.csv')
    CLASSES = TRAIN_DF.discourse_type.unique().tolist()
    TAGS = defaultdict()
    E = [0, 7, 7, 7, 1, 1, 8, 8, 8, 9, 9, 9, 14, 4, 4, 4]
    BATCH_SIZE = 20000


class TrainingHyperParameters:
    BS = 4
    GRAD_ACC = 8
    LR = 5e-5
    WD = 0.01
    WARMUP = 0.1
    N_EPOCHS = 5
    MAX_LENGTH = 1024

