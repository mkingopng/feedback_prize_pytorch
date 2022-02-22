"""

"""
import os


class Parameters:
    EXPERIMENT_NAME = 'feedback_prize_exp_52-1'  # increment for each experiment
    MODEL_NAME = 'longformer-base-4096'  # base works, large should work better, augmentation should help
    DATA_DIR = 'data'
    OUTPUT_DIR = os.path.join(f'output/{EXPERIMENT_NAME}')  #


class HyperParameters:
    FOLD = 0  # the fold you want to train
    LR = 5e-5  # tried 3e-5 for the first three runs. should try 2e-5
    MAX_LEN = 1024  # this is a reasonable starting point but should be tuned
    BATCH_SIZE = 8  # keep an eye on GPU memory
    N_EPOCH = 20  # nominal starting point. seems reasonable based on experiments so far.
    ACCUMULATION_STEPS = 1  #


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
