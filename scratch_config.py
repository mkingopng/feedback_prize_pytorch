"""

"""
import os
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

from torch import cuda

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForTokenClassification

from ast import literal_eval

# declare how many gpus you wish to use. kaggle only has 1, but offline, you can use more
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0,1,2,3 for four gpu


class Parameters:
    VER = 1  # version for saving model weights
    EXPERIMENT_NAME = f'feedback_prize_exp_{VER}'  # increment for each experiment
    MODEL_SAVENAME = f'longformer_{EXPERIMENT_NAME}'  # I've read bigbird gives 1-3% better performance. Consider.
    MODEL_NAME = 'longformer'  # this is longformer-base-4096
    DATA_DIR = 'data'
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


# hyperparameters
class HyperParameterss:
    """

    """
    N_EPOCH = 20  # seems too high. I guess it doesn't matter because of early stopping
    N_FOLDS = 5  # just a number for now. consider N-1. Consider 5
    VERBOSE_STEPS = 500  # 2000 steps based on paper
    PATIENCE = 5  # use 1 for debugging
    DELTA = 0.001
    RANDOM_SEED = 42
    MAX_LENGTH = 1024  # +/- 1024. Abishek's name implies 1536 for one of the models
    BATCH_SIZE = 4  # 4 is ok for longfromer-large. 8 is ok for longformer-base
    LR = 3e-5  # ref longformer paper. abishek uses 5e-5
    NUM_LABELS = 15  # question: is this needed? seems redundant. num_labels is an output of the model class
    NUM_JOBS = 12  # question: is this needed? seems redundant. num_jobs is an output of prepare_training_data funct.
    WEIGHT_DECAY_1 = 0.01
    WEIGHT_DECAY_2 = 0.0
    ACCUMULATION_STEPS = 1
    LABEL_SUBTOKENS = True  # question: is the needed? can't see it used in functions. relic from baseline?
    OUTPUT_HIDDEN_STATES = True  # question: is this needed? output_hidden_states is a funct of model class
    HIDDEN_DROPOUT_PROB = 0.1  # this looks similar to the Pdrop from "attention is all you need"
    LAYER_NORM_EPS = 1e-7  # eps is epsilon parameter, a small number to prevent any division by zero. default is 1e-8
    ADD_POOLING_LAYER = False  # question: is this needed? add_pooling_layer is a funct of model class


class TargetIdMap:
    """

    """
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


# config
class Config:
    """

    """
    # if variable is none, then notebook computes tokens otherwise notebook loads tokens from path
    LOAD_TOKENS_FROM = "longformer-base-4096"

    # if variable is none, then notebook trains a new model otherwise it loads your previously trained model
    LOAD_MODEL_FROM = "longformer-base-4096"

    # if following is none, then notebook uses internet and downloads huggingface  config, tokenizer, and model
    DOWNLOADED_MODEL_PATH = "longformer-base-4096"

    if DOWNLOADED_MODEL_PATH is None:
        DOWNLOADED_MODEL_PATH = "longformer-base-4096"
    MODEL_NAME = "longformer-base-4096"

    config = {'model_name': MODEL_NAME,
              'max_length': 1024,
              'train_batch_size': 4,
              'valid_batch_size': 4,
              'epochs': 5,
              'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],  # todo: rethink this
              'max_grad_norm': 10,  # this is missing from AT code
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
    output_labels = [
        'O',
        'B-Lead',
        'I-Lead',
        'B-Position',
        'I-Position',
        'B-Claim',
        'I-Claim',
        'B-Counterclaim',
        'I-Counterclaim',
        'B-Rebuttal',
        'I-Rebuttal',
        'B-Evidence',
        'I-Evidence',
        'B-Concluding Statement',
        'I-Concluding Statement'
    ]

    labels_to_ids = {v: k for k, v in enumerate(output_labels)}
    ids_to_labels = {k: v for k, v in enumerate(output_labels)}
    print(labels_to_ids)