"""


https://huggingface.co/transformers/v3.0.2/notebooks.html
https://github.com/patil-suraj/Notebooks/blob/master/longformer_qa_training.ipynb
https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb
https://github.com/christianversloot/machine-learning-articles/blob/main/transformers-for-long-text-code-examples-with-longformer.md
https://jesusleal.io/2020/11/24/Longformer-with-IMDB/
https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tune_Longformer_Encoder_Decoder_(LED)_for_Summarization_on_pubmed.ipynb
"""
# imports
from __future__ import print_function
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Mapping, Union
import numpy as np
from collections import OrderedDict
from collections import Counter
import pandas as pd
import json
import string
import re
import argparse
import sys

from transformers import (
    LongformerConfig,
    LongformerModel,
    LongformerTokenizerFast,
    EvalPrediction,
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers import LongformerForQuestionAnswering
from tqdm.auto import tqdm

import torch
import nlp

configuration = LongformerConfig()


def get_correct_alignement(context, answer):  # mk: this needs to be tweaked
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text'][0]
    start_idx = answer['answer_start'][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx  # When the gold label position is good
    elif context[start_idx - 1:end_idx - 1] == gold_text:
        return start_idx - 1, end_idx - 1  # When the gold label is off by one character
    elif context[start_idx - 2:end_idx - 2] == gold_text:
        return start_idx - 2, end_idx - 2  # When the gold label is off by two character
    else:
        raise ValueError()


# Tokenize our training dataset
def convert_to_features(example):
    # Tokenize contexts and questions (as pairs of inputs)
    input_pairs = [example['question'], example['context']]
    encodings = tokenizer.encode_plus(input_pairs, pad_to_max_length=True, max_length=512)
    context_encodings = tokenizer.encode_plus(example['context'])

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes. this will give
    # us the position of answer span in the context text
    start_idx, end_idx = get_correct_alignement(example['context'], example['answers'])
    start_positions_context = context_encodings.char_to_token(start_idx)
    end_positions_context = context_encodings.char_to_token(end_idx - 1)

    # here we will compute the start and end position of the answer in the whole example as the example is encoded like
    # this <s> question</s></s> context</s> and we know the postion of the answer in the context we can just find out
    #  the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens) this will give
    #  us the position of the answer span in whole example
    sep_idx = encodings['input_ids'].index(tokenizer.sep_token_id)
    start_positions = start_positions_context + sep_idx + 1
    end_positions = end_positions_context + sep_idx + 1

    if end_positions > 512:
        start_positions, end_positions = 0, 0

    encodings.update({'start_positions': start_positions,
                      'end_positions': end_positions,
                      'attention_mask': encodings['attention_mask']})
    return encodings


# load TRAIN_DF and validation split of squad
train_dataset = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

train_dataset = train_dataset.map(convert_to_features)
valid_dataset = valid_dataset.map(convert_to_features, load_from_cache_file=False)

# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)

# cach the dataset, so we can load it directly for training
torch.save(train_dataset, 'train_data.pt')
torch.save(valid_dataset, 'valid_data.pt')

# training script
logger = logging.getLogger(__name__)


@dataclass
class DummyDataCollator(DataCollator):
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        start_positions = torch.stack([example['start_positions'] for example in batch])
        end_positions = torch.stack([example['end_positions'] for example in batch])

        return {
            'input_ids': input_ids,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'attention_mask': attention_mask
        }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or TRAIN_DIR if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached TRAIN_DF dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file,
    # make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args.json'))

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = LongformerTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = LongformerForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print('loading data')
    train_dataset = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    print('loading done')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DummyDataCollator(),
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


# Train
# write the arguments in a dict and store in a json file. The above code will load this file and parse the arguments.
args_dict = {
    "n_gpu": 1,
    "model_name_or_path": 'allenai/longformer-base-4096',
    "max_len": 512,
    "output_dir": './models',
    "overwrite_output_dir": True,
    "per_gpu_train_batch_size": 8,
    "per_gpu_eval_batch_size": 8,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "do_train": True
}

with open('args.json', 'w') as f:
    json.dump(args_dict, f)

# start training
main()


# evaluation

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.kfolds_split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
model = model.cuda()
model.eval()

valid_dataset = torch.load('valid_data.pt')
dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16)

answers = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        start_scores, end_scores = model(input_ids=batch['input_ids'].cuda(),
                                         attention_mask=batch['attention_mask'].cuda())
        for i in range(start_scores.shape[0]):
            all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
            answer = ' '.join(all_tokens[torch.argmax(start_scores[i]): torch.argmax(end_scores[i]) + 1])
            ans_ids = tokenizer.convert_tokens_to_ids(answer.split())
            answer = tokenizer.decode(ans_ids)
            answers.append(answer)

predictions = []
references = []
for ref, pred in zip(valid_dataset, answers):
    predictions.append(pred)
    references.append(ref['answers']['text'])


evaluate(references, predictions)
