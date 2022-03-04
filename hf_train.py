"""

"""
import wandb
from hf_config import *
from hf_functions import *
from wandb_creds import *
from torch import cuda
import gc
from datasets import Dataset, load_metric
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

SAMPLE = False  # set True for debugging


if __name__ == "__main__":
    # wandb.login(key=API_KEY)
    # wandb.init(project="feedback_prize_pytorch", entity=ENTITY)

    label_to_index = label_to_index(classes=Config.CLASSES, tags=Config.TAGS)  #

    index_to_label = index_to_label(label_to_index=label_to_index, index_to_label=index_to_label)

    n_labels = create_n_labels(index_to_label=index_to_label)  # mk: this is too close to the capitalized constant name

    data = get_raw_text(text_ids=str, path=Config.path)

    processed_df = preprocess_text(df=Config.TRAIN_DATA)

    # debugging
    if SAMPLE: 
        df = Config.TRAIN_DATA.sample(n=100).reset_index(drop=True)

    dataset = dataset(df=processed_df)

    tokenizer = get_tokenizer(
        Config.MODEL_CHECKPOINT,
        add_prefix_space=True
    )

    fix_beginnings(Parameters.E)

    o = tokenize_and_align_labels(  # FIX_ME: Need to have a nore meaningful variable name than o
        max_length=TrainingHyperParameters.MAX_LENGTH,
        labels_to_index=label_to_index,  #
        tokenizer=tokenizer,
        stride=tokenizer.STRIDE,
        examples=tokenizer.examples,
    )

    tokenized_datasets = map_datasets(
        batch_size=Parameters.BATCH_SIZE,
        datasets=dataset,
        tokenize_and_align_labels=o,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        Config.MODEL_CHECKPOINT,
        num_labels=n_labels
    )

    model_name = Config.MODEL_CHECKPOINT.kfolds_split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-{Config.TASK}",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",  # mk: read the docs on this. I don't like how its saving right now.
        learning_rate=TrainingHyperParameters.LEARNING_RATE,
        per_device_train_batch_size=TrainingHyperParameters.BATCH_SIZE,
        per_device_eval_batch_size=TrainingHyperParameters.BATCH_SIZE,
        num_train_epochs=TrainingHyperParameters.N_EPOCHS,
        weight_decay=TrainingHyperParameters.WEIGHT_DECAY,
        report_to='wandb',
        gradient_accumulation_steps=TrainingHyperParameters.GRAD_ACC,
        warmup_ratio=TrainingHyperParameters.WARMUP_RATIO
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    metric = load_metric("seqeval")  # FIX_ME: not the competition metric, but for now its be better than nothing...

    p = []  # mk: Issue with p unfulfilled. Not present in notebook. try using an empty list prior to function call.

    compute_metrics(
        p=p,
        index_to_label=index_to_label,
        metric=metric
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["TRAIN_DF"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use the GPU

    wandb.watch(model)  # mk: new addition

    trainer.train()  # mk: this needs to be refactored into a for loop to allow for Kfolds

    wandb.log()  # mk: new addition

    wandb.finish()  #

    trainer.save_model(Config.MODEL_PATH)  # mk: need to save as weights not model. one set of saved weights per fold

    tokenized_val = dataset.map(tokenize_for_validation, batched=True)  #

    ground_truth_df = ground_truth_for_validation(tokenized_val=tokenized_val)  #

    predictions, labels, _ = trainer.predict(tokenized_val['test'])

    preds = np.argmax(predictions, axis=-1)

    dfs = []

    for i in range(len(tokenized_val['test'])):
        dfs.append(
            pred_to_span(
                preds[i],
                tokenized_val['test'][i],
                classes=Config.CLASSES,
                min_tokens=Config.MIN_TOKENS,
                visualize=visualize
            )
        )

    pred_df = pd.concat(dfs, axis=0)

    pred_df['class'] = pred_df['discourse_type']

    print(pred_df)

    final_score = score_feedback_comp(
        pred_df=pred_df,
        ground_truth_df=ground_truth_df,
        return_class_scores=True,
        score_feedback_comp_micro=score_feedback_comp_micro
    )  # mk: does this make sense? seems very circular and self-referential

    torch.cuda.empty_cache()
    gc.collect()
