"""

"""
import wandb

from hf_functions import *
from wandb_creds import *

SAMPLE = False  # set True for debugging

# setup wandb for experiment tracking
# source: https://www.kaggle.com/debarshichanda/pytorch-w-b-jigsaw-starter


if __name__ == "__main__":
    wandb.login(key=API_KEY)
    wandb.init(project="feedback_prize_pytorch", entity=ENTITY)

    l2i = create_l2i(Parameters.CLASSES, Parameters.TAGS)
    i2l = create_i2l(l2i=l2i)
    n_labels = create_n_labels(i2l)

    df1 = Parameters.TRAIN_DF.groupby('id')['discourse_type'].apply(list).reset_index(name='classlist')
    df2 = Parameters.TRAIN_DF.groupby('id')['discourse_start'].apply(list).reset_index(name='starts')
    df3 = Parameters.TRAIN_DF.groupby('id')['discourse_end'].apply(list).reset_index(name='ends')
    df4 = Parameters.TRAIN_DF.groupby('id')['predictionstring'].apply(list).reset_index(name='predictionstrings')

    df = pd.merge(df1, df2, how='inner', on='id')
    df = pd.merge(df, df3, how='inner', on='id')
    df = pd.merge(df, df4, how='inner', on='id')
    df['text'] = df['id'].apply(get_raw_text)

    # debugging
    if SAMPLE: 
        df = df.sample(n=100).reset_index(drop=True)

    dataset = dataset(df)

    tokenizer = set_tokenizer(Config.MODEL_CHECKPOINT)

    fix_beginnings(Parameters.E)

    o = tokenize_and_align_labels(
        max_length=TrainingHyperParameters.MAX_LENGTH,
        l2i=l2i,
        tokenizer=tokenizer,
        stride=tokenizer.stride,
        examples=tokenizer.examples,
    )

    tokenized_datasets = tokenize_datasets(
        batch_size=Parameters.BATCH_SIZE,
        datasets=dataset
    )

    model = AutoModelForTokenClassification.from_pretrained(
        Config.MODEL_CHECKPOINT,
        num_labels=n_labels
    )

    model_name = Config.MODEL_CHECKPOINT.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-{Config.TASK}",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=TrainingHyperParameters.LR,
        per_device_train_batch_size=TrainingHyperParameters.BS,
        per_device_eval_batch_size=TrainingHyperParameters.BS,
        num_train_epochs=TrainingHyperParameters.N_EPOCHS,
        weight_decay=TrainingHyperParameters.WD,
        report_to='wandb',
        gradient_accumulation_steps=TrainingHyperParameters.GRAD_ACC,
        warmup_ratio=TrainingHyperParameters.WARMUP
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # this is not the competition metric, but for now this will be better than nothing...
    metric = load_metric("seqeval")

    compute_metrics(p=p, i2l=i2l, metric=metric)  # mk: an issue with p unfulfilled

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    wandb.finish()

    trainer.save_model(Config.MODEL_PATH)

    tokenized_val = dataset.map(tokenize_for_validation, batched=True)

    ground_truth_df = ground_truth_for_validation(tokenized_val=tokenized_val)

    predictions, labels, _ = trainer.predict(tokenized_val['test'])

    preds = np.argmax(predictions, axis=-1)

    predictions_0 = pred2span(preds[0], tokenized_val['test'][0], viz=True)
    print(predictions_0)

    predictions_1 = pred2span(preds[1], tokenized_val['test'][1], viz=True)
    print(predictions_1)

    dfs = []
    for i in range(len(tokenized_val['test'])):
        dfs.append(pred2span(preds[i], tokenized_val['test'][i]))

    pred_df = pd.concat(dfs, axis=0)
    pred_df['class'] = pred_df['discourse_type']
    print(pred_df)

    score_feedback_comp(pred_df=pred_df, gt_df=ground_truth_df, return_class_scores=True)

