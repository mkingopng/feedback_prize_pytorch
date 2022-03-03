"""

"""
import gc
import wandb
from at_config import *
from at_functions import *
from wandb_creds import *

warnings.filterwarnings("ignore")

hyperparameter_defaults = dict(
    number_of_epochs=HyperParameters.N_EPOCH,
    number_of_folds=HyperParameters.N_FOLDS,
    verbose_steps=HyperParameters.VERBOSE_STEPS,
    random_seed=HyperParameters.RANDOM_SEED,
    max_length=HyperParameters.MAX_LENGTH,
    batch_size=HyperParameters.BATCH_SIZE,
    learning_rate=HyperParameters.LR,
    number_of_labels=HyperParameters.NUM_LABELS,
    hidden_dropout_probability=HyperParameters.HIDDEN_DROPOUT_PROB,
    accumulation_steps=HyperParameters.ACCUMULATION_STEPS,
    delta=HyperParameters.DELTA,
    patience=HyperParameters.PATIENCE
)

config_dictionary = dict(
    yaml='config.yaml',
    params=hyperparameter_defaults
)

if __name__ == "__main__":

    wandb.login(key=API_KEY)
    # wandb.init(entity=ENTITY, project=PROJECT, tags=TAGS, config=config_dictionary)
    wandb.init(
        config=config_dictionary,
        project="feedback_prize_pytorch",
        tags=TAGS,
        entity="feedback_prize_michael_and_wilson")

    split(Parameters.TRAIN_DF)

    for fold in range(5):
        HyperParameters.FOLD = fold
        split(df=Parameters.TRAIN_DF)
        num_jobs = HyperParameters.NUM_JOBS
        args = parse_args()
        seed_everything(HyperParameters.RANDOM_SEED)
        os.makedirs(args.output, exist_ok=True)
        folds_df = Parameters.FOLDS_DF
        print(folds_df.columns)
        train_df = folds_df[folds_df["kfold"] != args.fold].reset_index(drop=True)
        valid_df = folds_df[folds_df["kfold"] == args.fold].reset_index(drop=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        training_samples = prepare_training_data(train_df, tokenizer, args, num_jobs=HyperParameters.NUM_JOBS)
        valid_samples = prepare_training_data(valid_df, tokenizer, args, num_jobs=HyperParameters.NUM_JOBS)
        train_dataset = FeedbackDataset(training_samples, args.max_len, tokenizer)
        num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
        print(num_train_steps)

        model = FeedbackModel(
            model_name=args.model,
            num_train_steps=num_train_steps,
            learning_rate=args.lr,
            num_labels=len(Targets.target_id_map) - 1,
            steps_per_epoch=len(train_dataset) / args.batch_size
        )

        es = EarlyStopping(
            model_path=os.path.join(args.output, f"model_{args.fold}.bin"),
            valid_df=valid_df,
            valid_samples=valid_samples,
            batch_size=args.batch_size,
            patience=HyperParameters.PATIENCE,
            mode="max",
            delta=HyperParameters.DELTA,
            save_weights_only=True,
            tokenizer=tokenizer
        )

        model.fit(
            train_dataset,
            train_bs=args.batch_size,
            device="cuda",
            epochs=args.epochs,
            callbacks=[es],
            fp16=True,
            accumulation_steps=args.accumulation_steps,
        )

        # wandb.log({"mode": mode})
        wandb.watch(model)
        torch.cuda.empty_cache()
        gc.collect()

    wandb.finish()
