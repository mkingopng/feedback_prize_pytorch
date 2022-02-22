"""

"""
import gc
import wandb
from wandb_creds import *
from new_config import *
from new_functions import *

if __name__ == "__main__":
    wandb.login(key=API_KEY)
    wandb.init()

    for fold in range(5):
        NUM_JOBS = 12
        args = parse_args()
        seed_everything(42)
        os.makedirs(args.output, exist_ok=True)
        df = pd.read_csv('5_train_folds.csv')

        train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
        valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        training_samples = prepare_training_data(train_df, tokenizer, args, num_jobs=NUM_JOBS)
        valid_samples = prepare_training_data(valid_df, tokenizer, args, num_jobs=NUM_JOBS)

        train_dataset = FeedbackDataset(training_samples, args.max_len, tokenizer)

        num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
        print(num_train_steps)

        model = FeedbackModel(model_name=args.model, num_train_steps=num_train_steps, learning_rate=args.lr,
                              num_labels=len(Targets.target_id_map) - 1,
                              steps_per_epoch=len(train_dataset) / args.batch_size)

        es = EarlyStopping(model_path=os.path.join(args.output, f"model_{args.fold}.bin"), valid_df=valid_df,
                           valid_samples=valid_samples, batch_size=args.valid_batch_size, patience=5, mode="max",
                           delta=0.001, save_weights_only=True, tokenizer=tokenizer)

        model.fit(train_dataset, train_bs=args.batch_size, device="cuda", epochs=args.epochs, callbacks=[es],
                  fp16=True, accumulation_steps=args.accumulation_steps)
