"""
step 1: clean this code up

step 2: identify promisin saved checkpoints

step 3: submit predictions based on one fold at a time.

step 4 submit predictions based on naive average

step 5 submit predictions based on bayesian optimized average
"""
import optuna
import gc

from at_approach.at_functions import score_feedback_comp
from at_approach.at_config import *


def loss_func(trial: optuna.trial.Trial):
    params = {
        'w1': trial.suggest_float('w1', 0, 1),
        'w2': trial.suggest_float('w2', 0, 1),
        'num_models': trial.suggest_int('num_models', 1, 10)
    }

    preds_ensembled = predict(**params)

    kaggle_loss = score_feedback_comp(
        pred_df=,
        gt_df=,
    )
    return kaggle_loss


# 10000
# 80% training, 20% val

# 8000 for training, 2000 for ensemble study
# 80%*8000 for train, the rest val

def predict(w1, w2):
    """
    have to have a saving mechanism to check if a saved has got predictions generated
    if yes, load them, if not, predict and save

    :param w1:
    :param w2:
    :return:
    """
    df = pd.read_csv(os.path.join("data", "sample_submission.csv"))

    df_ids = df["id"].unique()

    tokenizer = AutoTokenizer.from_pretrained(Args1.model)
    test_samples = prepare_test_data(df, tokenizer, Args1)
    collate = Collate(tokenizer=tokenizer)

    raw_preds = []
    for fold_ in range(10):
        current_idx = 0
        test_dataset = FeedbackDataset(test_samples, Args1.max_len, tokenizer)

        if fold_ < 5:
            model = FeedbackModel(
                model_name=Args1.model,  # need to change this in config depending on the checkpoints we want to test
                num_labels=len(Targets.target_id_map) - 1,
                learning_rate=HyperParameters.LR,
                num_train_steps=HyperParameters.STEPS,  # this is currently an arbitrary number guessed from AT's code
                steps_per_epoch=HyperParameters.STEPS
            )

            model.load(os.path.join(
                Args1.tez_model,
                f"model_{fold_}.bin"),
                weights_only=True
            )

            preds_iter = model.predict(
                test_dataset,
                batch_size=Args1.batch_size,
                n_jobs=-1,
                collate_fn=collate
            )

        else:
            model = FeedbackModel(
                model_name=Args2.model,  # need to change this in config depending on the checkpoints we want to test
                num_labels=len(Targets.target_id_map) - 1,
                learning_rate=HyperParameters.LR,
                num_train_steps=HyperParameters.STEPS,  # this is currently an arbitrary number guessed from AT's code
                steps_per_epoch=HyperParameters.STEPS  # this is currently an arbitrary number guessed from AT's code
            )

            model.load(os.path.join(
                Args2.tez_model,
                f"model_{fold_ - 5}.bin"),
                weights_only=True
            )

            preds_iter = model.predict(
                test_dataset,
                batch_size=Args2.batch_size,
                n_jobs=-1,
                collate_fn=collate
            )

        current_idx = 0

        for preds in preds_iter:
            preds = preds.astype(np.float16)
            if fold_ == 0:
                preds *= w1
            elif fold_ == 1:
                preds *= w2
            else:
                preds *= 1 - w1 - w2
            # preds = preds / 10
            if fold_ == 0:
                raw_preds.append(preds)
            else:
                raw_preds[current_idx] += preds
                current_idx += 1
        torch.cuda.empty_cache()
        gc.collect()
    return raw_preds


study_db_path = 'ensemble_study.db'
study = optuna.create_study(
    direction='maximize', study_name='ensemble_study',
    storage=f'sqlite:///{study_db_path}',
    load_if_exists=True
)
study.optimize(loss_func, n_trials=1000)  # take a while

best_params = study.best_params
