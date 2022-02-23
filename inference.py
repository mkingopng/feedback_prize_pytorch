"""
this should yield about 0.689 - 0.690 using abishek's checkpoints and probability thresholds
"""


from functions import *
from config import *
import gc

if __name__ == "__main__":
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
                model_name=Args1.model,
                num_labels=len(Targets.target_id_map) - 1,
                learning_rate=HyperParameters.LR,
                num_train_steps=1560,
                steps_per_epoch=1560
            )

            model.load(os.path.join(Args1.tez_model, f"model_{fold_}.bin"), weights_only=True)

            preds_iter = model.predict(
                test_dataset,
                batch_size=Args1.batch_size,
                n_jobs=-1,
                collate_fn=collate
            )

        else:
            model = FeedbackModel(
                model_name=Args2.model,
                num_labels=len(Targets.target_id_map) - 1,
                learning_rate=HyperParameters.LR,
                num_train_steps=1560,  # this is just based on observation
                steps_per_epoch=1560  # this is just based on observation
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
            preds = preds / 10
            if fold_ == 0:
                raw_preds.append(preds)
            else:
                raw_preds[current_idx] += preds
                current_idx += 1
        torch.cuda.empty_cache()
        gc.collect()

        final_preds = []
        final_scores = []

        for rp in raw_preds:
            pred_class = np.argmax(rp, axis=2)
            pred_scrs = np.max(rp, axis=2)
            for pred, pred_scr in zip(pred_class, pred_scrs):
                pred = pred.tolist()
                pred_scr = pred_scr.tolist()
                final_preds.append(pred)
                final_scores.append(pred_scr)

        for j in range(len(test_samples)):
            tt = [Targets.id_target_map[p] for p in final_preds[j][1:]]
            tt_score = final_scores[j][1:]
            test_samples[j]["preds"] = tt
            test_samples[j]["pred_scores"] = tt_score

    submission = []
    for sample_idx, sample in enumerate(test_samples):
        preds = sample["preds"]
        offset_mapping = sample["offset_mapping"]
        sample_id = sample["id"]
        sample_text = sample["text"]
        sample_input_ids = sample["input_ids"]
        sample_pred_scores = sample["pred_scores"]
        sample_preds = []

        if len(preds) < len(offset_mapping):
            preds = preds + ["O"] * (len(offset_mapping) - len(preds))
            sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

        idx = 0
        phrase_preds = []
        while idx < len(offset_mapping):
            start, _ = offset_mapping[idx]
            if preds[idx] != "O":
                label = preds[idx][2:]
            else:
                label = "O"
            phrase_scores = [sample_pred_scores[idx]]
            idx += 1
            while idx < len(offset_mapping):
                if label == "O":
                    matching_label = "O"
                else:
                    matching_label = f"I-{label}"
                if preds[idx] == matching_label:
                    _, end = offset_mapping[idx]
                    phrase_scores.append(sample_pred_scores[idx])
                    idx += 1
                else:
                    break
            if "end" in locals():
                phrase = sample_text[start:end]
                phrase_preds.append((phrase, start, end, label, phrase_scores))

        temp_df = []
        for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
            word_start = len(sample_text[:start].split())
            word_end = word_start + len(sample_text[start:end].split())
            word_end = min(word_end, len(sample_text.split()))
            ps = " ".join([str(x) for x in range(word_start, word_end)])
            if label != "O":
                if sum(phrase_scores) / len(phrase_scores) >= Targets.proba_thresh[label]:
                    if len(ps.split()) >= Targets.min_thresh[label]:
                        temp_df.append((sample_id, label, ps))

        temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])
        submission.append(temp_df)

    submission = pd.concat(submission).reset_index(drop=True)
    submission = link_evidence(submission)
    submission.to_csv("submission.csv", index=False)
    print(submission.head())
