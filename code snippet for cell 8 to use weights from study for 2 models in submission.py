# Wilson's code snippet

df = pd.read_csv(os.path.join("../input/feedback-prize-2021/", "sample_submission.csv"))
df_ids = df["id"].unique()

tokenizer = AutoTokenizer.from_pretrained(args1.model)
test_samples = prepare_test_data(df, tokenizer, args1)
collate = Collate(tokenizer=tokenizer)

weights = [0.10390109220169462, 0.19006673843496108, 0.12927655749570174, 0.23494827264219648, 0.02550391845416928, 0.04257092129687859, 0.034103476382664165, 0.0038334649142933983, 0.14664136535520478, 0.089154192822236]
raw_preds = []
for fold_ in range(10):
    current_idx = 0
    test_dataset = FeedbackDataset(test_samples, args1.max_len, tokenizer)
    weight = weights[fold_]
    if fold_ < 5:
        model = FeedbackModel(model_name=args1.model, num_labels=len(target_id_map) - 1)
        model.load(os.path.join(args2.tez_model, f"model_{fold_}.bin"), weights_only=True)
        preds_iter = model.predict(test_dataset, batch_size=args1.batch_size, n_jobs=-1, collate_fn=collate)
    else:
        model = FeedbackModel(model_name=args2.model, num_labels=len(target_id_map) - 1)
        model.load(os.path.join(args2.tez_model, f"model_{fold_-5}.bin"), weights_only=True)
        preds_iter = model.predict(test_dataset, batch_size=args2.batch_size, n_jobs=-1, collate_fn=collate)
       
    current_idx = 0
   
    for preds in preds_iter:
        preds = preds.astype(np.float16)
        preds = preds * weight
        if fold_ == 0:
            raw_preds.append(preds)
        else:
            raw_preds[current_idx] += preds
            current_idx += 1
    torch.cuda.empty_cache()
    gc.collect()