"""

"""
def xyz(model, max_length, tokenizer, train_samples, test_samples, batch_size, num_workers, collate, target_id_map):
    path_to_saved_model = os.path.join('model_stores', model)
    checkpoints = glob.glob(os.path.join(path_to_saved_model, '*.bin'))
    pred_folds = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for ch in checkpoints:
        ch_parts = ch.split(os.path.sep)
        pred_folder = os.path.join('preds', ch_parts[1])
        os.makedirs(pred_folder, exist_ok=True)
        pred_path = os.path.join(pred_folder, f'{os.path.splitext(ch_parts[-1])[0]}.dat')
        if os.path.exists(pred_path):
            with open(pred_path, 'rb') as f:
                pred_iter = pickle.load(f)
        else:
            test_dataset = FeedbackDataset(test_samples, max_length, tokenizer)
            train_dataset = FeedbackDataset(train_samples, max_length, tokenizer)
            model = FeedbackModel(model_name=os.path.join('model_stores', 'longformer'), num_labels=len(target_id_map) - 1)
            model_path = os.path.join(ch)
            model_dict = torch.load(model_path)
            model.load_state_dict(model_dict)  # this loads the nn.Module and match all the parameters in model.transformer
            model.to(device)
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)
            train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)
            iterator = iter(train_data_loader)
            a_data = next(iterator)  # WKNOTE: get a sample from an iterable object
            model.eval()
            pred_iter = []
            with torch.no_grad():
                for sample in tqdm(train_data_loader, desc='Predicting. '):
                    sample_gpu = {k: sample[k].to(device) for k in sample}
                    pred = model(**sample_gpu)
                    pred_prob = pred[0].cpu().detach().numpy().tolist()
                    pred_prob = [np.array(l) for l in pred_prob]  # to list of ndarray
                    pred_iter.extend(pred_prob)
                del sample_gpu
                torch.cuda.empty_cache()
                gc.collect()
            with open(pred_path, 'wb') as f:
                pickle.dump(pred_iter, f)
        pred_folds.append(pred_iter)
