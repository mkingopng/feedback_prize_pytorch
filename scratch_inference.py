"""

"""
from scratch_functions import *


if __name__ == "__main__":
    # inference and validation loop
    def inference(batch):
        # move batch to gpu and infer
        ids = batch["input_ids"].to(config['device'])
        mask = batch["attention_mask"].to(config['device'])
        outputs = model(ids, attention_mask=mask, return_dict=False)
        all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()

        # iterate through each text and get pred
        predictions = []
        for k, text_preds in enumerate(all_preds):
            token_preds = [ids_to_labels[i] for i in text_preds]

            prediction = []
            word_ids = batch['wids'][k].numpy()
            previous_word_idx = -1
            for idx, word_idx in enumerate(word_ids):
                if word_idx == -1:
                    pass
                elif word_idx != previous_word_idx:
                    prediction.append(token_preds[idx])
                    previous_word_idx = word_idx
            predictions.append(prediction)

        return predictions
