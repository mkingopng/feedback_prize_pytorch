"""

"""
from scratch_functions import *

# Inference and Validation Code
# We will infer in batches using our data loader which is faster than inferring one text at a time with a for-loop. The
# metric code is taken from Rob Mulla's great notebook [here][2]. Our model achieves validation F1 score 0.615!

# During inference our model will make predictions for each subword token. Some single words consist of multiple subword
# tokens. In the code below, we use a word's first subword token prediction as the label for the entire word. We can try
# other approaches, like averaging all subword predictions or taking `B` labels before `I` labels etc.

# [1]: https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
# [2]: https://www.kaggle.com/robikscube/student-writing-competition-twitch

if __name__ == "__main__":
    if COMPUTE_VAL_SCORE:  # note this doesn't run during submit
        # valid targets
        valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

        # OOF predictions
        oof = get_predictions(test_dataset, testing_loader)

        # compute f1 score
        f1s = []
        CLASSES = oof['class'].unique()
        print()

        for c in CLASSES:
            pred_df = oof.loc[oof['class'] == c].copy()
            gt_df = valid.loc[valid['discourse_type'] == c].copy()
            f1 = score_feedback_comp(pred_df, gt_df)
            print(c, f1)
            f1s.append(f1)
        print()
        print('Overall', np.mean(f1s))
        print()

    # Infer Test Data and Write Submission CSV. We will now infer the test data and write submission CSV
    sub = get_predictions(test_texts, test_texts_loader)
    print(sub.head())
    sub.to_csv("submission.csv", index=False)
