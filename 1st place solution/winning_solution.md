https://www.kaggle.com/c/feedback-prize-2021/discussion/313177

choose the method of segmented prediction and splicing,

Finally we choice
- longformer-large,
- roberta-large, deberta-xxlarge,
- distilbart_mnli_12_9,
- bart_large_finetuned_squadv1
for ensemble.

# stage 2: lgb sentence prediction:
- claim: 0.983
- concluding_statement: 0.972
- counterclaim: 0.906
- evidence: 0.974
- lead: 0.970
- position: 0.928
- rebuttal: 0.895

# longformer train
https://www.kaggle.com/wht1996/feedback-nn-train

# lgb_train
https://www.kaggle.com/wht1996/feedback-lgb-train

# 5 fold longformer with post-processing
https://www.kaggle.com/wht1996/feedback-longformer-5fold-0-697

# 5fold_longformer with lgb:
https://www.kaggle.com/wht1996/feedback-two-stage-lb0-727

# cv ensemble with lgb:
https://www.kaggle.com/wht1996/feedback-two-stage-cv0-747

Our code and data is published on GitHub here: https://github.com/antmachineintelligence/Feedback_1st