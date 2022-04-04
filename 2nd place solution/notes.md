# Second Place Solution

Thank you Georgia State University, The Learning Agency Lab, and Kaggle for an awesome competition. The data was high 
quality and interesting. The metric and train test split were well done. This competition was a success and will benefit 
a good cause.

Teaming with Chun Ming Lee and Udbhav Bamba has been wonderful. I learned more about NLP than if I had taken an online 
course. These two Kagglers are grandmasters at NLP and I now know how to use HuggingFace Trainer efficiently and 
effectively. I also know how to customize models and losses. Thank you @leecming @ubamba98
Solution Summary

The secret sauce to our solution is powerful post process by Chun Ming (boost CV LB!), a huge variety of models 
implemented by Udbhav (many long sequence models!), and weighted box fusion by Chris (used in my previous comp here). 
Everything was fine-tuned on local CV and achieved 2nd Place $35,000!

## Weighted Box Fusion - CV 741, Public 727, Private 740

Our final solution contained the 10 models listed above. We included 3 out of 10 K-folds for each of the 10 models. 
Our 8 hour 30 minute submission inferred 27 models! (we removed 3 folds from 30)

Weighted Box Fusion by @zfturbo from his GitHub here was the magic that combined all the model predictions together. 
Individual models had average CV around 700 and the WBF ensemble achieved CV 741. (Folds from same models were averaged 
before WBF).

Weighted Box Fusion can do something that no other ensemble method can do. If one model has prediction string 8 9 10 11 
and another model has 10 11 12 13. Then weighted box fusion takes the average of the two starts, (8 + 10)/2 = 9 and the 
average of the two ends, (11 + 13)/2 = 12 resulting in 9 10 11 12. Furthermore, WBF reads submission.csv files and 
doesn't care what tokenizer you used, so its easy to use. (Note we add an extra column to submission dataframe with 
span confidence).

If we averaged the token/word probabilities, or used a voting classifier, we would either get the union 8 9 10 11 12 13 
or the intersection 10 11. Neither of which we want. Also averaging two "BIO" predictions gives a new prediction with 
two different "B" which we do not want either.

## Post Process
Chun Ming is the mastermind behind our team's strong PP. We applied the same PP to each model before WBF. Our PP code 
is posted here and Chun Ming will now describe it. Our PP significantly boosted CV LB!

We applied heavy post-processing to the word-level soft predictions each model made (after averaging token probabilities 
across folds and before the WBF ensembler). Overall CV was improved by ~.008.

In order of descending impact on CV:
- Repairing span predictions: Since we trained using cross-entropy loss at the token level, the raw predictions would 
often result in broken spans. For example, given a chain of Lead token predictions, there might be a misprediction of an 
“Other” in the middle. We had a rule to convert that “Other” back to Lead.

- Discourse-specific rules : We came up with common sense heuristics for the discourses. For example, for Lead, 
Position, and Concluding Statement - there should only be a maximum of one each for a text. We’d predict the most 
probable candidate for each or merge close duplicate spans.

- Adjusting lengths of predicted spans: We adjusted lengths of spans based on their original predicted length. For 
example, for a predicted Evidence span that was less than 45 words long, we’d shift the start of the predicted span 
back by 9 words. These rules and the improvement in LB/CV suggest we were taking advantage of the evaluation metric 
only requiring a 50% overlap in prediction and label to be counted as correct.

It’s not sexy but sitting down and eye-balling model predictions and comparing them with ground truths using a tool 
like displacy was rewarding in helping us better understand the task, how our models were behaving, and how to improve 
model predictions. 

## Model Details
Surprisingly many more models than LongFormer and BigBird could be trained with sequence lengths greater than 512. All 
our models were trained with sequence length 1536 and inferred at 1536 (except BigBird-base at 1024).

The model DeBERTa accepts any input size, so we just train with 1536 as is.

The model Funnel-transformer can accept any size after updating its config.

```angular2html
model = AutoModelForTokenClassification.from_pretrained(
    'funnel-transformer/large', num_labels=15,
    max_position_embeddings=1536)
```

For BigBird, we used full attention

```angular2html
model = AutoModelForTokenClassification.from_pretrained(
    'google/bigbird-roberta-base', num_labels=15,
    attention_type="original_full")
```

For YOSO, we disabled lsh_backward
```angular2html
model = AutoModelForTokenClassification.from_pretrained(
    'uw-madison/yoso-4096', num_labels=15,
    lsh_backward=False)
```

## Model Training
Thank you Nvidia for providing us some V100 32GB GPUs and A100 40GB GPUs. For each model, we trained 6 out of 10 KFolds 
and built our CV from the resultant 60% train data OOF. Each fold would take approximately 4 hours on 1xGPU and required 
between 3 and 5 epochs to maximize val score.

For two models, we used my train public notebooks here and here. For the other eight models, we used HuggingFace 
Trainer. Most models, used the following parameters. And we added @robikscube Kaggle metric here and our PP directly to 
the Trainer evaluation, so we could watch the Kaggle metric score every 500 steps and save the model weights with best 
Kaggle score.

Notice how we use FP16, gradient_accumulation_steps, and gradient_checkpointing. With these settings, you can basically 
train any size model on any GPU including Kaggle's P100 16GB GPU.

```angular2html
from transformers import TrainingArguments, Trainer
args = TrainingArguments( NAME,
                     PRETRAINED_MODEL,
                     evaluation_strategy = 'steps',
                     eval_steps = 500,
                     dataloader_num_workers=8,
                     warmup_ratio=0,
                     lr_scheduler_type = 'linear',
                     learning_rate = 2e-5,
                     log_level = 'warning',
                     fp16 = True,
                     per_device_train_batch_size = 2,
                     per_device_eval_batch_size = 2,
                     gradient_accumulation_steps = 2,
                     gradient_checkpointing = True,
                     num_train_epochs = 5,
                     save_strategy = 'no',
                     save_total_limit = 1)

trainer = Trainer(model,
              args,
              train_dataset=train_dataset,
              eval_dataset=valid_dataset,
              compute_metrics=KaggleMetric(valid_df, valid_dataset),
              callbacks=[SaveBestModelCallback],
              data_collator=data_collator,
              tokenizer=tokenizer)
```

## Inference Code with PP with WBF

We posted our final submission inference code on Kaggle here. 

https://www.kaggle.com/cdeotte/2nd-place-solution-cv741-public727-private740

It includes our post process and weighted box fusion implementation. Enjoy!

## Training Code on GitHub

Our training code is published on GitHub here: https://github.com/ubamba98/feedback-prize
