# feedback_prize_pytorch
starting over

# Key points on the competition so far:
- there seems to be consensus that this is an NER problem: the models have to find parts of the text that correspond 
to entities/targets. In this instance, the targets are the various parts of the essay.
- It is clear that each section of the essays on average exceeds the maximum number of characters handled by most 
transformer NLP models. There are limited model choices and consensus has narrowed onto a handful.
- Because of the consensus that this is a long-text NER problem, model choice is an interesting variable. Efforts have 
focused on longformer, and most competitors seem to be using longformer-large-4096, however I have yet to be able to 
replicate or improve the results I'm achieving on longformer-base-4096. Early notebooks looked at bigbird and team names 
indicate that some teams are ensembling bigbird and longformer. I have yet to try bigbird.
- Most model directories show the addition of an NER.csv which is unsurprising given the notes above. How this is 
generated, the quality of the data in this file, and how it is used will have a major impact on performance. 
- The distribution of number of characters in a section of the essay is not normally distributed. It shows both skew and 
kurtosis. The max_length hyperparameter is arbitrarily set at 1024 however it may be that normalizing the distribution, 
taking a mean and standard deviation then reverting would result in a more meaningful number.
- There is a problem with spelling, punctuation and grammar that means that the normal tokenizer dictionary doesn't give 
as good a coverage as it should. This can be augmented or improved.
- post-processing is agreed to be an important part of optimizing performance. I don't understand why NER is often 
mentioned in the context of post-processing. To me, generating the dictionary of named entities is preprocessing.
- tweaking the probabilities used in inference has a measurable impact on performance. This makes sense, but I don't know 
how to optimise this yet.
- from what I've gathered on the hugging face course, this task suits an encoder only model. Longformer has all variants.
- the current framework relies on tez, which I don't like. I need to refactor into pytorch/transformers style code, so 
I have a better change of optimizing it.

So in summary, OFI's are:
1) refactor to pytorch/transformers, get rid of tez. Keep in mind that transformers is built around distributed compute.
2) chose the right model -> longformer-base, longformer-large, bigbird.
3) ensemble -> can i use GBDT for the embeddings?
4) choose the right max_char
5) choose the right LR
6) NER
7) spelling, punctuation & grammar
8) post-processing -> probability_thresholds etc.
9) print CV for the whole training run at the end, rather than manually calculating the average
10) how can I schedule this to start a new training run at completion of the current? keep rolling... Notionally I can 
complete 2-3 runs per day if I do this.


https://github.com/allenai/longforme



100% 390/390 [00:48<00:00,  8.03it/s, stage=test]
  0% 0/1560 [00:00<?, ?it/s]
(0.5841826333216795, {'Claim': 0.5564063848144952, 'Concluding Statement': 0.7694693314955203, 'Counterclaim': 0.41275978733687774, 'Evidence': 0.6981887942221972, 'Lead': 0.7563965884861408, 'Position': 0.6368273819318743, 'Rebuttal': 0.25923016496465046})

100% 390/390 [00:49<00:00,  7.91it/s, stage=test]
(0.592181063167317, {'Claim': 0.5739038189533239, 'Concluding Statement': 0.6950113378684807, 'Counterclaim': 0.42184964845862627, 'Evidence': 0.7286460068865933, 'Lead': 0.7682505972922751, 'Position': 0.66429418742586, 'Rebuttal': 0.29331184528605964})
Validation score improved (0.5841826333216795 --> 0.592181063167317). Saving model!

100% 390/390 [00:48<00:00,  8.05it/s, stage=test]
  0% 0/1560 [00:00<?, ?it/s]
(0.626167517933549, {'Claim': 0.6100343942785391, 'Concluding Statement': 0.8121408045977011, 'Counterclaim': 0.4496521420725009, 'Evidence': 0.6989627013904215, 'Lead': 0.7736943907156673, 'Position': 0.6711782057955926, 'Rebuttal': 0.36750998668442075})

100% 390/390 [00:48<00:00,  8.06it/s, stage=test]
(0.6414914470296118, {'Claim': 0.630400338016267, 'Concluding Statement': 0.8220432076162578, 'Counterclaim': 0.46799116997792495, 'Evidence': 0.7350475125768586, 'Lead': 0.7910817506193228, 'Position': 0.6692689850958127, 'Rebuttal': 0.37460716530483973})
Validation score improved (0.626167517933549 --> 0.6414914470296118). Saving model!



100% 390/390 [00:49<00:00,  7.81it/s, stage=test]
(0.6478409994874595, {'Claim': 0.6204413353222571, 'Concluding Statement': 0.8368487470704885, 'Counterclaim': 0.4722222222222222, 'Evidence': 0.7360847741215839, 'Lead': 0.7839866555462885, 'Position': 0.6761857869686525, 'Rebuttal': 0.4091174751607247})
Validation score improved (0.6414914470296118 --> 0.6478409994874595). Saving model!



100% 390/390 [00:49<00:00,  7.82it/s, stage=test]
  0% 0/1560 [00:00<?, ?it/s]
(0.6357730060293809, {'Claim': 0.6159570509753893, 'Concluding Statement': 0.8136991213914291, 'Counterclaim': 0.449685534591195, 'Evidence': 0.7256617317182593, 'Lead': 0.7931488801054019, 'Position': 0.6756800781886301, 'Rebuttal': 0.37657864523536166})















































































































































































































































































































































































100% 1560/1560 [12:17<00:00,  2.12it/s, f1=0.798, loss=0.286, stage=train]























100% 390/390 [00:48<00:00,  8.11it/s, stage=test]
(0.6458560119979746, {'Claim': 0.6258067694733211, 'Concluding Statement': 0.8404527199707923, 'Counterclaim': 0.46524307133121307, 'Evidence': 0.7278581173260573, 'Lead': 0.7994659546061416, 'Position': 0.6798125101018264, 'Rebuttal': 0.38235294117647056})
EarlyStopping counter: 2 out of 5
























































































































































































































































































































































































100% 1560/1560 [12:37<00:00,  2.06it/s, f1=0.827, loss=0.234, stage=train]
























100% 390/390 [00:48<00:00,  8.06it/s, stage=test]
  0% 1/1560 [00:00<16:49,  1.54it/s, f1=0.847, loss=0.24, stage=train]
(0.6420215468739449, {'Claim': 0.6196699120112459, 'Concluding Statement': 0.8386515233459527, 'Counterclaim': 0.4575902566333188, 'Evidence': 0.7311616740328477, 'Lead': 0.7932310946589106, 'Position': 0.6749226006191951, 'Rebuttal': 0.3789237668161435})
















































































































































































































































































































































































100% 1560/1560 [12:22<00:00,  2.10it/s, f1=0.847, loss=0.195, stage=train]

























100% 390/390 [00:48<00:00,  8.01it/s, stage=test]
(0.641052985093215, {'Claim': 0.6134808853118713, 'Concluding Statement': 0.8260401721664276, 'Counterclaim': 0.4570230607966457, 'Evidence': 0.7265796961400118, 'Lead': 0.8006345848757271, 'Position': 0.6732289817653704, 'Rebuttal': 0.39038351459645104})
EarlyStopping counter: 4 out of 5












































































































































































































































































































































































100% 1560/1560 [12:13<00:00,  2.13it/s, f1=0.861, loss=0.171, stage=train]

























100% 390/390 [00:47<00:00,  8.13it/s, stage=test]
(0.6400299249682968, {'Claim': 0.6096439749608764, 'Concluding Statement': 0.8298904257230106, 'Counterclaim': 0.454661558109834, 'Evidence': 0.7199743781360094, 'Lead': 0.7943900502778513, 'Position': 0.6794027933857762, 'Rebuttal': 0.3922462941847206})
EarlyStopping counter: 5 out of 5
12477 3117
12474 3120
12475 3119
12475 3119
12475 3119
0    28997
2    28968
3    28904
1    28737
4    28687
Name: kfold, dtype: int64
Index(['id', 'discourse_id', 'discourse_start', 'discourse_end',
       'discourse_text', 'discourse_type', 'discourse_type_num',
       'predictionstring', 'kfold'],
      dtype='object')



100% 1040/1040 [00:06<00:00, 151.74it/s]
100% 1040/1040 [00:06<00:00, 153.53it/s]
100% 1040/1040 [00:06<00:00, 160.62it/s]
100% 1040/1040 [00:07<00:00, 143.35it/s]
100% 1040/1040 [00:06<00:00, 153.78it/s]
100% 1039/1039 [00:06<00:00, 156.67it/s]
100% 1039/1039 [00:07<00:00, 146.08it/s]
100% 1040/1040 [00:08<00:00, 129.27it/s]
100% 1039/1039 [00:06<00:00, 148.44it/s]
100% 1039/1039 [00:07<00:00, 143.75it/s]
100% 1039/1039 [00:07<00:00, 137.89it/s]
100% 1039/1039 [00:08<00:00, 128.85it/s]
100% 260/260 [00:00<00:00, 275.04it/s]
100% 260/260 [00:00<00:00, 286.18it/s]
 78% 203/260 [00:00<00:00, 297.40it/s]
100% 260/260 [00:00<00:00, 285.44it/s]
100% 260/260 [00:00<00:00, 308.01it/s]
100% 260/260 [00:01<00:00, 231.68it/s]
100% 260/260 [00:00<00:00, 316.17it/s]
100% 260/260 [00:00<00:00, 296.73it/s]
100% 260/260 [00:00<00:00, 276.51it/s]
100% 260/260 [00:00<00:00, 282.20it/s]
100% 260/260 [00:00<00:00, 270.98it/s]
100% 260/260 [00:01<00:00, 257.65it/s]
100% 260/260 [00:01<00:00, 218.39it/s]
Some weights of the model checkpoint at longformer were not used when initializing LongformerModel: ['lm_head.layer_norm.weight', 'lm_head.dense.weight', 'lm_head.bias', 'lm_head.dense.bias', 'lm_head.decoder.weight', 'lm_head.layer_norm.bias']
- This IS expected if you are initializing LongformerModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing LongformerModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).











































































































































































































































































































































































100% 1560/1560 [12:11<00:00,  2.13it/s, f1=0.298, loss=1.1, stage=train]

























100% 390/390 [00:48<00:00,  8.02it/s, stage=test]
(0.588309318858171, {'Claim': 0.549692532942899, 'Concluding Statement': 0.7540758537841085, 'Counterclaim': 0.4240864642305713, 'Evidence': 0.7193386477708887, 'Lead': 0.7323259579060982, 'Position': 0.6430868167202572, 'Rebuttal': 0.29555895865237364})
Validation score improved (-inf --> 0.588309318858171). Saving model!


















































































































































































































































































































































































100% 1560/1560 [12:25<00:00,  2.09it/s, f1=0.575, loss=0.627, stage=train]
























100% 390/390 [00:48<00:00,  8.00it/s, stage=test]
(0.6217601484817463, {'Claim': 0.5717940907706366, 'Concluding Statement': 0.7812798471824259, 'Counterclaim': 0.46210141139571353, 'Evidence': 0.7287004782429002, 'Lead': 0.7998889814043852, 'Position': 0.6588854183790893, 'Rebuttal': 0.3496708119970739})
Validation score improved (0.588309318858171 --> 0.6217601484817463). Saving model!




















































































































































































































































































































































































100% 1560/1560 [12:28<00:00,  2.08it/s, f1=0.633, loss=0.555, stage=train]
























100% 390/390 [00:48<00:00,  8.09it/s, stage=test]
  0% 1/1560 [00:00<17:31,  1.48it/s, f1=0.494, loss=0.66, stage=train]
(0.63897428768757, {'Claim': 0.6272702629438872, 'Concluding Statement': 0.8140601120733572, 'Counterclaim': 0.4904377390565236, 'Evidence': 0.6691176470588235, 'Lead': 0.785887422569351, 'Position': 0.675263774912075, 'Rebuttal': 0.41078305519897307})












































































































































































































































































































































































100% 1560/1560 [12:13<00:00,  2.13it/s, f1=0.68, loss=0.484, stage=train]
























100% 390/390 [00:47<00:00,  8.14it/s, stage=test]
  0% 0/1560 [00:00<?, ?it/s]
(0.6413993527275804, {'Claim': 0.6057458937743411, 'Concluding Statement': 0.786336702683862, 'Counterclaim': 0.4857017157941047, 'Evidence': 0.72955474784189, 'Lead': 0.7990249187432286, 'Position': 0.67660734327401, 'Rebuttal': 0.4068241469816273})














































































































































































































































































































































































100% 1560/1560 [12:16<00:00,  2.12it/s, f1=0.72, loss=0.417, stage=train]
























100% 390/390 [00:48<00:00,  8.12it/s, stage=test]
(0.6517966553349905, {'Claim': 0.6282304044333138, 'Concluding Statement': 0.8127029953085528, 'Counterclaim': 0.5018382352941176, 'Evidence': 0.7446271253016105, 'Lead': 0.7841385087964255, 'Position': 0.6804023361453602, 'Rebuttal': 0.4106369820655535})
Validation score improved (0.6413993527275804 --> 0.6517966553349905). Saving model!












































































































































































































































































































































































100% 1560/1560 [12:13<00:00,  2.13it/s, f1=0.759, loss=0.355, stage=train]
























100% 390/390 [00:47<00:00,  8.13it/s, stage=test]
  0% 0/1560 [00:00<?, ?it/s]
(0.6485703462417225, {'Claim': 0.62576877358003, 'Concluding Statement': 0.8021689697393738, 'Counterclaim': 0.4834193072955048, 'Evidence': 0.7221435450065145, 'Lead': 0.8123655913978495, 'Position': 0.6795727636849133, 'Rebuttal': 0.4145534729878721})












































































































































































































































































































































































100% 1560/1560 [12:12<00:00,  2.13it/s, f1=0.796, loss=0.292, stage=train]























100% 390/390 [00:47<00:00,  8.13it/s, stage=test]
(0.6519340223507442, {'Claim': 0.6138656364026616, 'Concluding Statement': 0.8291038154392192, 'Counterclaim': 0.49815346737792365, 'Evidence': 0.7236691236691236, 'Lead': 0.8062227074235808, 'Position': 0.688935962196513, 'Rebuttal': 0.40358744394618834})
EarlyStopping counter: 2 out of 5











































































































































































































































































































































































100% 1560/1560 [12:12<00:00,  2.13it/s, f1=0.821, loss=0.24, stage=train]
























100% 390/390 [00:47<00:00,  8.13it/s, stage=test]
  0% 0/1560 [00:00<?, ?it/s]
(0.646185168408021, {'Claim': 0.6136020151133501, 'Concluding Statement': 0.8228882833787466, 'Counterclaim': 0.48892603426661096, 'Evidence': 0.7265256929987948, 'Lead': 0.7897914379802415, 'Position': 0.6738373932299905, 'Rebuttal': 0.40772532188841204})

















































































































































































































































































































































































100% 1560/1560 [12:23<00:00,  2.10it/s, f1=0.845, loss=0.199, stage=train]
























100% 390/390 [00:48<00:00,  8.04it/s, stage=test]
(0.6467958616827596, {'Claim': 0.606426722168638, 'Concluding Statement': 0.8185383244206773, 'Counterclaim': 0.4940107393638992, 'Evidence': 0.7209592534317183, 'Lead': 0.8052364413572001, 'Position': 0.6786175710594315, 'Rebuttal': 0.40378197997775306})
EarlyStopping counter: 4 out of 5














































































































































































































































































































































































100% 1560/1560 [12:16<00:00,  2.12it/s, f1=0.859, loss=0.175, stage=train]

























100% 390/390 [00:48<00:00,  8.10it/s, stage=test]
(0.6446960892204688, {'Claim': 0.6057840170643385, 'Concluding Statement': 0.8209169054441261, 'Counterclaim': 0.48932676518883417, 'Evidence': 0.715527950310559, 'Lead': 0.8004291845493562, 'Position': 0.6727681438664097, 'Rebuttal': 0.4081196581196581})
EarlyStopping counter: 5 out of 5
12477 3117
12474 3120
12475 3119
12475 3119
12475 3119
0    28997
2    28968
3    28904
1    28737
4    28687
Name: kfold, dtype: int64
Index(['id', 'discourse_id', 'discourse_start', 'discourse_end',
       'discourse_text', 'discourse_type', 'discourse_type_num',
       'predictionstring', 'kfold'],
      dtype='object')


100% 1040/1040 [00:06<00:00, 153.45it/s]
100% 1040/1040 [00:06<00:00, 152.04it/s]
100% 1040/1040 [00:07<00:00, 139.46it/s]
100% 1040/1040 [00:07<00:00, 145.25it/s]
100% 1040/1040 [00:06<00:00, 154.84it/s]
100% 1040/1040 [00:06<00:00, 152.29it/s]
100% 1039/1039 [00:07<00:00, 142.52it/s]
100% 1039/1039 [00:07<00:00, 142.52it/s]
100% 1039/1039 [00:07<00:00, 143.40it/s]
100% 1039/1039 [00:07<00:00, 140.08it/s]
100% 1039/1039 [00:07<00:00, 138.36it/s]
100% 1039/1039 [00:08<00:00, 124.93it/s]
100% 260/260 [00:01<00:00, 253.85it/s]
100% 260/260 [00:00<00:00, 284.33it/s]
100% 260/260 [00:00<00:00, 274.45it/s]
100% 260/260 [00:00<00:00, 326.19it/s]
100% 260/260 [00:00<00:00, 294.42it/s]
100% 260/260 [00:01<00:00, 215.95it/s]
100% 260/260 [00:00<00:00, 286.50it/s]
100% 260/260 [00:00<00:00, 263.72it/s]
100% 260/260 [00:01<00:00, 259.66it/s]
100% 260/260 [00:00<00:00, 272.59it/s]
100% 259/259 [00:01<00:00, 248.78it/s]
100% 260/260 [00:01<00:00, 211.69it/s]
Some weights of the model checkpoint at longformer were not used when initializing LongformerModel: ['lm_head.layer_norm.weight', 'lm_head.dense.weight', 'lm_head.bias', 'lm_head.dense.bias', 'lm_head.decoder.weight', 'lm_head.layer_norm.bias']
- This IS expected if you are initializing LongformerModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing LongformerModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
  0% 0/1560 [00:00<?, ?it/s]
  0% 3/1560 [00:01<13:36,  1.91it/s, f1=0.023, loss=3.16, stage=train]
  1% 8/1560 [00:04<12:14,  2.11it/s, f1=0.0256, loss=3.14, stage=train]
  1% 12/1560 [00:05<12:05,  2.13it/s, f1=0.0261, loss=3.14, stage=train]
  1% 16/1560 [00:07<12:02,  2.14it/s, f1=0.0257, loss=3.14, stage=train]
  1% 21/1560 [00:10<11:59,  2.14it/s, f1=0.0252, loss=3.15, stage=train]
  2% 25/1560 [00:11<11:59,  2.13it/s, f1=0.025, loss=3.14, stage=train]
  2% 29/1560 [00:13<11:57,  2.13it/s, f1=0.0249, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.5666474951354751, {'Claim': 0.5531210772566473, 'Concluding Statement': 0.7722563652326603, 'Counterclaim': 0.3521388716676999, 'Evidence': 0.7238472418670439, 'Lead': 0.7601332593003887, 'Position': 0.6383689839572193, 'Rebuttal': 0.16666666666666666})
Validation score improved (-inf --> 0.5666474951354751). Saving model!
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6323885668878628, {'Claim': 0.6202226440260606, 'Concluding Statement': 0.8421248142644874, 'Counterclaim': 0.4288681204569055, 'Evidence': 0.7404410507721237, 'Lead': 0.796281540504648, 'Position': 0.6465090709180868, 'Rebuttal': 0.3522727272727273})
Validation score improved (0.5666474951354751 --> 0.6323885668878628). Saving model!
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6482565297545373, {'Claim': 0.6238581992170509, 'Concluding Statement': 0.8273692489186001, 'Counterclaim': 0.4827930174563591, 'Evidence': 0.7585155058464667, 'Lead': 0.8028438610883237, 'Position': 0.6761000862812769, 'Rebuttal': 0.3663157894736842})
Validation score improved (0.6323885668878628 --> 0.6482565297545373). Saving model!
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6529634800115508, {'Claim': 0.6333245312912595, 'Concluding Statement': 0.8509695290858725, 'Counterclaim': 0.48921523634694813, 'Evidence': 0.7351908218215875, 'Lead': 0.7940348902644907, 'Position': 0.6811881188118812, 'Rebuttal': 0.38682123245881633})
Validation score improved (0.6482565297545373 --> 0.6529634800115508). Saving model!
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.649598469385794, {'Claim': 0.6178798782175478, 'Concluding Statement': 0.8547486033519553, 'Counterclaim': 0.4798973481608212, 'Evidence': 0.7486884413606363, 'Lead': 0.7936333699231614, 'Position': 0.676441186271243, 'Rebuttal': 0.3759004584151932})
EarlyStopping counter: 1 out of 5
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6506526910302703, {'Claim': 0.633486472690148, 'Concluding Statement': 0.8294529623320084, 'Counterclaim': 0.4876867178037949, 'Evidence': 0.7387154727150236, 'Lead': 0.8012718600953895, 'Position': 0.661745406824147, 'Rebuttal': 0.4022099447513812})
EarlyStopping counter: 2 out of 5
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6555061738440559, {'Claim': 0.6279877425944842, 'Concluding Statement': 0.8505832253286428, 'Counterclaim': 0.49211087420042643, 'Evidence': 0.7408056042031523, 'Lead': 0.8135235076597993, 'Position': 0.6812478806375042, 'Rebuttal': 0.3822843822843823})
Validation score improved (0.6529634800115508 --> 0.6555061738440559). Saving model!
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6486368976324262, {'Claim': 0.624464588106385, 'Concluding Statement': 0.8448625714812765, 'Counterclaim': 0.48267008985879334, 'Evidence': 0.7300690179881528, 'Lead': 0.8033315421816228, 'Position': 0.676968601019234, 'Rebuttal': 0.37809187279151946})
EarlyStopping counter: 1 out of 5
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6465194890709745, {'Claim': 0.6200378071833649, 'Concluding Statement': 0.8400658014988119, 'Counterclaim': 0.48113590263691686, 'Evidence': 0.7283174327840416, 'Lead': 0.801906779661017, 'Position': 0.66645056726094, 'Rebuttal': 0.3877221324717286})
EarlyStopping counter: 2 out of 5
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6434725725204382, {'Claim': 0.6124216183935504, 'Concluding Statement': 0.83327239488117, 'Counterclaim': 0.4815265935850589, 'Evidence': 0.7283199655487969, 'Lead': 0.8012752391073327, 'Position': 0.6689091782796732, 'Rebuttal': 0.3785830178474851})
EarlyStopping counter: 3 out of 5
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6439010186417372, {'Claim': 0.6151059767976895, 'Concluding Statement': 0.8334247577253612, 'Counterclaim': 0.4798685291700904, 'Evidence': 0.7253657487091222, 'Lead': 0.8029739776951673, 'Position': 0.6657032755298651, 'Rebuttal': 0.3848648648648649})
EarlyStopping counter: 4 out of 5
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
(0.6437244539448452, {'Claim': 0.6136509030067417, 'Concluding Statement': 0.8331807361289141, 'Counterclaim': 0.48344370860927155, 'Evidence': 0.7278504874239242, 'Lead': 0.7993647432503971, 'Position': 0.6675245295158436, 'Rebuttal': 0.38105606967882416})
EarlyStopping counter: 5 out of 5
12477 3117
12474 3120
12475 3119
12475 3119
12475 3119
0    28997
2    28968
3    28904
1    28737
4    28687
Name: kfold, dtype: int64
Index(['id', 'discourse_id', 'discourse_start', 'discourse_end',
       'discourse_text', 'discourse_type', 'discourse_type_num',
       'predictionstring', 'kfold'],
      dtype='object')
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
  2% 33/1560 [00:15<11:55,  2.14it/s, f1=0.0247, loss=3.14, stage=train]
100% 1039/1039 [00:07<00:00, 142.78it/s]=0.0247, loss=3.14, stage=train]
100% 1039/1039 [00:07<00:00, 142.78it/s]=0.0247, loss=3.14, stage=train]
100% 1039/1039 [00:07<00:00, 142.78it/s]=0.0247, loss=3.14, stage=train]
100% 1039/1039 [00:07<00:00, 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.603826138801785, {'Claim': 0.5664507706983983, 'Concluding Statement': 0.7908370044052864, 'Counterclaim': 0.4216614090431125, 'Evidence': 0.6998090102235703, 'Lead': 0.764063811922754, 'Position': 0.6460792239288602, 'Rebuttal': 0.3378817413905133})
Validation score improved (-inf --> 0.603826138801785). Saving model!
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.6134919364399519, {'Claim': 0.5696649029982364, 'Concluding Statement': 0.8104291146116241, 'Counterclaim': 0.42957366432811656, 'Evidence': 0.7213818141811981, 'Lead': 0.8045664582767056, 'Position': 0.6559836345039209, 'Rebuttal': 0.30284396617986165})
Validation score improved (0.603826138801785 --> 0.6134919364399519). Saving model!
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.643475204398003, {'Claim': 0.612694877505568, 'Concluding Statement': 0.8248712288447387, 'Counterclaim': 0.49834827748938176, 'Evidence': 0.7506474074926627, 'Lead': 0.7673561915556815, 'Position': 0.6694560669456067, 'Rebuttal': 0.38095238095238093})
Validation score improved (0.6134919364399519 --> 0.643475204398003). Saving model!
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.6517860334422788, {'Claim': 0.6360427696460085, 'Concluding Statement': 0.836, 'Counterclaim': 0.4726166328600406, 'Evidence': 0.7546916139151868, 'Lead': 0.8061252392671588, 'Position': 0.6850961538461539, 'Rebuttal': 0.3719298245614035})
Validation score improved (0.643475204398003 --> 0.6517860334422788). Saving model!
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.6569487522235408, {'Claim': 0.6316558018252934, 'Concluding Statement': 0.8228720432110262, 'Counterclaim': 0.4858599907278628, 'Evidence': 0.7546895122907218, 'Lead': 0.808487486398259, 'Position': 0.6942612137203166, 'Rebuttal': 0.4008152173913043})
Validation score improved (0.6517860334422788 --> 0.6569487522235408). Saving model!
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.6541133471527323, {'Claim': 0.626278197420482, 'Concluding Statement': 0.8326212149359093, 'Counterclaim': 0.4898663216903838, 'Evidence': 0.7401888294847002, 'Lead': 0.8019323671497585, 'Position': 0.6926229508196722, 'Rebuttal': 0.3952835485682201})
EarlyStopping counter: 1 out of 5
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.647192309461339, {'Claim': 0.6197419490191964, 'Concluding Statement': 0.8244525547445255, 'Counterclaim': 0.48832759280520477, 'Evidence': 0.7315841475463827, 'Lead': 0.7951153324287653, 'Position': 0.6852284094698233, 'Rebuttal': 0.385896180215475})
EarlyStopping counter: 2 out of 5
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.650574030784594, {'Claim': 0.6217407081022238, 'Concluding Statement': 0.8249863412857403, 'Counterclaim': 0.4839675074818298, 'Evidence': 0.7346294178511137, 'Lead': 0.7953500946201676, 'Position': 0.692049647769205, 'Rebuttal': 0.40129449838187703})
EarlyStopping counter: 3 out of 5
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.6475952625003268, {'Claim': 0.6226790129441994, 'Concluding Statement': 0.81948216548977, 'Counterclaim': 0.48029256399837467, 'Evidence': 0.729471467904767, 'Lead': 0.7903010924593659, 'Position': 0.6873578160545987, 'Rebuttal': 0.4035827186512118})
EarlyStopping counter: 4 out of 5
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
(0.6450492252314602, {'Claim': 0.6155516084468127, 'Concluding Statement': 0.8175209014903672, 'Counterclaim': 0.4814509480626546, 'Evidence': 0.7257320604955486, 'Lead': 0.7896579585241045, 'Position': 0.6820977662674005, 'Rebuttal': 0.4033333333333333})
EarlyStopping counter: 5 out of 5
12477 3117
12474 3120
12475 3119
12475 3119
12475 3119
0    28997
2    28968
3    28904
1    28737
4    28687
Name: kfold, dtype: int64
Index(['id', 'discourse_id', 'discourse_start', 'discourse_end',
       'discourse_text', 'discourse_type', 'discourse_type_num',
       'predictionstring', 'kfold'],
      dtype='object')
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
  0% 0/1560 [00:00<?, ?it/s] 142.78it/s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
(0.559987971477731, {'Claim': 0.5516122278056952, 'Concluding Statement': 0.7794010226442659, 'Counterclaim': 0.38912579957356075, 'Evidence': 0.6421196797302992, 'Lead': 0.721081081081081, 'Position': 0.625106455459036, 'Rebuttal': 0.2114695340501792})
Validation score improved (-inf --> 0.559987971477731). Saving model!
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
(0.5986673833238801, {'Claim': 0.6196229085578527, 'Concluding Statement': 0.7976210705182668, 'Counterclaim': 0.3901264298615292, 'Evidence': 0.7021288546874123, 'Lead': 0.7756339581036383, 'Position': 0.6582857142857143, 'Rebuttal': 0.24725274725274726})
Validation score improved (0.559987971477731 --> 0.5986673833238801). Saving model!
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
(0.6431836670285783, {'Claim': 0.6085199528540158, 'Concluding Statement': 0.8304848273456574, 'Counterclaim': 0.4857496902106567, 'Evidence': 0.7358196010407633, 'Lead': 0.8023095958207314, 'Position': 0.6712778834514957, 'Rebuttal': 0.36812411847672777})
Validation score improved (0.5986673833238801 --> 0.6431836670285783). Saving model!
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
(0.6505941364454894, {'Claim': 0.6166168424615037, 'Concluding Statement': 0.8378823960432313, 'Counterclaim': 0.4892578125, 'Evidence': 0.7418215505761102, 'Lead': 0.8054945054945055, 'Position': 0.6753612971448714, 'Rebuttal': 0.38772455089820357})
Validation score improved (0.6431836670285783 --> 0.6505941364454894). Saving model!
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
(0.6505753235892521, {'Claim': 0.630526258265479, 'Concluding Statement': 0.828221946463393, 'Counterclaim': 0.5024727992087042, 'Evidence': 0.7046546297830047, 'Lead': 0.7969009407858328, 'Position': 0.6687541199736322, 'Rebuttal': 0.4224965706447188})
EarlyStopping counter: 1 out of 5
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]
(0.6561794786694257, {'Claim': 0.6227809852695192, 'Concluding Statement': 0.8288522158353421, 'Counterclaim': 0.5085555113410266, 'Evidence': 0.7307542008597109, 'Lead': 0.8074433656957929, 'Position': 0.6747311827956989, 'Rebuttal': 0.4201388888888889})
Validation score improved (0.6505941364454894 --> 0.6561794786694257). Saving model!


(0.6533699322610094, {'Claim': 0.6201424211597152, 'Concluding Statement': 0.8220724822436715, 'Counterclaim': 0.5079365079365079, 'Evidence': 0.7279723399509257, 'Lead': 0.8030059044551798, 'Position': 0.6757932034143984, 'Rebuttal': 0.4166666666666667})
EarlyStopping counter: 1 out of 5


(0.6499253318656446, {'Claim': 0.6165206261691444, 'Concluding Statement': 0.8269090909090909, 'Counterclaim': 0.4975566414926699, 'Evidence': 0.7173370920524775, 'Lead': 0.8, 'Position': 0.673887685385769, 'Rebuttal': 0.4172661870503597})
EarlyStopping counter: 2 out of 5

(0.6516461869994894, {'Claim': 0.6160270431497316, 'Concluding Statement': 0.8251039971061674, 'Counterclaim': 0.5076271186440678, 'Evidence': 0.716227191510098, 'Lead': 0.8012769353551477, 'Position': 0.6739095184044106, 'Rebuttal': 0.42135150482680295})
EarlyStopping counter: 3 out of 5

(0.6501158613053689, {'Claim': 0.6129080488412659, 'Concluding Statement': 0.8293212669683258, 'Counterclaim': 0.4997892962494732, 'Evidence': 0.7179182637801652, 'Lead': 0.7995718490767996, 'Position': 0.6727243225701769, 'Rebuttal': 0.41857798165137616})
EarlyStopping counter: 4 out of 5

(0.6499231729207512, {'Claim': 0.6138722009830694, 'Concluding Statement': 0.8288876791870804, 'Counterclaim': 0.49746621621621623, 'Evidence': 0.721488271771367, 'Lead': 0.7965830218900161, 'Position': 0.6698067240539224, 'Rebuttal': 0.4213580963435868})
EarlyStopping counter: 5 out of 5
100% 260/260 [00:00<00:00, 279.58it/s]s]=0.0247, loss=3.14, stage=train]