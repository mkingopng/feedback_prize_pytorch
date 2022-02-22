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


https://github.com/allenai/longformer