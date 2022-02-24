# feedback_prize_pytorch

Writing is a critical skill for success. However, less than a third of high school seniors are proficient writers, according to the National Assessment of Educational Progress. Unfortunately, low-income, Black, and Hispanic students fare even worse, with less than 15 percent demonstrating writing proficiency. One way to help students improve their writing is via automated feedback tools, which evaluate student writing and provide personalized feedback.

There are currently numerous automated writing feedback tools, but they all have limitations. Many often fail to identify writing structures, such as thesis statements and support for claims, in essays or do not do so thoroughly. Additionally, the majority of the available tools are proprietary, with algorithms and feature claims that cannot be independently backed up. More importantly, many of these writing tools are inaccessible to educators because of their cost. This problem is compounded for under-serviced schools which serve a disproportionate number of students of color and from low-income backgrounds. In short, the field of automated writing feedback is ripe for innovation that could help democratize education.

Georgia State University (GSU) is an undergraduate and graduate urban public research institution in Atlanta. U.S. News & World Report ranked GSU as one of the most innovative universities in the nation. GSU awards more bachelor’s degrees to African-Americans than any other non-profit college or university in the country. GSU and The Learning Agency Lab, an independent nonprofit based in Arizona, are focused on developing science of learning-based tools and programs for social good.

In this competition, you’ll identify elements in student writing. More specifically, you will automatically segment texts and classify argumentative and rhetorical elements in essays written by 6th-12th grade students. You'll have access to the largest dataset of student writing ever released in order to test your skills in natural language processing, a fast-growing area of data science.

If successful, you'll make it easier for students to receive feedback on their writing and increase opportunities to improve writing outcomes. Virtual writing tutors and automated writing systems can leverage these algorithms while teachers may use them to reduce grading time. The open-sourced algorithms you come up with will allow any educational organization to better help young writers develop.

### Data can be downloaded in the usual manner via the kaggle api:
```kaggle competitions download -c feedback-prize-2021```

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
3) named entity recognition (NER) -> NER_train.csv
4) spelling, punctuation & grammar (SPG) -> clean_word_dict.csv
5) post-processing -> probability_thresholds etc.
6) ensemble -> can i use GBDT to enhance?
7) tuning: choose the right max_char, LR and other hyperparameters
8) print CV for the whole training run at the end, rather than manually calculating the average

# NB: I ran inference on my own checkpoint files and  got a surprisingly good result: 0.666. 
This is surprising as it's higher than the average for either of the checkpoints I used 
(visionary-cherry and valiant-elevator). Also, both were trained on longformer-base-4096 using the current 
"new-baseline" files. Both used LR = 3e-5. So far, it seems like LR = 5e-5 is promising on the same model. It is tedious 
testing different learningg rates by training each 3 times. It would be nice to be able to use wandb or another library, 
but the integration with tez is an issue. I doubt this will be possible until the new functions are working.

Most people are using longformer-large-4096. I can't get it to deliver average f1 superior to longformer base yet. I
wonder why.

Original models can be found at:
- https://github.com/allenai/longformer
- https://huggingface.co/allenai/longformer-base-4096/tree/main
- https://huggingface.co/allenai/longformer-large-4096/tree/main
