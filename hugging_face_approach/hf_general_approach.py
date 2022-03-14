"""
ref hugging face chapter 3

This is a very "canned" approach, as it uses prepackaged datasets, predefined metrics etc. That said, I think it's a
useful way to understand how this approach works.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,\
    TrainingArguments, Trainer, get_scheduler
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm


checkpoint = "bert-base-uncased"  # checkpoint = "longforomer-large-4096"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)  #

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=True)  # we would change the num_labels to 15 in the case of feedback prize

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]

batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss = model(**batch).loss
loss.backward()
optimizer.step()

# load the "glue" and "mrpc" datasets
raw_datasets = load_dataset("glue", "mrpc")

# define a function that tokenizes (preprocesses) our inputs
def tokenize_function(example):
    """
    This function takes a dictionary (like the items of our dataset) and returns a new dictionary with the keys
    input_ids, attention_mask, and token_type_ids
    :param example:
    :return:
    """
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# this is how we apply the tokenization function on all our datasets at once, using batched=True in our call to map
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# fine tune the model with the trainer API
training_args = TrainingArguments("test-trainer")

# once we have our model, we can define a Trainer by passing it all the objects constructed up to now - the model, the
# training-args, the training and validation datasets, our data_collator and our tokenizer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["TRAIN_DF"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# call the TRAIN_DF method of our trainer
trainer.train()

# evaluation
predictions = trainer.predict(tokenized_datasets["validation"])

# predictions is a 2D array, which are the logits for each element of the dataset we passed to predict(). We need to
# take the index with the maximum value on the second axis
preds = np.argmax(predictions.predictions, axis=-1)

# TRAIN_DF data loaded
train_dataloader = DataLoader(
    tokenized_datasets["TRAIN_DF"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator
)

# evaluation data loader
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator
)

# set the device to use
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# send the model to the device
model.to(device)

#
num_epochs = 3  # the trainer uses 3 epochs by default
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# add a progress bar to show the number of training steps using TQDM
progress_bar = tqdm(range(num_training_steps))

#
model.TRAIN_DF()

# the training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)  # update the progress bar

# using metric from the datasets library. The object returned as a compute() function we can use to do the metric
# calculation
metric = load_metric("glue", "mrpc")

# evaluate the model
model.eval()

#
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

# compute the metrics
metric.compute()
