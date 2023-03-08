import random
from datasets import load_dataset

NUM_SAMPLES_PER_CLASS = 8

sst2 = load_dataset("glue", "sst2")

# select sample of train samples for each class
labels = sst2["train"]["label"]
zero_labels = [idx for idx, val in enumerate(labels) if val == 0]
one_labels = [idx for idx, val in enumerate(labels) if val == 1]

zero_sample = random.sample(zero_labels, NUM_SAMPLES_PER_CLASS)
one_sample = random.sample(one_labels, NUM_SAMPLES_PER_CLASS)
sample_idxs = zero_sample + one_sample

sample_train_dataset = sst2["train"].select(sample_idxs)

# save CSVs
sample_train_dataset.to_csv(f"sst2-train-{NUM_SAMPLES_PER_CLASS}-samples.csv")
sst2["validation"].to_csv("sst2-validation.csv")
