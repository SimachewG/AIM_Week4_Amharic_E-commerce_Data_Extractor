import os
import time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Features, Sequence, ClassLabel, Value
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, AutoConfig,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
from seqeval.metrics import f1_score

# ----------------------
# Configurations
# ----------------------
MODEL_NAMES = {
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
    "distilbert-multilingual": "Davlan/distilbert-base-multilingual-cased-ner-hrl",
    "mbert": "bert-base-multilingual-cased"
}

DATA_PATH = "data/processed/labeled_data.conll"
MAX_LEN = 128
NUM_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 5e-5


# ----------------------
# 1. Read CoNLL Data
# ----------------------
def read_conll(filepath):
    tokens, tags, all_data = [], [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    all_data.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                splits = line.split()
                if len(splits) == 2:
                    tokens.append(splits[0])
                    tags.append(splits[1])
    return Dataset.from_list(all_data)

raw_dataset = read_conll(DATA_PATH)

# ----------------------
# 2. Label Handling
# ----------------------
unique_labels = sorted(set(tag for example in raw_dataset["ner_tags"] for tag in example))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

features = Features({
    "tokens": [Value("string")],
    "ner_tags": Sequence(ClassLabel(names=unique_labels))
})
dataset = raw_dataset.cast(features)
dataset = dataset.train_test_split(test_size=0.2)
train_ds, test_ds = dataset["train"], dataset["test"]

# ----------------------
# 3. Tokenization Helper
# ----------------------
def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LEN,
        padding="max_length"
    )

    labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_value = label_seq[word_idx]
                if isinstance(label_value, int):
                    label_ids.append(label_value)
                else:
                    label_ids.append(label2id[label_value])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# ----------------------
# 4. Evaluation Function
# ----------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    true_preds = [
        [id2label[p] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(preds, labels)
    ]
    return {"f1": f1_score(true_labels, true_preds)}

# ----------------------
# 5. Loop over Models
# ----------------------
results = []
for model_key, model_name in MODEL_NAMES.items():
    print(f"\n🔄 Training model: {model_key} ({model_name})")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ✅ Fix classifier head mismatch by using config and ignore mismatched head
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True  # 🔥 This line fixes the size mismatch
    )

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(tokenizer, x),
        batched=True
    )

    training_args = TrainingArguments(
        output_dir=f"./outputs_{model_key}",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_strategy="no",
        logging_dir=f"./logs_{model_key}",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    runtime = time.time() - start_time

    metrics["model"] = model_key
    metrics["runtime"] = round(runtime, 2)
    results.append(metrics)

# ----------------------
# 6. Save & Show Results
# ----------------------
df_results = pd.DataFrame(results)
df_results = df_results[["model", "eval_loss", "eval_f1", "runtime"]]
df_results.sort_values(by="eval_f1", ascending=False, inplace=True)
df_results.reset_index(drop=True, inplace=True)

df_results.to_csv("model_comparison.csv", index=False)
print("\n✅ Model comparison complete. Results saved to 'model_comparison.csv':\n")
print(df_results)


