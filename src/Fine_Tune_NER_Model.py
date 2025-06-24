# 1. Install dependencies (uncomment if running in a new environment)
# !pip install torch transformers datasets seqeval --quiet

# 2. Imports
import os
import numpy as np
from datasets import Dataset, Features, Value, Sequence, ClassLabel
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments,
    Trainer, pipeline
)
from seqeval.metrics import classification_report, f1_score

# 3. Configuration
MODEL_NAME = "FacebookAI/xlm-roberta-base"  # ✅ Alternative: "rasyosef/bert-tiny-amharic"
DATA_PATH = "data/processed/labeled_data.conll"
NUM_EPOCHS = 4
BATCH_SIZE = 8
MAX_LEN = 128
LEARNING_RATE = 5e-5
OUTPUT_DIR = "data/processed/ner_amharic_finetuned"

# 4. Load CoNLL-format data
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

# 5. Label mapping
unique_labels = sorted(set(tag for example in raw_dataset["ner_tags"] for tag in example))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Define features and convert to Hugging Face dataset
features = Features({
    "tokens": [Value("string")],
    "ner_tags": Sequence(ClassLabel(names=unique_labels))
})
dataset = raw_dataset.cast(features).train_test_split(test_size=0.2)

# 6. Tokenization and alignment
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LEN
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                lbl = label[word_idx]
                label_ids.append(label2id.get(lbl, -100) if isinstance(lbl, str) else lbl)
            else:
                label_ids.append(
                    label2id[label[word_idx]] if isinstance(label[word_idx], str) and label[word_idx].startswith("I-") else -100
                )
            prev_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# 7. Load pre-trained model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# 8. TrainingArguments with no checkpoint saving
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",  # ✅ Avoids saving optimizer/scheduler
    report_to="none"     # Disable wandb integration
)

# 9. Evaluation function
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

# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

# 11. Train model
trainer.train()

# 12. Evaluate
print(trainer.evaluate())
predictions = trainer.predict(tokenized_dataset["test"])
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = [[id2label[t] for t in row if t != -100] for row in predictions.label_ids]
true_preds = [[id2label[p] for p, t in zip(pred_row, label_row) if t != -100]
              for pred_row, label_row in zip(pred_labels, predictions.label_ids)]

print(classification_report(true_labels, true_preds))

# 13. Save model and tokenizer
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 14. Inference
ner = pipeline("token-classification", model=OUTPUT_DIR, tokenizer=OUTPUT_DIR, aggregation_strategy="simple")
print(ner("እቃ ዋጋ 500 ብር ቦሌ"))
print(ner("ደረጃ እንደወጡ የቢሮ ቁጥር ያገኙናል"))

