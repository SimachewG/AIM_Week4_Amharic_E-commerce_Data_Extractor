import re
import os
import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
INPUT_PATH = "data/processed/final_amharic_preprocessed.csv"
OUTPUT_PATH = "data/processed/labeled_data.conll"

# -------------------------------
# Entity Patterns 
# -------------------------------
location_keywords = [
    'Addis Ababa', 'ቦሌ', 'ሜክሲኮ', 'ብስራተ', 'ገብርኤል', 'ገርጂ',
    'ኢምፔሪያል', 'ህንፃ', 'ፕላዛ', '4ኪሎ', 'ፎቅ', 'ላፍቶ', 'ሞል', 'ስላሴ',
    'መገናኛ', 'ለቡ', 'ለቡ መዳህኒዓለም', 'ዘፍመሽ', 'ኬኬር', 'አዲስ አበባ', 
    'ኢምፔሪያል', 'ባልቻ', 'ሆስፒታል', 'ልደታ', 'ኮሜርስ', 'ራሀ', 'ሞል',
     'ጉርድሾላ', 'ከሴንቸሪ', 'አልፎዝ', 'ቅድስት', 'አህመድ ህንፃ', 'መዚድ ፕላዛ', 
]

# -------------------------------
# Function: Label Single Message
# -------------------------------
def label_message_conll(message):
    if not isinstance(message, str) or message.strip() == "":
        return []

    labeled_tokens = []

    # Split first line and remaining
    if '\n' in message:
        first_line, rest = message.split('\n', 1)
    else:
        first_line, rest = message, ""

    # Label first line as PRODUCT
    first_tokens = re.findall(r'\S+', first_line)
    if first_tokens:
        labeled_tokens.append((first_tokens[0], "B-PRODUCT"))
        for token in first_tokens[1:]:
            labeled_tokens.append((token, "I-PRODUCT"))

    # Process remaining lines
    for line in rest.split('\n'):
        tokens = re.findall(r'\S+', line)
        for token in tokens:
            if re.match(r'^\d{10}$', token):
                label = "I-PHONE"
            elif re.match(r'^\d+(\.\d{1,2})?$', token) or any(x in token for x in ['ETB', '$', 'birr', 'ብር', 'ዋጋ']):
                label = "I-PRICE"
            elif any(loc in token for loc in location_keywords):
                label = "I-LOC"
            else:
                label = "O"
            labeled_tokens.append((token, label))

    return labeled_tokens

# -------------------------------
# Save to CoNLL format
# -------------------------------
def save_conll(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sentence in data:
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            f.write("\n")

# -------------------------------
# Load & Process Messages
# -------------------------------
df = pd.read_csv(INPUT_PATH)
messages = df['Message'].dropna().astype(str)

labeled_dataset = []
for i, msg in enumerate(messages[:50]):  # Only label 50 messages for demo
    labeled_sentence = label_message_conll(msg)
    labeled_dataset.append(labeled_sentence)

save_conll(labeled_dataset, OUTPUT_PATH)

print(f"✅ Labeled {len(labeled_dataset)} messages and saved to {OUTPUT_PATH}")



