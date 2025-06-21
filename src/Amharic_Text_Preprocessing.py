import pandas as pd
import re

from etnltk.lang.am.normalizer import (
    normalize_labialized,
    normalize_shortened,
    normalize_punct,
    normalize_char,
)
from etnltk.tokenize.am import word_tokenize

# Optional: emoji cleaner
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\u2600-\u26FF"
        u"\u2700-\u27BF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# --------------------------
# Amharic Preprocessing Function
# --------------------------
def preprocess_amharic_message(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Step 1: Remove emojis (optional)
    text = remove_emoji(text)

    # Step 2: Normalize in stages
    text = normalize_labialized(text)
    text = normalize_shortened(text)
    text = normalize_punct(text)
    text = normalize_char(text)

    # Step 3: Tokenize
    tokens = word_tokenize(text)

    return ' '.join(tokens)

# --------------------------
# Apply to Dataset
# --------------------------
input_path = 'data/processed/clean_telegram_data.csv'
output_path = 'data/processed/final_amharic_preprocessed.csv'

df = pd.read_csv(input_path)
df['Message'] = df['Message'].astype(str).apply(preprocess_amharic_message)

print(df[['Channel Username', 'Message']].head())
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"✅ Preprocessed Amharic messages saved to {output_path}")




