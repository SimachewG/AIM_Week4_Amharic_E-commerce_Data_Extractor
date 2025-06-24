# Amharic E-commerce Data Extractor
## Step-by-Step Instructions
# Task-1
1.	Clone the repository to your local machine using:
2.	Navigate to the project folder:
cd AIM_Week4_Amharic_E-commerce_Data_Extractor
3.	Create a virtual environment to isolate dependencies:
4.	Activate the virtual environment:
5.	Install all required packages by running:
pip install -r requirements.txt
6.	Create a .env file in the root directory of the project and add my Telegram API credentials like this:
7.	Run the Telegram scraper script to collect up to 1000 messages per channel and save them in data/raw/telegram_data.csv:
This script also downloads any media (like images) shared with the messages and saves them to the data/raw/photos/ directory.
8.	The scraper will automatically clean the raw messages (removing empty or corrupted entries) and save a cleaned version in data/processed/clean_telegram_data.csv.
9.	Run the Amharic text preprocessing script to normalize and tokenize the messages using the etnltk toolkit:
   python src/amharic_text_preprocessing.py
This script will remove emojis, normalize labialized letters, expand short forms, standardize punctuation, normalize characters, and tokenize the text.
10.	The final preprocessed Amharic messages will be saved in data/processed final_amharic_preprocessed.csv.

# Task-2

1. This task involves labeling a subset of Amharic messages from a dataset for Named Entity Recognition (NER) using the CoNLL format.

2. Each message contains product descriptions, prices, and locations in Amharic, collected from Telegram channels.

3. The CoNLL format requires each token to appear on a separate line, followed by a tab (`\t`) and its corresponding label.

4. The entity labels used are:

  * `B-PRODUCT`: Beginning of a product name
  * `I-PRODUCT`: Inside a product name
  * `B-LOC`: Beginning of a location name
  * `I-LOC`: Inside a location name
  * `B-PRICE`: Beginning of a price mention
  * `I-PRICE`: Inside a price mention
  * `I-PHONE`: Phone number entity (e.g., 10-digit mobile number)
  * `O`: Outside of any entity

5. A rule-based Python script reads and processes the first 50 messages from the dataset.

6. The labeled data is saved in CoNLL format to `data/processed/labeled_data.conll`.

# Task-3:NER Model Fine-Tuning 

1. Fine-tuned `xlm-roberta-base` on Amharic CoNLL-formatted e-commerce data using Hugging Face Transformers.
2. Tokenized inputs and aligned labels for NER tasks.
3. Trained using `Trainer` for 4 epochs (batch size: 8, learning rate: 5e-5).
4. Achieved **F1-score: 1.0** on the test set (perfect classification for `PRODUCT` entity).
5. Saved the model and tokenizer to `data/processed/ner_amharic_finetuned`.
6. Verified with sample inference texts like “እቃ ዋጋ 500 ብር ቦሌ”.

# task-4:Model Comparison & Selection
1. Data Preparation
2. Tokenization & Label Alignment
3. Model Fine-Tuning
4. Fine-tuned 3 multilingual transformer models:

  * 🔹 `xlm-roberta`: `FacebookAI/xlm-roberta-base`
  * 🔹 `distilbert`: `Davlan/distilbert-base-multilingual-cased-ner-hrl`
  * 🔹 `mbert`: `bert-base-multilingual-cased`
5. Used Trainer API with:

  * `epochs = 3`, `batch_size = 8`, `max_len = 128`, `lr = 5e-5`

6. Evaluation & Metric Logging

  * Used `seqeval` to compute F1-score
  * Collected metrics:

      * Evaluation loss
      * Evaluation F1-score
      * Total runtime
7. Results saved to `model_comparison.csv`

# task-5: Model Interpretability

1. Loaded fine-tuned xlm-roberta NER pipeline
2. Wrapped pipeline for compatibility with LIME and SHAP
3. Visualized token influence using LimeTextExplainer
4. Interpreted predictions using SHAP (with fallbacks for system constraints)
5. Analyzed model errors and uncertain predictions using:
6. Confidence threshold filtering (score < 0.7)
7. Generated CSV report with:
    - Text input
    - Extracted word
    - Predicted label
    - Confidence score

# Taks 6: FinTech Vendor Scorecard for Micro-Lending

1. Posting Frequency: Calculates average posts per week for each vendor to measure activity and consistency.

2. Market Reach & Engagement: Computes average views per post and identifies the top-performing post by view count.

3. Business Profile: Extracts average product price from NER-identified product entities.

4. Lending Score: Combines key metrics into a weighted score indicating vendor lending potential.

5. Output: Generates a summary CSV scorecard comparing vendors on key metrics.
