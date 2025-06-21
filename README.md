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

