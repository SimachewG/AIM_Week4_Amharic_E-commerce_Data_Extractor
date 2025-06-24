# improved SHAP and LIME for NER explanation
import os
import shap
import lime
import torch
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    logging as hf_logging
)
from pathlib import Path
from IPython.display import display, HTML

# ------------------------------
# Configuration & Setup
# ------------------------------
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path("data/processed/ner_amharic_finetuned").resolve().as_posix()

LABELS = ['O', 'B-PRODUCT', 'I-PRODUCT']
LABEL_MAPPING = {
    "LABEL_0": "O",
    "LABEL_1": "B-PRODUCT",
    "LABEL_2": "I-PRODUCT"
}

# ------------------------------
# Load Model & Pipeline
# ------------------------------
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return tokenizer, model, ner_pipeline

tokenizer, model, ner_pipeline = load_pipeline()

# ------------------------------
# Display Predictions
# ------------------------------
def display_predictions(texts):
    logging.info("Model Predictions:")
    for text in texts:
        print(f"\n {text}")
        print(ner_pipeline(text))

# ------------------------------
# LIME (Simple HTML Highlighting for NER)
# ------------------------------
def highlight_entities(text, entities):
    for ent in sorted(entities, key=lambda e: -e['start']):
        color = "#FFD700" if ent["entity_group"] == "B-PRODUCT" else "#87CEEB"
        text = (
            text[:ent["start"]]
            + f'<span style="background-color:{color};padding:2px;border-radius:4px;">'
            + text[ent["start"]:ent["end"]]
            + "</span>"
            + text[ent["end"]:]
        )
    return text

def run_lime_interpretation(texts):
    logging.info("Rendering LIME-style explanation (highlighted entities)...")
    for text in texts[:3]:
        ents = ner_pipeline(text)
        html = highlight_entities(text, ents)
        display(HTML(f"<div style='margin:10px 0;'>{html}</div>"))

# ------------------------------
# SHAP Interpretation (Token-Level Visualization)
# ------------------------------
def run_shap_interpretation(texts):
    try:
        logging.info(" Running SHAP interpretation (may take time)...")
        class NERPipelineWrapper:
            def __init__(self, tokenizer, model):
                self.tokenizer = tokenizer
                self.model = model.eval()

            def __call__(self, texts):
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs).logits
                return outputs

        wrapped_model = NERPipelineWrapper(tokenizer, model)
        explainer = shap.Explainer(wrapped_model, tokenizer)
        shap_values = explainer(texts[:2])
        shap.plots.text(shap_values[0])
    except Exception as e:
        logging.warning(f" SHAP skipped: {e}")

# ------------------------------
# Error Analysis
# ------------------------------
def perform_error_analysis(test_cases):
    logging.info("🔎 Error Analysis: Missed or Ambiguous Entities")
    for case in test_cases:
        print(f"\n🧪 Input: {case}")
        prediction = ner_pipeline(case)
        print("Prediction:", prediction)
        if len(prediction) == 0 or all(ent['score'] < 0.7 for ent in prediction):
            print(" Possibly ambiguous or missed entity")

# ------------------------------
# Save CSV Report
# ------------------------------
def save_csv_report(texts, filename="ner_interpretability_report.csv"):
    report_data = []
    for text in texts:
        ents = ner_pipeline(text)
        for ent in ents:
            report_data.append({
                "text": text,
                "word": ent['word'],
                "label": ent['entity_group'],
                "score": ent['score']
            })
    pd.DataFrame(report_data).to_csv(filename, index=False)
    logging.info(f"✅ Report saved to {filename}")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    sample_texts = [
        "ስድስት ፍሬ ብር ብር ለማዘዝ ውስን ፍሬ ነው የቀረው ጥራት ዋስትና ቅናሽ",
        "ኦርጂናል አንሶላ በፍፁም እርጥበት የማያስገባ የራሱ የትራስ ጨርቅ ያለው",
        "አዲስ አበባ ሜክሲኮ ከ ኬኬር ህንፃ ሜ ወረድ ብሎ አይመን ህንፃ ግራውንድ ፍሎር",
        "የሎሽን እና የቅባት መቀነሻ በቀላሉ ቦርሳ ዉስጥ ቀንሰዉ የሚያስቀምጡት ለማፅዳት",
        "መዚድ ፕላዛ የመጀመሪያ ደረጃ እንደወጡ የቢሮ ቁጥር",
        "የሚወዱትን እኛ ሁሉ እኛ ጋር ያገኙታል ይጎብኙን በምናቀርባቸዉ ይደሰታሉ"
    ]

    test_cases = [
        "ለማዘዝ ጥራት ዋስትና ቅናሽ አድራሻ ቁጥር መገናኛ ዘፍመሽ ግራንድ ሞል",
        "ቅናሽ በሚሉት እቃዎች በሙሉ እና ተመልከቱ",
        "ሜክሲኮ ከ ኬኬር ህንፃ ሜ ወረድ ብሎ አይመን ህንፃ",
        "በፈጣን ሞተረኞቻችን እንልክልዏታለን ለክፍለ ሀገር ደንበኞቻችን በመነሀሪያ በኩል",
        "ኮሜርስ ጀርባ መዚድ ፕላዛ የመጀመሪያ ደረጃ እንደወጡ",
        "ከጦር ሃይሎች ወደ ቤተል በሚወስደዉ ሜክሲኮ ወይንም ቦሌ የሚወዱትን መርጠዉ ይዉሰዱ"
    ]

    display_predictions(sample_texts)
    run_lime_interpretation(sample_texts)
    run_shap_interpretation(sample_texts)
    perform_error_analysis(test_cases)
    save_csv_report(sample_texts)








