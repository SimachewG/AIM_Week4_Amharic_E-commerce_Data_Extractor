import pandas as pd
import os
import ast
import random
from datetime import datetime

# ------------------------------
# Placeholder NER entity extraction
# Replace this with your fine-tuned model inference!
# ------------------------------
def extract_entities_from_text(text):
    """
    Dummy NER entity extractor that returns a list of dicts with label and price.
    This simulates extracting product entities with prices from the message text.
    """
    # Randomly decide if this message mentions a product
    if random.random() < 0.6:  # 60% chance
        product_name = "Product_" + str(random.randint(1, 10))
        price = round(random.uniform(100, 2000), 2)  # Random price ETB
        return [{"label": "B-PRODUCT", "word": product_name, "price": price}]
    else:
        return []

# ------------------------------
# Helper Functions
# ------------------------------

def safe_parse_entities(entity_str):
    """Safely parse string representation of entities to Python list."""
    try:
        entities = ast.literal_eval(entity_str)
        if isinstance(entities, list):
            return entities
    except Exception:
        pass
    return []

def calculate_posting_frequency(df):
    """Calculate average number of posts per week for a vendor DataFrame."""
    if df.empty:
        return 0
    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
    weeks = max(date_range / 7, 1)
    return round(len(df) / weeks, 2)

def get_avg_price(df):
    """Calculate average product price from entities in vendor posts."""
    prices = []
    for _, row in df.iterrows():
        entities = safe_parse_entities(row.get("entities", "[]"))
        for ent in entities:
            if isinstance(ent, dict) and ent.get('label') == 'B-PRODUCT':
                price_text = ent.get('price') or ent.get('word')
                if price_text:
                    try:
                        price_val = float(str(price_text).replace(',', '').replace('ETB', '').strip())
                        prices.append(price_val)
                    except ValueError:
                        continue
    return round(sum(prices) / len(prices), 2) if prices else 0

def get_top_post_info(df):
    """Get details of the post with highest views in vendor DataFrame."""
    if df.empty:
        return "", 0, "", 0
    top_post = df.loc[df['views'].idxmax()]
    text = top_post.get('Message', '')
    views = top_post.get('views', 0)
    entities = safe_parse_entities(top_post.get("entities", "[]"))
    top_product = ""
    top_price = 0
    for ent in entities:
        if isinstance(ent, dict) and ent.get('label') == 'B-PRODUCT' and not top_product:
            top_product = ent.get('word', '')
        if isinstance(ent, dict) and ent.get('label') == 'B-PRODUCT' and 'price' in ent:
            try:
                top_price = float(str(ent['price']).replace(',', '').replace('ETB', '').strip())
            except ValueError:
                continue
    return text, views, top_product, top_price

def calculate_lending_score(avg_views, post_freq):
    """Weighted lending score combining average views and posting frequency."""
    return round((avg_views * 0.5) + (post_freq * 0.5), 2)

# ------------------------------
# Main Vendor Analytics Engine
# ------------------------------

def analyze_vendors_from_preprocessed(preprocessed_file):
    # Load the cleaned preprocessed data (Task 1)
    df = pd.read_csv(preprocessed_file)

    # Fix column names for consistency
    df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    # Use 'Channel Title' as vendor identifier
    df['vendor'] = df['Channel Title']

    # If 'views' column does not exist, create dummy views for demo
    if 'views' not in df.columns:
        print("No 'views' column found — adding random dummy views for demo purposes.")
        df['views'] = [random.randint(50, 5000) for _ in range(len(df))]

    # If 'entities' column does not exist, run NER extraction placeholder
    if 'entities' not in df.columns:
        print("No 'entities' column found — extracting entities using dummy NER extractor.")
        df['entities'] = df['Message'].apply(lambda x: str(extract_entities_from_text(x)))

    results = []
    for vendor_name, vendor_df in df.groupby('vendor'):
        posting_freq = calculate_posting_frequency(vendor_df)
        avg_views = round(vendor_df['views'].mean(), 2) if not vendor_df.empty else 0
        avg_price = get_avg_price(vendor_df)
        _, top_views, top_product, top_price = get_top_post_info(vendor_df)
        score = calculate_lending_score(avg_views, posting_freq)

        results.append({
            "Vendor": vendor_name,
            "Avg Views/Post": avg_views,
            "Posts/Week": posting_freq,
            "Avg Price (ETB)": avg_price,
            "Top Product": top_product,
            "Top Price": top_price,
            "Lending Score": score
        })

    # Create dataframe for all vendors and sort by Lending Score
    result_df = pd.DataFrame(results)
    result_df.sort_values(by='Lending Score', ascending=False, inplace=True)
    return result_df

# ------------------------------
# Run batch process and save scorecard
# ------------------------------

def run_vendor_scorecard(preprocessed_file='data/processed/final_amharic_preprocessed.csv',
                         output_file='outputs/vendor_scorecard.csv'):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    vendor_scores = analyze_vendors_from_preprocessed(preprocessed_file)
    vendor_scores.to_csv(output_file, index=False)

    print(f"✅ Vendor Scorecard saved to {output_file}")
    print(vendor_scores[['Vendor', 'Avg Views/Post', 'Posts/Week', 'Avg Price (ETB)', 'Lending Score']])

# ------------------------------
# Entry point
# ------------------------------

if __name__ == "__main__":
    run_vendor_scorecard()


