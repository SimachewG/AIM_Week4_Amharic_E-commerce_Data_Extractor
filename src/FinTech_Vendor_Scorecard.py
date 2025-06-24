import pandas as pd
import os
from datetime import datetime
import ast

# ------------------------------
# Helper Functions
# ------------------------------

def load_vendor_data(file_path):
    """Load vendor data from CSV, convert timestamp to datetime."""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    return df

def calculate_posting_frequency(df):
    """Calculate average number of posts per week."""
    if df.empty:
        return 0
    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
    weeks = max(date_range / 7, 1)  # Avoid division by zero if less than a week
    return round(len(df) / weeks, 2)

def safe_parse_entities(entity_str):
    """Safely parse string representation of entities to Python list."""
    try:
        entities = ast.literal_eval(entity_str)
        if isinstance(entities, list):
            return entities
    except Exception:
        pass
    return []

def get_avg_price(df):
    """Extract prices from entities and calculate average price."""
    prices = []
    for _, row in df.iterrows():
        entities = safe_parse_entities(row.get("entities", "[]"))
        for ent in entities:
            if isinstance(ent, dict) and ent.get('label') == 'B-PRODUCT':
                price_text = ent.get('price') or ent.get('word')
                if price_text:
                    try:
                        # Remove commas, 'ETB', spaces and convert to float
                        price_val = float(str(price_text).replace(',', '').replace('ETB', '').strip())
                        prices.append(price_val)
                    except ValueError:
                        continue
    return round(sum(prices) / len(prices), 2) if prices else 0

def get_top_post_info(df):
    """Get text, views, top product, and price from the post with highest views."""
    if df.empty:
        return None, 0, "", 0
    top_post = df.loc[df['views'].idxmax()]
    text = top_post.get('text', '')
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
    """Calculate lending score based on average views and posting frequency."""
    return round((avg_views * 0.5) + (post_freq * 0.5), 2)

# ------------------------------
# Main Scoring Function
# ------------------------------

def analyze_vendor(file_path, vendor_name):
    df = load_vendor_data(file_path)
    if df.empty:
        return None

    posting_freq = calculate_posting_frequency(df)
    avg_views = round(df['views'].mean(), 2) if not df.empty else 0
    avg_price = get_avg_price(df)
    _, top_views, top_product, top_price = get_top_post_info(df)
    score = calculate_lending_score(avg_views, posting_freq)

    return {
        "Vendor": vendor_name,
        "Avg Views/Post": avg_views,
        "Posts/Week": posting_freq,
        "Avg Price (ETB)": avg_price,
        "Top Product": top_product,
        "Top Price": top_price,
        "Lending Score": score
    }

# ------------------------------
# Batch Processing
# ------------------------------

def run_vendor_scorecard(data_dir='data', output_file='outputs/vendor_scorecard.csv'):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    results = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            vendor_name = file.replace('_cleaned.csv', '')
            file_path = os.path.join(data_dir, file)
            vendor_result = analyze_vendor(file_path, vendor_name)
            if vendor_result:
                results.append(vendor_result)

    result_df = pd.DataFrame(results)
    result_df.sort_values(by='Lending Score', ascending=False, inplace=True)
    result_df.to_csv(output_file, index=False)

    print(f"✅ Vendor Scorecard saved to {output_file}")
    print(result_df[['Vendor', 'Avg Views/Post', 'Posts/Week', 'Avg Price (ETB)', 'Lending Score']])

# ------------------------------
# Run script if executed as main
# ------------------------------

if __name__ == "__main__":
    run_vendor_scorecard()
