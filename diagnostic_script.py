from datasets import load_dataset
print("Peeking at dataset columns...")
dataset = load_dataset(
    "Zihan1004/FNSPID", 
    data_files="Stock_news/nasdaq_exteral_data.csv", 
    split="train", 
    streaming=True
)

# 1. Print ALL column names found in the file
first_row = next(iter(dataset))
print("\n--- AVAILABLE COLUMNS ---")
print(list(first_row.keys()))

# 2. Check the first non-empty sentiment row
# We suspect the column might be named 'Sentiment_score', 'sentiment', or 'Label'
print("\n--- SEARCHING FOR SENTIMENT DATA ---")
print("Scanning first 2,000 rows for non-empty sentiment data...")

sentiment_col = None
# Guessing likely names based on the paper
potential_names = ["Sentiment_score", "Sentiment", "sentiment", "Score", "Label"]

# Find the actual column name
for key in first_row.keys():
    if any(name.lower() in key.lower() for name in potential_names):
        sentiment_col = key
        break

if sentiment_col:
    print(f"Found sentiment column: '{sentiment_col}'")
    
    count = 0
    for i, row in enumerate(dataset):
        val = row.get(sentiment_col)
        if val is not None and str(val).strip() != "":
            print(f"Row {i}: Stock={row.get('Stock_symbol')} | {sentiment_col}={val}")
            count += 1
            if count >= 3: break
else:
    print("Could not auto-detect a sentiment column. Please check the 'AVAILABLE COLUMNS' list above.")