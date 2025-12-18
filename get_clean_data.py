import pandas as pd
from datasets import load_dataset

# 1. Define the columns you actually want
# (Make sure these match the dataset's internal names exactly)
TARGET_COLUMNS = [
    "Date", 
    "Article_title", 
    "Stock_symbol", 
    "Lsa_summary", 
    "Luhn_summary", 
    "Textrank_summary", 
    "Lexrank_summary"
]

# 2. Connect to the remote dataset without downloading the 21GB file
print("Connecting to Hugging Face...")
dataset = load_dataset(
    "Zihan1004/FNSPID", 
    data_files="Stock_news/nasdaq_exteral_data.csv", 
    split="train", 
    streaming=True
)

filtered_rows = []
row_limit = 1000  # Set this to None if you want to download EVERYTHING
counter = 0

print(f"Starting stream (Limit: {row_limit} rows)...")

# 3. Loop through the data stream
for row in dataset:
    # Create a new dictionary with ONLY your target columns
    # We use .get() to avoid crashing if a column is missing
    clean_row = {col: row.get(col, None) for col in TARGET_COLUMNS}
    
    filtered_rows.append(clean_row)
    
    counter += 1
    if counter % 100 == 0:
        print(f"Processed {counter} rows...", end="\r")
        
    # Stop when we hit the limit
    if row_limit and counter >= row_limit:
        break

print(f"\nDone! Processed {len(filtered_rows)} rows.")

# 4. Convert to a Pandas DataFrame and save as Excel
print("Saving to Excel...")
df = pd.DataFrame(filtered_rows)

# This saves it as a real Excel file, which fixes the "messy" formatting issues
df.to_excel("clean_thesis_data_sample.xlsx", index=False)

print("Success! Open 'clean_thesis_data_sample.xlsx' to check your data.")