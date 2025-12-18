import pandas as pd

# --- CONFIGURATION ---
INPUT_FILE = "thesis_data_sampled_100perDay.parquet"
OUTPUT_CSV = "thesis_data_viewable.csv"

# 1. Load the Parquet file
print(f"Reading {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)

# 2. Show a mini report in the terminal
print("-" * 40)
print(f"✅ LOADED SUCCESSFULLY")
print(f"Total Rows: {len(df)}")
print("-" * 40)

print("\n--- COLUMN INFO ---")
print(df.info())

print("\n--- FIRST 5 ROWS (Preview) ---")
print(df.head())

print("\n--- RANDOM SAMPLE OF 3 ROWS ---")
print(df.sample(3))

# 3. Convert to CSV (Easy!)
print("-" * 40)
print(f"Converting to CSV: {OUTPUT_CSV}...")
# index=False keeps it clean (no row numbers column)
# escapechar ensures text with quotes doesn't break the CSV
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print("✅ Done! You can now open 'thesis_data_viewable.csv' in Excel.")