import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import os

# --- CONFIGURATION ---
TARGET_COLUMNS = ["Date", "Article_title", "Stock_symbol", "Lsa_summary"]
START_YEAR = 2019
SAMPLES_PER_DAY = 100  # <--- CHANGED TO 100 (Safe margin)
OUTPUT_FILE = "thesis_data_sampled_100perDay.parquet" # <--- Updated name for clarity

# Tracking dictionary: { '2023-01-01': 5, '2023-01-02': 10, ... }
daily_counts = {}

# Clean up old file
if os.path.exists(OUTPUT_FILE):
    try:
        os.remove(OUTPUT_FILE)
    except PermissionError:
        print(f"❌ ERROR: '{OUTPUT_FILE}' is locked. Close other apps using it.")
        exit()

print("1. Connecting to Hugging Face Stream...")
dataset = load_dataset(
    "Zihan1004/FNSPID", 
    data_files="Stock_news/nasdaq_exteral_data.csv", 
    split="train", 
    streaming=True
)

print(f"2. Streaming & Sampling (Max {SAMPLES_PER_DAY} items per day)...")

buffer = []
total_saved = 0
writer = None

for i, row in enumerate(dataset):
    # Progress Indicator (shows every 10k rows scanned)
    if i % 10000 == 0:
        print(f"   Scanned {i} rows... (Saved: {total_saved})", end='\r')

    # --- A. Date Filter ---
    date_str = str(row.get("Date", ""))
    try:
        # Assuming format YYYY-MM-DD
        date_only = date_str[:10] 
        year = int(date_str[:4])
        
        if year < START_YEAR:
            continue
    except:
        continue 

    # --- B. Sampling Logic ---
    # If we already have 100 news items for this specific date, SKIP IT.
    current_count = daily_counts.get(date_only, 0)
    if current_count >= SAMPLES_PER_DAY:
        continue

    # --- C. Add to Buffer ---
    clean_row = {col: row.get(col, None) for col in TARGET_COLUMNS}
    clean_row['Date'] = date_only 
    
    buffer.append(clean_row)
    daily_counts[date_only] = current_count + 1
    
    # --- D. Write Small Chunks ---
    if len(buffer) >= 5000:
        df_batch = pd.DataFrame(buffer)
        
        if writer is None:
            table = pa.Table.from_pandas(df_batch)
            writer = pq.ParquetWriter(OUTPUT_FILE, table.schema, compression='snappy')
        else:
            table = pa.Table.from_pandas(df_batch, schema=writer.schema)
            
        writer.write_table(table)
        total_saved += len(buffer)
        buffer = []

# Write leftovers
if buffer:
    df_batch = pd.DataFrame(buffer)
    if writer is None:
        table = pa.Table.from_pandas(df_batch)
        writer = pq.ParquetWriter(OUTPUT_FILE, table.schema, compression='snappy')
    else:
        table = pa.Table.from_pandas(df_batch, schema=writer.schema)
    writer.write_table(table)
    total_saved += len(buffer)

if writer:
    writer.close()

print(f"\n\n✅ DONE! Saved {total_saved} rows to {OUTPUT_FILE}")