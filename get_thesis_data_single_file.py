import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import os
import sys

# --- CONFIGURATION ---
TARGET_COLUMNS = [
    "Date", 
    "Article_title", 
    "Stock_symbol", 
    "Lsa_summary"
]

START_YEAR = 2019   # Updated to 2019
OUTPUT_FILE = "thesis_data_2019_v2.parquet" # Changed name to avoid conflict with locked file
BATCH_SIZE = 100_000 # Updated to 100k for faster updates

# ----------------

# Try to remove the file. If locked, warn the user and exit.
if os.path.exists(OUTPUT_FILE):
    try:
        os.remove(OUTPUT_FILE)
    except PermissionError:
        print(f"❌ ERROR: '{OUTPUT_FILE}' is locked by another process.")
        print("   Please close any other Python windows or change the OUTPUT_FILE name.")
        sys.exit(1)

print("Connecting to Hugging Face Stream...")
dataset = load_dataset(
    "Zihan1004/FNSPID", 
    data_files="Stock_news/nasdaq_exteral_data.csv", 
    split="train", 
    streaming=True
)

print(f"Starting processing (Year >= {START_YEAR})...")
print(f"Targeting single file: {OUTPUT_FILE}")

buffer = []
writer = None
total_rows = 0
start_schema = None 

# --- THE SAFETY BLOCK ---
try:
    for i, row in enumerate(dataset):
        # 1. Filter by Date
        date_val = str(row.get("Date", ""))
        try:
            if int(date_val[:4]) < START_YEAR:
                continue
        except:
            continue

        # 2. Extract columns
        clean_row = {col: row.get(col, None) for col in TARGET_COLUMNS}
        buffer.append(clean_row)

        # 3. Write to file when buffer is full
        if len(buffer) >= BATCH_SIZE:
            df_batch = pd.DataFrame(buffer)
            
            if writer is None:
                # First batch: Define the rules (Schema)
                table = pa.Table.from_pandas(df_batch)
                start_schema = table.schema 
                writer = pq.ParquetWriter(OUTPUT_FILE, start_schema, compression='snappy')
            else:
                # Later batches: FORCE the saved rules (Fixes the "Empty Batch" crash)
                table = pa.Table.from_pandas(df_batch, schema=start_schema)
            
            writer.write_table(table)
            
            total_rows += len(buffer)
            print(f"✅ Wrote batch. Total rows saved: {total_rows:,}")
            
            buffer = []

    # 4. Write remaining rows (if any)
    if buffer:
        df_batch = pd.DataFrame(buffer)
        if writer is None:
            table = pa.Table.from_pandas(df_batch)
            writer = pq.ParquetWriter(OUTPUT_FILE, table.schema, compression='snappy')
        else:
            table = pa.Table.from_pandas(df_batch, schema=start_schema)
        
        writer.write_table(table)
        total_rows += len(buffer)

except KeyboardInterrupt:
    print("\n⚠️  Script stopped by user (Ctrl+C).")
    print("   Closing file safely so it doesn't get corrupted...")

finally:
    # This block ALWAYS runs, even if you crash or Ctrl+C
    if writer:
        writer.close()
        print(f"🔒 File closed safely.")

print("-" * 30)
print(f"DONE! Saved {total_rows:,} rows to '{OUTPUT_FILE}'")