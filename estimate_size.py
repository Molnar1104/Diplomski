import sys
from datasets import load_dataset

# --- CONFIGURATION ---
TARGET_COLUMNS = [
    "Date", "Article_title", "Stock_symbol", 
    "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"
]
START_YEAR = 2017
TOTAL_DATASET_ROWS = 15_700_000  # From the dataset description
SAMPLE_SIZE = 5000  # How many rows to test to get the average

print(f"Probing the first {SAMPLE_SIZE} rows to estimate final size...")

# Load stream
dataset = load_dataset(
    "Zihan1004/FNSPID", 
    data_files="Stock_news/nasdaq_exteral_data.csv", 
    split="train", 
    streaming=True
)

valid_rows_count = 0
accumulated_bytes = 0
processed_count = 0

for row in dataset:
    processed_count += 1
    
    # 1. Check Date Filter
    # We assume Date format is standard (YYYY-MM-DD). 
    # If it starts with the year, string comparison works nicely.
    date_val = str(row.get("Date", ""))
    
    # Simple Year Extraction
    try:
        year = int(date_val[:4]) # Grabs first 4 chars as year
        if year < START_YEAR:
            continue # Skip this row, it's too old
    except:
        continue # Skip if date is broken
        
    # 2. If we are here, the row is valid (2017+). Measure it.
    valid_rows_count += 1
    
    # Create the 'clean' row string to measure its size
    clean_row_content = [str(row.get(col, "")) for col in TARGET_COLUMNS]
    # Join with commas to simulate CSV size
    row_string = ",".join(clean_row_content)
    
    # Count bytes (UTF-8 length)
    accumulated_bytes += len(row_string.encode('utf-8'))

    if processed_count >= SAMPLE_SIZE:
        break

# --- CALCULATE ESTIMATES ---

# What % of the data is from 2017-2023?
retention_rate = valid_rows_count / SAMPLE_SIZE

# Average size of a SINGLE row (in bytes)
if valid_rows_count > 0:
    avg_row_size = accumulated_bytes / valid_rows_count
else:
    avg_row_size = 0

# Extrapolate to the full 15.7 Million rows
estimated_total_rows = TOTAL_DATASET_ROWS * retention_rate
estimated_total_size_bytes = estimated_total_rows * avg_row_size
estimated_size_gb = estimated_total_size_bytes / (1024**3)

print("-" * 30)
print(f"PROBE RESULTS (Based on first {SAMPLE_SIZE} rows)")
print("-" * 30)
print(f"Rows kept (2017-2023): {valid_rows_count} out of {SAMPLE_SIZE} ({retention_rate:.1%} match rate)")
print(f"Estimated TOTAL Rows:    ~{int(estimated_total_rows):,}")
print(f"Estimated FINAL Size:    ~{estimated_size_gb:.2f} GB")
print("-" * 30)

if estimated_total_rows > 1_048_576:
    print("⚠️ WARNING: This will EXCEED the Excel row limit (1.04 Million).")
    print("   You cannot save this to a single .xlsx file.")