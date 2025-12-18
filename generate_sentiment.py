import pandas as pd
import pyarrow.parquet as pq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# --- CONFIGURATION ---
INPUT_FILE = "thesis_data_sampled_100perDay.parquet"
OUTPUT_FILE = "daily_sentiment.csv"
BATCH_SIZE = 10000

def generate_sentiment():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run get_thesis_data_single_file.py first.")
        return

    print(f"Reading {INPUT_FILE}...")
    
    # Initialize VADER analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Read Parquet file
    parquet_file = pq.ParquetFile(INPUT_FILE)
    
    sentiment_results = []
    total_rows = 0
    
    print("Processing batches...")
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Calculate sentiment for each row
        # We use 'Lsa_summary' as the text source
        for index, row in df.iterrows():
            text = str(row['Lsa_summary']) if row['Lsa_summary'] is not None else ""
            
            if not text or text.lower() == 'nan':
                score = 0.0
            else:
                vs = analyzer.polarity_scores(text)
                score = vs['compound']
            
            sentiment_results.append({
                'Date': row['Date'],
                'Sentiment_Score': score
            })
        
        total_rows += len(df)
        print(f"Processed {total_rows} rows...", end='\r')

    print(f"\nTotal rows processed: {total_rows}")
    
    # Create DataFrame from results
    sentiment_df = pd.DataFrame(sentiment_results)
    
    # Drop rows with invalid dates
    sentiment_df = sentiment_df.dropna(subset=['Date'])
    
    # Aggregate by Date
    print("Aggregating by Date...")
    daily_sentiment = sentiment_df.groupby(sentiment_df['Date'].dt.date).agg({
        'Sentiment_Score': 'mean',
        'Date': 'count' # Count number of articles per day
    }).rename(columns={'Date': 'News_Volume'})
    
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    
    # Save to CSV
    daily_sentiment.to_csv(OUTPUT_FILE)
    print(f"Saved daily sentiment data to {OUTPUT_FILE}")
    print(daily_sentiment.head())

if __name__ == "__main__":
    generate_sentiment()
