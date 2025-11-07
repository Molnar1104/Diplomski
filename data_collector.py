import pandas as pd
import numpy as np 
import yfinance as yf
import requests # Used for custom session headers
from datetime import datetime
import time 

# --- Configuration ---
INDEX_TICKER = 'SPY' 
VIX_TICKER = '^VIX' 
START_DATE = '2019-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
API_KEY = "YOUR_SENTIMENT_API_KEY" 

def safe_download(ticker, start, end, retries=5):
    """
    Attempts to download data using yf.Ticker().history() with a custom User-Agent 
    to bypass common connection issues.
    """
    # Use a custom session with a User-Agent to improve reliability
    session = requests.Session()
    # Adding a robust User-Agent header
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    t = yf.Ticker(ticker, session=session)

    for i in range(retries):
        try:
            print(f"Attempting to fetch data for {ticker} (Attempt {i+1})...")
            # Use the history method for the period
            data = t.history(start=start, end=end, interval='1d', auto_adjust=True)
            
            if not data.empty and 'Close' in data.columns:
                # Return the full DataFrame for the wrappers to process
                return data
            
            time.sleep(2) 
        
        except Exception as e:
            print(f"Error fetching {ticker}: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    print(f"FAILED to fetch {ticker} after {retries} attempts.")
    return pd.DataFrame() 

def get_market_data(ticker, start, end):
    """Fetches historical price data (Market Data)."""
    data = safe_download(ticker, start, end)
    if data.empty:
        raise ValueError(f"Failed to load Market Data for {ticker}. Check the ticker/date range.")
    # Return Close and Volume columns
    return data[['Close', 'Volume']]

def get_vix_data(ticker, start, end):
    """Fetches VIX (Fundamental/Human Element Data)."""
    vix_data = safe_download(ticker, start, end)
    if vix_data.empty:
        print(f"WARNING: VIX data failed to load. The VIX feature will be excluded.")
        return pd.DataFrame() 
    # Only need the daily closing price of the VIX
    return vix_data[['Close']].rename(columns={'Close': 'VIX_Close'})

def get_sentiment_data(ticker, start, end, api_key):
    """
    Conceptual function to fetch Sentiment Data. (Still using DUMMY data)
    """
    print(f"Fetching Sentiment Data (requires a valid API setup)...")
    
    dates = pd.date_range(start, end, freq='B') # 'B' for business days
    sentiment_df = pd.DataFrame(index=dates, columns=['Sentiment_Score', 'News_Volume'])

    # Fill with illustrative dummy data for script testing
    sentiment_df.loc[dates, 'Sentiment_Score'] = np.random.uniform(-0.5, 0.5, len(dates))
    sentiment_df.loc[dates, 'News_Volume'] = np.random.randint(50, 500, len(dates))
    
    print("WARNING: Using DUMMY sentiment data. REPLACE THIS FUNCTION with a real API call.")
    # Drop rows that weren't business days (e.g., public holidays not in yfinance data)
    return sentiment_df.dropna()


def combine_and_engineer_features():
    """Combines all data and creates target/technical features."""
    
    # 1. Fetch Data
    market_df = get_market_data(INDEX_TICKER, START_DATE, END_DATE)
    vix_df = get_vix_data(VIX_TICKER, START_DATE, END_DATE)
    sentiment_df = get_sentiment_data(INDEX_TICKER, START_DATE, END_DATE, API_KEY)
    
    # 2. Merge DataFrames on the Index (Date)
    combined_df = market_df.copy()
    
    if not vix_df.empty:
        combined_df = combined_df.join(vix_df, how='inner')
        
    combined_df = combined_df.join(sentiment_df, how='inner')

    # --- 3. Feature Engineering ---
    
    # A. Target Variable (Prediction Target)
    combined_df['Target_Next_Day'] = (combined_df['Close'].shift(-1) > combined_df['Close']).astype(int)
    
    # B. Technical/Market Features
    combined_df['Daily_Return'] = combined_df['Close'].pct_change()
    combined_df['SMA_50'] = combined_df['Close'].rolling(window=50).mean()

    # C. Fundamental/Human Element Features (Engineered Sentiment)
    if 'Sentiment_Score' in combined_df.columns and 'News_Volume' in combined_df.columns:
        # Advanced Feature (Separating TONE and INTENSITY):
        combined_df['Weighted_Sentiment'] = combined_df['Sentiment_Score'] * combined_df['News_Volume']

    final_df = combined_df.dropna()
    print("\n--- Final Data Snapshot (SPY, VIX, Sentiment) ---")
    print(final_df.head())
    print(f"\nTotal Data Points: {len(final_df)}")
    
    return final_df

if __name__ == '__main__':
    try:
        final_dataset = combine_and_engineer_features()
    except ValueError as e:
        print(f"\nFATAL ERROR: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")