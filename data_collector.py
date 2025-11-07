import pandas as pd
import numpy as np 
import yfinance as yf
from datetime import datetime
import time 
import pandas_ta as ta

# --- Configuration ---
INDEX_TICKER = 'SPY' 
VIX_TICKER = '^VIX' 
START_DATE = '2019-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
# NOTE: A real API key and function for sentiment data will be needed later.

def safe_download(ticker, start, end, retries=5):
    """
    Attempts to download data using yf.Ticker().history() with retries.
    yfinance now handles session management, so custom session is removed.
    """
    t = yf.Ticker(ticker)

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
    return data[['Close', 'Volume']]

def get_vix_data(ticker, start, end):
    """Fetches VIX (Fundamental/Human Element Data)."""
    vix_data = safe_download(ticker, start, end)
    if vix_data.empty:
        print(f"WARNING: VIX data failed to load. The VIX feature will be excluded.")
        return pd.DataFrame() 
    return vix_data[['Close']].rename(columns={'Close': 'VIX_Close'})

def combine_and_engineer_features():
    """Combines all data and creates target/technical features."""
    
    # 1. Fetch Data
    market_df = get_market_data(INDEX_TICKER, START_DATE, END_DATE)
    vix_df = get_vix_data(VIX_TICKER, START_DATE, END_DATE)
    
    # 2. Standardize Timezone Information (CRITICAL STEP)
    # yfinance data is timezone-aware. For robust joins, make all indices tz-naive.
    if market_df.index.tz is not None:
        market_df.index = market_df.index.tz_localize(None)
    if not vix_df.empty and vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)

    # 3. Generate Dummy Sentiment Data (Aligned with the NOW tz-naive Market Data Index)
    print("WARNING: Using DUMMY sentiment data. REPLACE THIS with a real API call.")
    sentiment_df = pd.DataFrame(index=market_df.index)
    sentiment_df['Sentiment_Score'] = np.random.uniform(-0.5, 0.5, len(market_df))
    sentiment_df['News_Volume'] = np.random.randint(50, 500, len(market_df))

    # 4. Merge DataFrames
    combined_df = market_df.copy()
    
    if not vix_df.empty:
        combined_df = combined_df.join(vix_df, how='left')
        
    combined_df = combined_df.join(sentiment_df, how='left')

    # 5. Feature Engineering
    
    # A. Target Variable
    combined_df['Target_Next_Day'] = (combined_df['Close'].shift(-1) > combined_df['Close']).astype(int)
    
    # B. Technical Indicators
    print("Adding technical indicators...")
    combined_df['Daily_Return'] = combined_df['Close'].pct_change()

    combined_df['SMA_10'] = ta.sma(combined_df['Close'], length=10)
    combined_df['SMA_20'] = ta.sma(combined_df['Close'], length=20)
    combined_df['SMA_50'] = ta.sma(combined_df['Close'], length=50)
    combined_df['SMA_200'] = ta.sma(combined_df['Close'], length=200)

    combined_df['RSI_14'] = ta.rsi(combined_df['Close'], length=14)

    macd = ta.macd(combined_df['Close'], fast=12, slow=26, signal=9)
    combined_df = pd.concat([combined_df, macd], axis=1)

    # C. Sentiment Features
    if 'Sentiment_Score' in combined_df.columns and 'News_Volume' in combined_df.columns:
        combined_df['Weighted_Sentiment'] = combined_df['Sentiment_Score'] * combined_df['News_Volume']

    # 6. Finalize DataFrame
    # Forward-fill VIX data for any non-trading days where it might be missing
    if 'VIX_Close' in combined_df.columns:
        combined_df['VIX_Close'] = combined_df['VIX_Close'].ffill()

    final_df = combined_df.dropna()
    print("\n--- Final Data Snapshot (SPY, VIX, Sentiment & Technicals) ---")
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