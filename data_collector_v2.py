import pandas as pd
import numpy as np 
import yfinance as yf
from datetime import datetime
import time 
import pandas_ta as ta

# --- Configuration ---
INDEX_TICKER = 'SPY' 
VIX_TICKER = '^VIX' 
GOLD_TICKER = 'GC=F'      # Gold Futures
BOND_TICKER = '^TNX'      # 10-Year Treasury Yield
CURRENCY_TICKER = 'JPY=X' # USD/JPY Exchange Rate

START_DATE = '2019-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

def safe_download(ticker, start, end, retries=3):
    """
    Attempts to download data using yf.Ticker().history() with retries.
    """
    t = yf.Ticker(ticker)
    for i in range(retries):
        try:
            print(f"   Fetching {ticker}...")
            data = t.history(start=start, end=end, interval='1d', auto_adjust=True)
            if not data.empty and 'Close' in data.columns:
                return data[['Close']].rename(columns={'Close': ticker})
            time.sleep(1)
        except Exception as e:
            print(f"   Error fetching {ticker}: {e}. Retrying...")
            time.sleep(2)
    print(f"   FAILED to fetch {ticker}.")
    return pd.DataFrame() 

def combine_and_engineer_features():
    print("--- 1. Fetching Market Data ---")
    market_df = safe_download(INDEX_TICKER, START_DATE, END_DATE)
    market_df = market_df.rename(columns={INDEX_TICKER: 'Close'})
    
    # Fetch Macro Variables
    vix_df = safe_download(VIX_TICKER, START_DATE, END_DATE)
    gold_df = safe_download(GOLD_TICKER, START_DATE, END_DATE)
    bond_df = safe_download(BOND_TICKER, START_DATE, END_DATE)
    yen_df = safe_download(CURRENCY_TICKER, START_DATE, END_DATE)
    
    # Get Volume separately (usually only available for stocks/ETFs)
    spy_full = yf.Ticker(INDEX_TICKER).history(start=START_DATE, end=END_DATE, interval='1d')
    market_df['Volume'] = spy_full['Volume']

    # --- 2. Load Real Sentiment Data ---
    print("\n--- 2. Loading Sentiment Data ---")
    try:
        sentiment_df = pd.read_csv('daily_sentiment.csv', index_col='Date', parse_dates=True)
        if sentiment_df.index.tz is not None:
            sentiment_df.index = sentiment_df.index.tz_localize(None)
    except FileNotFoundError:
        print("WARNING: 'daily_sentiment.csv' not found. Using dummy zeros.")
        sentiment_df = pd.DataFrame(index=market_df.index)
        sentiment_df['Sentiment_Score'] = 0
        sentiment_df['News_Volume'] = 0

    # --- 3. Merge Everything ---
    print("\n--- 3. Merging Data ---")
    # Ensure all indices are timezone-naive
    dfs = [market_df, vix_df, gold_df, bond_df, yen_df, sentiment_df]
    for df in dfs:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
    combined_df = pd.concat(dfs, axis=1).sort_index()
    
    # Forward Fill missing macro data (e.g., holidays where currencies trade but stocks don't)
    combined_df = combined_df.ffill().dropna()

    # --- 4. Feature Engineering ---
    print("\n--- 4. Engineering Features ---")
    
    # A. Target: Next Day Direction
    combined_df['Target_Next_Day'] = (combined_df['Close'].shift(-1) > combined_df['Close']).astype(int)
    
    # B. Technicals
    combined_df['Daily_Return'] = combined_df['Close'].pct_change()
    combined_df['SMA_20'] = ta.sma(combined_df['Close'], length=20)
    combined_df['RSI_14'] = ta.rsi(combined_df['Close'], length=14)
    macd = ta.macd(combined_df['Close'])
    combined_df = pd.concat([combined_df, macd], axis=1)
    
    # C. Macro Correlations (New Human Variables)
    # Does Gold moving UP mean Stocks go DOWN?
    combined_df['Gold_Return'] = combined_df[GOLD_TICKER].pct_change()
    combined_df['Bond_Return'] = combined_df[BOND_TICKER].pct_change()
    combined_df['Yen_Return'] = combined_df[CURRENCY_TICKER].pct_change()

    # D. Sentiment Features (The "Momentum" Discovery)
    if 'Sentiment_Score' in combined_df.columns:
        combined_df['Sentiment_SMA_3'] = combined_df['Sentiment_Score'].rolling(window=3).mean()
        combined_df['Sentiment_Momentum'] = combined_df['Sentiment_Score'] - combined_df['Sentiment_Score'].shift(1)
        # Interaction: Fear (VIX) * Bad News (Sentiment)
        combined_df['Fear_Sentiment_Index'] = combined_df[VIX_TICKER] * (combined_df['Sentiment_Score'] * -1)

    final_df = combined_df.dropna()
    
    print("\n--- Final Data Snapshot ---")
    print(final_df[[GOLD_TICKER, BOND_TICKER, CURRENCY_TICKER, 'Sentiment_Momentum', 'Close']].tail(3))
    print(f"\nTotal Rows: {len(final_df)}")
    
    return final_df

if __name__ == '__main__':
    df = combine_and_engineer_features()
    df.to_csv('final_dataset.csv')
    print("Saved to final_dataset.csv")