import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def plot_sp500_closing_price():
    """
    Fetches the S&P 500 (SPY) closing price for the last 10 years and plots it.
    """
    # --- Configuration ---
    ticker = 'SPY'
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10*365)

    # --- Data Fetching ---
    try:
        print(f"Fetching S&P 500 data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        sp500_data = yf.download(ticker, start=start_date, end=end_date)

        if sp500_data.empty:
            print("No data fetched. Please check the ticker and date range.")
            return

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(sp500_data['Close'], label='S&P 500 Closing Price')
    plt.title('S&P 500 (SPY) Closing Price - Last 10 Years')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.grid(True)

    # --- Save the Plot ---
    output_filename = 'sp500_closing_price.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == '__main__':
    plot_sp500_closing_price()
