import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Customizable Text Labels ---
# You can easily translate these variables into Croatian
SP500_TITLE = 'S&P 500 (SPY) Closing Price - Last 10 Years'
SP500_Y_LABEL = 'Closing Price (USD)'
DATE_LABEL = 'Date'
SP500_LEGEND = 'S&P 500 Closing Price'

VIX_TITLE = 'S&P 500 vs. VIX Index - Last 10 Years'
VIX_Y_LABEL = 'VIX Index'
VIX_LEGEND = 'VIX Index'

def plot_sp500_closing_price(sp500_data):
    """
    Plots the S&P 500 (SPY) closing price.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(sp500_data['Close'], label=SP500_LEGEND)
    plt.title(SP500_TITLE)
    plt.xlabel(DATE_LABEL)
    plt.ylabel(SP500_Y_LABEL)
    plt.legend()
    plt.grid(True)

    output_filename = 'sp500_closing_price.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

def plot_sp500_vix_comparison(sp500_data, vix_data):
    """
    Plots the S&P 500 vs. VIX on a dual-axis chart.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot S&P 500
    ax1.set_xlabel(DATE_LABEL)
    ax1.set_ylabel(SP500_Y_LABEL, color='tab:blue')
    ax1.plot(sp500_data.index, sp500_data['Close'], color='tab:blue', label=SP500_LEGEND)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for the VIX
    ax2 = ax1.twinx()
    ax2.set_ylabel(VIX_Y_LABEL, color='tab:red')
    ax2.plot(vix_data.index, vix_data['Close'], color='tab:red', label=VIX_LEGEND)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(VIX_TITLE)
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))

    output_filename = 'sp500_vix_comparison.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

def main():
    """
    Main function to fetch data and generate plots.
    """
    # --- Configuration ---
    tickers = ['SPY', '^VIX']
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10*365)

    # --- Data Fetching ---
    try:
        print(f"Fetching S&P 500 and VIX data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        data = yf.download(tickers, start=start_date, end=end_date)

        if data.empty:
            print("No data fetched. Please check the tickers and date range.")
            return

        sp500_data = data['Close']['SPY'].to_frame().rename(columns={'SPY': 'Close'})
        vix_data = data['Close']['^VIX'].to_frame().rename(columns={'^VIX': 'Close'})

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return

    # --- Generate Plots ---
    plot_sp500_closing_price(sp500_data)
    plot_sp500_vix_comparison(sp500_data, vix_data)

if __name__ == '__main__':
    main()
