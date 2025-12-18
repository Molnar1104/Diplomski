import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# --- CONFIGURATION ---
DATA_FILE = 'final_dataset.csv'

def plot_thesis_chart():
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("Run data_collector.py first!")
        return

    # 2. Filter for a specific interesting year (e.g., 2022 or 2020 crash)
    # or just plot the last 2 years to keep it readable
    subset = df.loc['2022-01-01':'2023-12-31']

    # 3. Create the Dual-Axis Plot
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Stock Price (Background)
    color = 'tab:blue'
    ax1.set_xlabel('Datum')
    ax1.set_ylabel('S&P 500 Cijena ($)', color=color)
    ax1.plot(subset.index, subset['Close'], color=color, label='S&P 500 Cijena', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Sentiment Momentum (The Signal)
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:purple'
    ax2.set_ylabel('Promjena Sentimenta', color=color)
    
    ax2.plot(subset.index, subset['Sentiment_Momentum'], color=color, label='Sentiment Momentum', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add a zero line for momentum (Neutral)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Titles and Legends
    plt.title('Vodi li promjena u sentimentu vijesti prema kretanju tržišta? (2022-2023)')
    fig.tight_layout()
    
    # Save
    plt.savefig('thesis_momentum_chart2.png')
    print(" Saved chart to 'thesis_momentum_chart.png'")

if __name__ == "__main__":
    plot_thesis_chart()