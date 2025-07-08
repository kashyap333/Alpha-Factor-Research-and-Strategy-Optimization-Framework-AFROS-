import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_portfolio_performance_with_entry_split(performance_df, entry_date, save_path="portfolio_plot.png"):
    """
    Plot and save portfolio value and daily return with entry marker.

    Args:
        performance_df (pd.DataFrame): DataFrame with 'date', 'portfolio_value', 'daily_return', etc.
        entry_date (datetime.date or str): The date of portfolio entry.
        save_path (str): Path to save the image (e.g., 'portfolio_plot.png').
    """
    entry_date = pd.to_datetime(entry_date).date()
    performance_df['date'] = pd.to_datetime(performance_df['date'])
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(performance_df['date'], performance_df['portfolio_value'], label="Portfolio Value", color="tab:blue")
    ax1.axvline(entry_date, color='red', linestyle='--', label='Entry Date')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value ($)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(performance_df['date'], performance_df['daily_return'], label="Daily Return", color="tab:green", alpha=0.5)
    ax2.set_ylabel("Daily Return (%)", color="tab:green")
    ax2.tick_params(axis='y', labelcolor="tab:green")

    fig.suptitle("Portfolio Performance with Entry Highlight")
    fig.tight_layout()
    fig.legend(loc="upper left")

    # Save image
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    plt.close()  # Close the figure to avoid showing in notebooks or re-renders