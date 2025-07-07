import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_portfolio_performance_with_entry_split(performance_df, entry_date):
    """
    Plot portfolio value over time, using a different color before and after entry_date.

    Args:
        performance_df (pd.DataFrame): DataFrame with 'date', 'portfolio_value', 'return_pct'.
        entry_date (datetime.date or str): Entry date to split the plot.
    """
    # Ensure datetime types
    performance_df['date'] = pd.to_datetime(performance_df['date'])
    entry_date = pd.to_datetime(entry_date)

    # Split data before and after entry
    before_entry = performance_df[performance_df['date'] < entry_date]
    after_entry = performance_df[performance_df['date'] >= entry_date]

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Plot pre-entry in gray
    plt.plot(before_entry['date'], before_entry['cumulative_profit'], color='gray', linestyle='--', label='Before Entry')

    # Plot post-entry in blue
    plt.plot(after_entry['date'], after_entry['cumulative_profit'], color='tab:blue', linewidth=2, label='After Entry')

    plt.axvline(entry_date, color='black', linestyle=':', linewidth=1)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()
