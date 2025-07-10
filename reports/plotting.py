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

def plot_rolling_sharpe_ratio(performance_df, window=1):
    """
    Plot rolling Sharpe ratio over time.

    Args:
        performance_df (pd.DataFrame): Must contain 'date' and 'daily_return'.
        window (int): Rolling window size in days.
        risk_free_rate_annual (float): Annualized risk-free rate (e.g., 0.05 for 5%).
    """
    # Ensure proper types
    performance_df['date'] = pd.to_datetime(performance_df['date'])
    performance_df = performance_df.sort_values('date')
    

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(performance_df['date'], performance_df['sharpe_ratio'], label='Sharpe', color='tab:green')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"{window}-Day Rolling Sharpe Ratio Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.tight_layout()
    plt.show()
