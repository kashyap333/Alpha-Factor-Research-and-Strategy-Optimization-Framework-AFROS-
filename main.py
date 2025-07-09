from optimize.optimisation import *
from backtest.backtest import *
from metrics.metrics import *
from data.data_loading import *
from functions.functions import *
from strategy.strategy import *

def run_pipeline():
    # Load prices 
    price_df = load_price_data()
    price_df = price_df[price_df['Date'] < '2025-01-01']

    # Construct Kelly portfolio weights
    kelly_weights = construct_kelly_portfolio(price_df, window=60, cap=0.01, scale=True, target_vol=0.05)
    
    # Calculate momentum (signals)
    momentum_scores, signals = ewma_momentum_signals(price_df)

    # Apply signal mask to Kelly weights
    filtered_kelly_weights = apply_signal_mask(kelly_weights, signals)
    
    # Backtest portfolio with fixed holding period
    kelly_returns = backtest_portfolio_holding_period(filtered_kelly_weights, price_df, holding_period=60)
    
    # Calculate performance metrics
    metrics = performance_metrics(kelly_returns)
    print("Cumulative Returns:", metrics['Cumulative Return'])
    print("Annualized Return:", metrics['Annualized Return'])
    print("Max Drawdown:", metrics['Max Drawdown'])
    print("Volatility:", metrics['Volatility']) 
    print("Sharpe Ratio:", metrics['Sharpe Ratio']) 

if __name__ == '__main__':
    run_pipeline()
    