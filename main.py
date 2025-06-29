from factors.momentum import *
from portfolio.risk_parity import *
from optimize.kelly_risk import *
from backtest.backtest import *
from metrics.metrics import *

def run_pipeline():
    # Load prices (wide format)
    price_df = load_price_data()

    # Construct Kelly portfolio weights
    kelly_weights = construct_kelly_portfolio(price_df, window=60)
    
    # Calculate momentum (signals)
    momentum_scores, signals = ewma_momentum_signals(price_df)

    # Apply signal mask to Kelly weights
    filtered_kelly_weights = apply_signal_mask(kelly_weights, signals)
    
    # Backtest portfolio with fixed holding period
    kelly_returns = backtest_portfolio_holding_period(filtered_kelly_weights, price_df, holding_period=20)
    
    # Calculate performance metrics
    metrics = performance_metrics(kelly_returns)
    print("Cumulative Returns:", metrics['Cumulative Return'])
    print("Volatility:", metrics['Volatility']) 
    print("Sharpe Ratio:", metrics['Sharpe Ratio']) 

if __name__ == '__main__':
    run_pipeline()
    