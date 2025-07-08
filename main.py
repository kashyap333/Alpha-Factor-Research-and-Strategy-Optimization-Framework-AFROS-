from factors.momentum import *
from portfolio.risk_parity import *
from optimize.kelly_risk import *
from backtest.backtest import *
from metrics.metrics import *

def run_pipeline():
    # Load prices (wide format)
    price_df = load_price_data()
    returns_df = price_df.pct_change().dropna()

    # Construct Kelly portfolio weights
    kelly_weights = construct_kelly_portfolio(price_df, window=60)
    
    # Calculate momentum (signals)
    momentum_scores, signals = ewma_momentum_signals(price_df)
    print(signals)
    # Apply signal mask to Kelly weights
    filtered_kelly_weights = apply_signal_mask(kelly_weights, signals)
    
    # Volatility targeting
    vol_targeted_weights = scale_to_target_volatility(filtered_kelly_weights, returns_df, target_vol=0.05)
    
    # Backtest portfolio with fixed holding period
    kelly_returns = backtest_portfolio_holding_period(vol_targeted_weights, price_df, holding_period=60)
    
    # Calculate performance metrics
    metrics = performance_metrics(kelly_returns)
    print("Cumulative Returns:", metrics['Cumulative Return'])
    print("Annualized Return:", metrics['Annualized Return'])
    print("Max Drawdown:", metrics['Max Drawdown'])
    print("Volatility:", metrics['Volatility']) 
    print("Sharpe Ratio:", metrics['Sharpe Ratio']) 

if __name__ == '__main__':
    run_pipeline()
    