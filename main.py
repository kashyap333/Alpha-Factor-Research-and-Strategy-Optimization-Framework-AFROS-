from factors.momentum import *
from portfolio.risk_parity import construct_portfolio

def run_pipeline():
    # Load prices (wide format)
    prices = load_price_data()

    # Calculate momentum (signals)
    momentum = momentum_factor(lookback=20)

    # Construct portfolio weights (risk parity) using momentum + prices
    weights = construct_portfolio(momentum_df=momentum, price_df=prices, window=60)

    print(weights.tail())

if __name__ == '__main__':
    run_pipeline()
