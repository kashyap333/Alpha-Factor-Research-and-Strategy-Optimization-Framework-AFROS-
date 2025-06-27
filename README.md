# AFROS: Alpha Factor Research and Optimization System

A modular pipeline for building and evaluating systematic trading strategies using alpha factors, robust backtesting, and advanced portfolio construction techniques.

## Key Features
- Factor generation (momentum, value, sentiment, etc.)
- Walk-forward backtesting with realistic constraints
- Portfolio construction: Risk Parity, HRP, Min Var
- ML-driven tuning of strategies
- Execution modeling and performance attribution

## How to Run
```bash
python main.py
```

## Requirements
- pandas, numpy, scipy, sklearn, matplotlib
- cvxpy, optuna, statsmodels

## Project Structure
See inline comments in `main.py` for the pipeline steps.