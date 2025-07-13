import pandas as pd
import os
import numpy as np
from datetime import datetime

def load_price_data(start_date='2020-01-01', end_date=datetime.today(), path="D:\\Quant\\afros\\data\\master_stock_data.csv"):
    if end_date is None:
        end_date = datetime.now()
    
    filepath = path
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Set Date as index
    df.set_index('Date', inplace=True)

    # Convert start_date and end_date to Timestamp if not already
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter by date range on index
    filtered_df = df.loc[(df.index > start_date) & (df.index <= end_date)]
    
    return filtered_df