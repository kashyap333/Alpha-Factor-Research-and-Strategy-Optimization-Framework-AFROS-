import pandas as pd
import os
import numpy as np
from datetime import datetime

def load_price_data(
    start_date='2020-01-01',
    end_date=datetime.today(),
    path="data\\master_stock_data.csv",
    merge=True
):
    if end_date is None:
        end_date = datetime.now()

    # Convert to Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    def load_and_filter(filepath, asset_type):
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df = df[(df['Date'] > start_date) & (df['Date'] <= end_date)]
        df['AssetType'] = asset_type
        return df

    # Load stock data
    combined = [load_and_filter(path, "Stock")]

    if merge:
        try:
            combined.append(load_and_filter("data\\master_bond_etf_data.csv", "Bond"))
        except FileNotFoundError:
            pass
        try:
            combined.append(load_and_filter("data\\master_commodity_etf_data.csv", "Commodity"))
        except FileNotFoundError:
            pass

    # Combine all into one DataFrame
    final_df = pd.concat(combined, ignore_index=True)
    final_df.sort_values(by="Date", inplace=True)

    return final_df