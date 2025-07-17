import yfinance as yf
import pandas as pd
import datetime
import os

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
sp500_df = tables[0]

# Get the symbol list and fix formats for Yahoo Finance
symbols = sp500_df['Symbol'].str.replace('.', '-', regex=False).tolist()
today = datetime.datetime.today().date()
master_file = 'Data/master_stock_data.csv'

if os.path.exists(master_file):
    with open(master_file, 'r') as f:
        first_line = f.readline()
        second_line = f.readline()

    skiprows = [1] if 'Ticker' in second_line else []
    master_df = pd.read_csv(master_file, parse_dates=['Date'], skiprows=skiprows)

    # Clean up any legacy pivoted/multiindex columns
    expected = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'}
    master_df = master_df[[col for col in master_df.columns if col in expected]]

else:
    master_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])

new_rows = []

for symbol in symbols:
    symbol_df = master_df[master_df['Symbol'] == symbol] if not master_df.empty else pd.DataFrame()

    if not symbol_df.empty:
        last_date = symbol_df['Date'].max().date()
        start_date = last_date + datetime.timedelta(days=1)
    else:
        start_date = datetime.date(2020, 1, 1)

    if start_date >= today:
        print(f"{symbol} is already up to date.")
        continue

    print(f"Downloading {symbol} from {start_date} to {today}")
    new_data = yf.download(symbol, start=start_date, end=today + datetime.timedelta(days=1), auto_adjust=True, group_by='column')

    if not new_data.empty:
        # Flatten columns if MultiIndex
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = [col[0] for col in new_data.columns]

        new_data = new_data.reset_index()
        new_data['Symbol'] = symbol
        new_rows.append(new_data)
    else:
        print(f"No new data for {symbol}.")

if new_rows:
    update_df = pd.concat(new_rows, ignore_index=True)

    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
    update_df = update_df[[col for col in required_columns if col in update_df.columns]]

    master_df = pd.concat([master_df, update_df], ignore_index=True)
    master_df = master_df.drop_duplicates(subset=['Date', 'Symbol'])
    master_df = master_df.sort_values(by=['Date', 'Symbol'])

    master_df.to_csv(master_file, index=False)
    
import yfinance as yf
import pandas as pd
import datetime
import os

# Bond ETFs you want to track
bond_etfs = [
    # U.S. Treasury Bond ETFs
    "IEF",  # iShares 7-10 Year Treasury
    "SHY",  # iShares 1-3 Year Treasury
    "TLT",  # iShares 20+ Year Treasury
    "VGSH", # Vanguard Short-Term Treasury
    "VGIT", # Vanguard Intermediate-Term Treasury
    "VGLT", # Vanguard Long-Term Treasury

    # Corporate Bond ETFs
    "LQD",  # iShares Investment Grade Corporate Bond
    "VCIT", # Vanguard Intermediate-Term Corporate Bond
    "VCLT", # Vanguard Long-Term Corporate Bond

    # High Yield (Junk) Bond ETFs
    "HYG",  # iShares iBoxx $ High Yield Corporate Bond
    "JNK",  # SPDR Bloomberg High Yield Bond

    # Municipal Bond ETFs
    "MUB",  # iShares National Muni Bond
    "VTEB", # Vanguard Tax-Exempt Bond ETF (correct ticker)
    
    # International Bond ETFs
    "BNDX", # Vanguard Total International Bond
    "IGOV", # iShares International Treasury Bond

    # Inflation-Protected Bond ETFs
    "TIP",  # iShares TIPS Bond
    "VTIP", # Vanguard Short-Term Inflation-Protected Securities

    # Aggregate Bond ETFs (Broad Market)
    "AGG",  # iShares Core U.S. Aggregate Bond
    "BND"   # Vanguard Total Bond Market
]

today = datetime.datetime.today().date()
master_file = 'Data/master_bond_etf_data.csv'

# Load existing master file if it exists
if os.path.exists(master_file):
    with open(master_file, 'r') as f:
        first_line = f.readline()
        second_line = f.readline()

    skiprows = [1] if 'Ticker' in second_line else []
    master_df = pd.read_csv(master_file, parse_dates=['Date'], skiprows=skiprows)

    # Clean to expected columns
    expected = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'}
    master_df = master_df[[col for col in master_df.columns if col in expected]]
else:
    master_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])

new_rows = []

for symbol in bond_etfs:
    symbol_df = master_df[master_df['Symbol'] == symbol] if not master_df.empty else pd.DataFrame()

    if not symbol_df.empty:
        last_date = symbol_df['Date'].max().date()
        start_date = last_date + datetime.timedelta(days=1)
    else:
        start_date = datetime.date(2020, 1, 1)

    if start_date >= today:
        print(f"{symbol} is already up to date.")
        continue

    print(f"Downloading {symbol} from {start_date} to {today}")
    new_data = yf.download(symbol, start=start_date, end=today + datetime.timedelta(days=1), auto_adjust=True, group_by='column', progress=False)

    if not new_data.empty:
        # Flatten columns if MultiIndex (should not happen for single ticker but safe)
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = [col[0] for col in new_data.columns]

        new_data = new_data.reset_index()
        new_data['Symbol'] = symbol
        new_rows.append(new_data)
    else:
        print(f"No new data for {symbol}.")

if new_rows:
    update_df = pd.concat(new_rows, ignore_index=True)

    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
    update_df = update_df[[col for col in required_columns if col in update_df.columns]]

    master_df = pd.concat([master_df, update_df], ignore_index=True)
    master_df = master_df.drop_duplicates(subset=['Date', 'Symbol'])
    master_df = master_df.sort_values(by=['Date', 'Symbol'])

    # Make sure folder exists
    os.makedirs(os.path.dirname(master_file), exist_ok=True)

    master_df.to_csv(master_file, index=False)
    print(f"Updated master bond ETF data saved to {master_file}")
else:
    print("All bond ETFs are already up to date.")


import yfinance as yf
import pandas as pd
import datetime
import os

# Commodity ETFs you want to track
commodity_etfs = [
    "GLD",   # Gold
    "SLV",   # Silver
    "USO",   # Crude Oil (WTI)
    "UNG",   # Natural Gas
    "DBA",   # Agriculture
    "DBC",   # Broad Commodities Index
    "PALL",  # Palladium
    "PPLT",  # Platinum
    "BAL",   # Element 47 Balancer ETF (metals)
    "RJA",   # Agriculture
    "JO",    # Coffee
    "CORN",  # Corn
    "WEAT",  # Wheat
    "SOYB",  # Soybeans
    "LIT",   # Lithium & Battery Metals
    "WOOD",  # Timber/Forestry
]

today = datetime.datetime.today().date()
master_file = 'Data/master_commodity_etf_data.csv'

# Load existing master file if it exists
if os.path.exists(master_file):
    with open(master_file, 'r') as f:
        first_line = f.readline()
        second_line = f.readline()

    skiprows = [1] if 'Ticker' in second_line else []
    master_df = pd.read_csv(master_file, parse_dates=['Date'], skiprows=skiprows)

    # Clean to expected columns
    expected = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'}
    master_df = master_df[[col for col in master_df.columns if col in expected]]
else:
    master_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])

new_rows = []

for symbol in commodity_etfs:
    symbol_df = master_df[master_df['Symbol'] == symbol] if not master_df.empty else pd.DataFrame()

    if not symbol_df.empty:
        last_date = symbol_df['Date'].max().date()
        start_date = last_date + datetime.timedelta(days=1)
    else:
        start_date = datetime.date(2020, 1, 1)

    if start_date >= today:
        print(f"{symbol} is already up to date.")
        continue

    print(f"Downloading {symbol} from {start_date} to {today}")
    new_data = yf.download(
        symbol,
        start=start_date,
        end=today + datetime.timedelta(days=1),
        auto_adjust=True,
        group_by='column',
        progress=False
    )

    if not new_data.empty:
        # Flatten columns if MultiIndex (usually not needed for single ticker)
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = [col[0] for col in new_data.columns]

        new_data = new_data.reset_index()
        new_data['Symbol'] = symbol
        new_rows.append(new_data)
    else:
        print(f"No new data for {symbol}.")

if new_rows:
    update_df = pd.concat(new_rows, ignore_index=True)

    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
    update_df = update_df[[col for col in required_columns if col in update_df.columns]]

    master_df = pd.concat([master_df, update_df], ignore_index=True)
    master_df = master_df.drop_duplicates(subset=['Date', 'Symbol'])
    master_df = master_df.sort_values(by=['Date', 'Symbol'])

    # Make sure folder exists
    os.makedirs(os.path.dirname(master_file), exist_ok=True)

    master_df.to_csv(master_file, index=False)
    print(f"Updated master commodity ETF data saved to {master_file}")
else:
    print("All commodity ETFs are already up to date.")
