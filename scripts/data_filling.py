import pandas as pd
import numpy as np

def fill_data(df):

    date_range = pd.date_range(start=min(df['time']), end=max(df['time']))
    unique_tickers = df['ticker'].unique()
    nunique_tickers = df['ticker'].nunique()

    full_df = pd.DataFrame({'time': np.repeat(date_range, nunique_tickers)})
    full_df['ticker'] = full_df.time.nunique() * list(unique_tickers)
    full_df = pd.merge(full_df, df, on=['time', 'ticker'], how='left')
    full_df[['open', 'high', 'low', 'close', 'volume']] = full_df.groupby('ticker')[['open', 'high', 'low', 'close', 'volume']].ffill()
    full_df = full_df.drop(columns=['id'])

    return full_df