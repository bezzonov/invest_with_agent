import pandas as pd
import numpy as np

def trades_history(trade, df_account_value, df_actions):

    action_dict = {}
    shares_dict = {}
    share_assets_dict = {}

    for date, row in df_actions.iterrows():
        non_zero_values = {col: [val, val * trade['close'][(trade['date'] == date) & (trade['tic'] == col)].values[0]] for col, val in row.items() if val != 0}
        if non_zero_values:
            action_dict[date] = non_zero_values
        else:
            action_dict[date] = np.nan
    action_dict[max(df_account_value['date'])] = np.nan
    action_dict_df = pd.DataFrame(list(action_dict.items()), columns=['date', 'actions'])

    for k, v in action_dict.items():
        if k == df_account_value['date'].values[0] and pd.isna(v):
            continue
        elif k == df_account_value['date'].values[0] and not pd.isna(v):
            shares_dict[k] = {share:[quanity[0], quanity[0]*trade['close'][(trade['date'] == k) & (trade['tic'] == share)].values[0]] for share, quanity in v.items()}
        elif pd.isna(v) and len(shares_dict) != 0:
            shares_dict[k] = {share:[quanity[0], quanity[0]*trade['close'][(trade['date'] == k) & (trade['tic'] == share)].values[0]] for share,quanity in sorted(shares_dict.items(), key=lambda x: x[0])[-1][1].items()}
        elif pd.isna(v) and len(shares_dict) == 0:
            continue
        else:
            if len(shares_dict) != 0:
                first_dict = {share:quanity[0] for share,quanity in sorted(shares_dict.items(), key=lambda x: x[0])[-1][1].items()}
                second_dict = {share:price[0] for share, price in v.items()}
                share_quanity_dict = {key: first_dict.get(key, 0) + second_dict.get(key, 0) for key in set(first_dict) | set(second_dict) if first_dict.get(key, 0) + second_dict.get(key, 0) != 0}
                shares_dict[k] = {share:[quanity, quanity*trade['close'][(trade['date'] == k) & (trade['tic'] == share)].values[0]] for share, quanity in share_quanity_dict.items()}
            elif len(shares_dict) == 0:
                shares_dict[k] = {share:[quanity[0], quanity[0]*trade['close'][(trade['date'] == k) & (trade['tic'] == share)].values[0]] for share, quanity in v.items()}
    share_dict_df = pd.DataFrame(list(shares_dict.items()), columns=['date', 'portfolio'])

    for k, v in shares_dict.items():
        share_assets_dict[k] = sum([asset[-1] for asset in v.values()])
    share_assets_dict_df = pd.DataFrame(list(share_assets_dict.items()), columns=['date', 'share_assets'])

    trades_history = df_account_value
    trades_history = trades_history.merge(action_dict_df, how='left', on='date')
    trades_history = trades_history.merge(share_dict_df, how='left', on='date')
    trades_history = trades_history.merge(share_assets_dict_df, how='left', on='date')
    trades_history['free_assets'] = trades_history['account_value'] - trades_history['share_assets']
    trades_history = trades_history[['date', 'account_value', 'share_assets', 'free_assets', 'actions', 'portfolio']]

    trades_history['account_value'] = trades_history['account_value'].apply(lambda x: round(x, 2))
    trades_history['share_assets'] = trades_history['share_assets'].apply(lambda x: round(x, 2))
    trades_history['free_assets'] = trades_history['free_assets'].apply(lambda x: round(x, 2))

    return trades_history
