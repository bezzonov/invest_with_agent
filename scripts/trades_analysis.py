import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from scripts.config import sector_info

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


def calculate_fifo_portfolio(trades, df_account_value):
    shares_last_day = trades['portfolio'][trades['date'] == max(df_account_value['date'])].values[0].keys()
    data = defaultdict(lambda: {'покупки': [], 'продажи': []})

    for actions in trades['actions']:
        if pd.isna(actions):
            continue
        else:
            for key, val in actions.items():
                if key not in shares_last_day:
                    continue
                qty, price = val
                if qty > 0:
                    data[key]['покупки'] += [qty, price]
                elif qty < 0:
                    data[key]['продажи'] += [abs(qty), abs(price)]

    result = {}

    for stock, trades in data.items():
        purchases_raw = trades.get('покупки', [])
        sales_raw = trades.get('продажи', [])

        purchases = [(purchases_raw[i], purchases_raw[i+1]/purchases_raw[i]) for i in range(0, len(purchases_raw), 2)]
        sales = [(sales_raw[i], sales_raw[i+1]/sales_raw[i]) for i in range(0, len(sales_raw), 2)]

        purchase_queue = [{'qty': qty, 'price': price} for qty, price in purchases]

        for sold_qty, sold_price in sales:
            qty_to_sell = sold_qty
            while qty_to_sell > 0 and purchase_queue:
                lot = purchase_queue[0]
                if lot['qty'] <= qty_to_sell:
                    qty_to_sell -= lot['qty']
                    purchase_queue.pop(0)
                else:
                    lot['qty'] -= qty_to_sell
                    qty_to_sell = 0

        total_qty = sum(lot['qty'] for lot in purchase_queue)
        if total_qty == 0:
            average_price = 0
        else:
            total_cost = sum(lot['qty'] * lot['price'] for lot in purchase_queue)
            average_price = total_cost / total_qty

        result[stock] = round(average_price, 2)

    return result


def calc_profit(trades, portfolio_structure):

    data = defaultdict(lambda: {'покупки': [], 'продажи': []})

    for actions in trades['actions']:
        if pd.isna(actions):
            continue
        else:
            for key, val in actions.items():
                qty, price = val
                if qty > 0:
                    data[key]['покупки'] += [qty, price]
                elif qty < 0:
                    data[key]['продажи'] += [abs(qty), abs(price)]

    current_prices = dict(zip(portfolio_structure['Акция'], portfolio_structure['Стоимость в ПДТ']))


    # for actions in trades['actions']:
    #     if pd.isna(actions):
    #         continue
    #     else:
    #         for key, val in actions.items():
    #             if key not in data:
    #                 data[key] = val[1]
    #             else:
    #                 data[key] += val[1]

    # not_saled = portfolio_structure[['Акция', 'Количество (шт.)', 'Средняя стоимость покупки']][portfolio_structure['Акция'] != '']
    # # for share in not_saled['Акция']:
    # #     # print(share)
    # #     # print(data[share])
    # #     # print(not_saled['Общая стоимость (руб.)'][not_saled['Акция'] == share])
    # #     data[share] = not_saled['Количество (шт.)'][not_saled['Акция'] == share].values[0] * not_saled['Средняя стоимость покупки'][not_saled['Акция'] == share].values[0] - data[share]
    # data = dict(sorted(data.items(), key=lambda x: x[1]))

    # stocks = list(data.keys())
    # profits = [int(i) for i in list(data.values())]

    # colors = ['#75e68f' if val >= 0 else '#ff1a6a' for val in profits]
    # border_colors = ['#3c8d40' if val >= 0 else '#b3134a' for val in profits]  # Темнее для обводки

    # fig = go.Figure()

    # fig.add_trace(go.Bar(
    #     x=stocks,
    #     y=profits,
    #     marker_color=colors,
    #     marker_line_color=border_colors,
    #     marker_line_width=2,
    #     text=[f"{val:,} ₽" for val in profits],  # форматируем числа с разделителем тысяч и ₽
    #     textposition=['outside' if val >= 0 else 'inside' for val in profits],
    #     textfont=dict(
    #         color=border_colors,
    #         size=14
    #     ),
    #     insidetextanchor='middle'
    # ))

    # fig.update_layout(
    #     title='Профит по акциям',
    #     yaxis_title='Сумма (руб.)',
    #     xaxis_title='Акция',
    #     yaxis=dict(
    #         zeroline=True,
    #         rangemode='tozero',
    #         showgrid=True,
    #         gridwidth=1,
    #         zerolinecolor='black',
    #         zerolinewidth=2,
    #         minor=dict(
    #             ticklen=4,
    #             showgrid=True,
    #             gridcolor='LightGray',
    #             gridwidth=0.5
    #         )
    #     ),
    #     xaxis=dict(
    #         showgrid=False,
    #         tickfont=dict(size=14)
    #     ),
    #     bargap=0.3,
    #     plot_bgcolor='white',
    #     margin=dict(t=60, b=60)
    # )

    # return fig

    results = {}

    for stock, trades in data.items():
        # Преобразуем покупки и продажи в списки (qty, price_per_share)
        purchases_raw = trades.get('покупки', [])
        sales_raw = trades.get('продажи', [])

        # Преобразуем из [qty, total_cost, ...] в [(qty, price_per_share), ...]
        purchases = []
        for i in range(0, len(purchases_raw), 2):
            qty = purchases_raw[i]
            total_cost = purchases_raw[i+1]
            price = total_cost / qty if qty != 0 else 0
            purchases.append({'qty': qty, 'price': price})

        sales = []
        for i in range(0, len(sales_raw), 2):
            qty = sales_raw[i]
            total_cost = sales_raw[i+1]
            price = total_cost / qty if qty != 0 else 0
            sales.append({'qty': qty, 'price': price})

        purchase_queue = purchases.copy()
        profit_realized = 0.0

        # Расчет реализованной прибыли методом FIFO
        for sale in sales:
            qty_to_sell = sale['qty']
            sale_price = sale['price']

            while qty_to_sell > 0 and purchase_queue:
                lot = purchase_queue[0]
                if lot['qty'] <= qty_to_sell:
                    profit_realized += lot['qty'] * (sale_price - lot['price'])
                    qty_to_sell -= lot['qty']
                    purchase_queue.pop(0)
                else:
                    profit_realized += qty_to_sell * (sale_price - lot['price'])
                    lot['qty'] -= qty_to_sell
                    qty_to_sell = 0

            # Если продано больше, чем куплено (короткая позиция), можно обработать отдельно

        # Остаток акций после продаж
        total_qty = sum(lot['qty'] for lot in purchase_queue)
        if total_qty > 0:
            total_cost = sum(lot['qty'] * lot['price'] for lot in purchase_queue)
            avg_price = total_cost / total_qty
        else:
            avg_price = 0.0

        # Нереализованный доход по остаткам
        current_price = current_prices.get(stock, avg_price)  # если нет текущей цены, берем avg_price
        unrealized_profit = (current_price - avg_price) * total_qty if total_qty > 0 else 0.0

        results[stock] = round(profit_realized + unrealized_profit, 2)
            # 'реализованная_прибыль': round(profit_realized, 2),
            # 'нереализованный_доход': round(unrealized_profit, 2),

            # 'остаток_акций': total_qty,
            # 'средняя_цена_остатка': round(avg_price, 2)




    stocks = list(results.keys())
    values = [int(results[stock]) for stock in stocks]

    sorted_data = sorted(zip(stocks, values), key=lambda x: x[1])
    sorted_stocks, sorted_values = zip(*sorted_data) if sorted_data else ([], [])

    colors = ['#adffb3' if val >= 0 else '#ffa8a8' for val in sorted_values]
    border_colors = ['#008f21' if val >= 0 else '#bd0000' for val in sorted_values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(sorted_stocks),
        y=list(sorted_values),
        marker_color=colors,
        marker_line_color=border_colors,
        marker_line_width=1.5,
        text=[f"{val:,}" for val in sorted_values],
        textposition=['outside' if val >= 0 else 'inside' for val in sorted_values],
        textfont=dict(color=border_colors, size=14)
    ))

    fig.update_layout(
        title='Профит по акциям',
        yaxis_title='Сумма (руб.)',
        xaxis_title='Акция',
        yaxis=dict(
            zeroline=True,
            rangemode='tozero',
            showgrid=True,
            gridwidth=1,
            zerolinecolor='black',
            zerolinewidth=2,
            minor=dict(
                ticklen=4,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.5
            )
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=14)
        ),
        bargap=0.3,
        plot_bgcolor='white',
        margin=dict(t=60, b=60)
    )


    return fig, results #, shares_activity

def shares_tree(trades, tab):
    shares_activity = {}
    for actions in trades['actions']:
        if pd.isna(actions):
            continue
        else:
            for key, val in actions.items():
                qty, price = val
                if  key not in shares_activity:
                    shares_activity[key] = qty
                else:
                    if qty > 0:
                        shares_activity[key] += qty

        df = pd.DataFrame.from_dict(sector_info, orient='index', columns=['Сектор'])
        df['Куплено, шт.'] = df.index.map(shares_activity)
        df['Профит, руб.'] = df.index.map(tab)
        df = df.dropna(subset=['Куплено, шт.'])
        df = df.reset_index().rename(columns={'index': 'Акция'})

    fig = px.treemap(
        df,
        path=['Сектор', 'Акция'],
        color='Профит, руб.',           # Цвет по профиту
        values='Куплено, шт.',          # Размер по количеству куплено
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,    # Центр цвета на 0 для баланса прибыли/убытка
        hover_data={'Профит, руб.': True, 'Куплено, шт.': True},
        title='Дерево акций по секторам с отображением прибыли и количества купленных акций'
    )

    fig.update_traces(
        root_color="white",
        marker_line_width=0,
        tiling=dict(pad=0),
        hovertemplate='<b>%{label}</b><br>Куплено: %{value:.0f} шт.<extra></extra>'
    )

    fig.update_layout(
        height=450,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=40, l=60, r=24, b=30),  # увеличен левый отступ для цветбара
        font=dict(size=14, family="Arial, sans-serif", color="#2a3f5f"),
        coloraxis_colorbar=dict(
            title="Профит,<br>руб.",
            thickness=15,
            len=0.9,
            x=-0.15,   # сдвиг цветовой шкалы влево (отрицательное значение)
            y=0.5
        )
    )

    return fig, df


# def sharpe(df):
#     df['daily_return'] = df['account_value'].pct_change()
#     returns = df['daily_return'].dropna()

#     risk_free_rate = 0.11  # годовая безрисковая ставка
#     trading_days = df.shape[0]

#     daily_risk_free = (1 + risk_free_rate)**(1/trading_days) - 1
#     excess_returns = returns - daily_risk_free

#     sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days)
#     return round(sharpe_ratio,2)

def max_drawdown(account_values):
    """
    Рассчитывает максимальную просадку по серии значений стоимости портфеля.
    """
    running_max = account_values.cummax()
    drawdowns = (account_values - running_max) / running_max
    return drawdowns.min()


def max_runup(account_values):
    """
    Рассчитывает максимальный рост (run-up) по серии значений стоимости портфеля.
    """
    running_min = account_values.cummin()
    runups = (account_values - running_min) / running_min
    return runups.max()


def volatility(account_values):
    """
    Рассчитывает годовую волатильность по дневным доходностям портфеля.
    """
    daily_returns = account_values.pct_change().dropna()
    std_daily = daily_returns.std()
    return std_daily * np.sqrt(len(account_values))