import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, TIMESTAMP
from scripts.data_filling import fill_data

def connection():
    db_params = {
    'dbname': 'rl_trade',
    'user': 'bezzonov',
    'password': 'bezzonov_rl_trade',
    'host': '46.17.100.206',
    'port': '5432'}
    conn = create_engine(f'postgresql+psycopg2://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["dbname"]}')
    return conn

def plot_price_chart(conn, ticker, table_name, threshold_date):
    shares_data = pd.read_sql_query(f"""select *
                                    from {table_name}
                                    where ticker = '{ticker}'
                                    and time >= '{threshold_date}'""", conn)

    full_shares_data = fill_data(shares_data)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=full_shares_data['time'],
        open=full_shares_data['open'],
        high=full_shares_data['high'],
        low=full_shares_data['low'],
        close=full_shares_data['close'],
        name=ticker,
        visible=False
    ))
    fig.data[0].visible = True

    buttons = []

    fig.update_layout(
        title=f'Димамика стоимости акции {ticker}',
        xaxis_title='Дата',
        yaxis_title='Стоимость акции (руб.)',
        xaxis_rangeslider_visible=False,
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
        }],
        height=450,
        width=1300


    )

    return fig
