import streamlit as st
import pandas as pd
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, TIMESTAMP

def connection():
    db_params = {
    'dbname': 'rl_trade',
    'user': 'bezzonov',
    'password': 'bezzonov_rl_trade',
    'host': '46.17.100.206',
    'port': '5432'}
    conn = create_engine(f'postgresql+psycopg2://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["dbname"]}')
    return conn

def format_number(value):
    if isinstance(value, (int, float)):
        return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
    return str(value)

def show_metrics(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(
                f"""
                <div style="
                    border: 2px solid #ccc;
                    border-radius: 8px;
                    padding: 5px;
                    margin-bottom:20px;
                    text-align: center;
                    background-color: #f9f9f9;
                    ">
                    <div style="font-size: 18px; font-weight: bold; color: #333;">
                        {format_number(value)}
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def calc_metrics(conn, ticker, table_name, threshold_date):

    data = pd.read_sql_query(f"""select *
                                from {table_name}
                                where ticker = '{ticker}'
                                and time >= '{threshold_date}'
                            """, conn)
    price_today = data['close'][data['time'] == max(data.time)].values[0]
    price_year_ago = data['close'][data['time'] == datetime.strftime(datetime.today() - timedelta(days=364), '%Y-%m-%d')].values[0]
    price_month_ago = data['close'][data['time'] == datetime.strftime(datetime.today() - timedelta(days=7*4), '%Y-%m-%d')].values[0]

    metrics = {
        'Текущая стоимость' : f"{price_today} руб.",
        'Волатильность' : f"{round(data['close'].rolling(window=60).std().values[-1],1)} руб.",
        'Изменение цены за месяц' : f"{round(100*(price_today - price_month_ago)/price_month_ago, 1)} %",
        'Изменение цены за год' : f"{round(100*(price_today - price_year_ago)/price_year_ago, 1)} %"
    }
    return metrics
