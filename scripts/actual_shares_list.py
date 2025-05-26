import pandas as pd
import numpy as np
from sqlalchemy import create_engine, TIMESTAMP, JSON
from datetime import datetime, timedelta

def connection():
    db_params = {
    'dbname': 'rl_trade',
    'user': 'bezzonov',
    'password': 'bezzonov_rl_trade',
    'host': '46.17.100.206',
    'port': '5432'}
    conn = create_engine(f'postgresql+psycopg2://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["dbname"]}')
    return conn

def shares_list_for_user(conn, table_name, threshold_date):
    query = f"""
        WITH shares_today_available as
            (
            SELECT DISTINCT ticker
            FROM {table_name}
            WHERE time = '{datetime.strftime(datetime.today() - timedelta(days=1), '%Y-%m-%d')}' or
                  time = '{datetime.strftime(datetime.today() - timedelta(days=2), '%Y-%m-%d')}' or
                  time = '{datetime.strftime(datetime.today() - timedelta(days=3), '%Y-%m-%d')}' 
            ),

        shares_rows as
            (
            SELECT ticker, count(*) as shares_rows_num
            FROM {table_name}
            WHERE time >= '{threshold_date}'
            GROUP BY ticker
            ORDER BY count(*) DESC
            ),

        shares_most_common as
            (
            SELECT ticker
            FROM shares_rows
            WHERE shares_rows_num = (
                                    select max(shares_rows_num)
                                    from shares_rows
                                    )
            )

        select t1.ticker, t3.name
        from shares_today_available t1
        inner join shares_most_common t2
        on t1.ticker = t2.ticker
        inner join shares_info t3
        on t1.ticker = t3.ticker
    """
    data = pd.read_sql_query(query, conn)
    return data

shares_list = {row[1]:row[2] for row in shares_list_for_user(connection(), 'hour_shares_data', '2017-01-01').itertuples()}
