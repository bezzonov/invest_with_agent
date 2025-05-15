import asyncio
import pandas as pd
import requests
from datetime import datetime, timedelta
from sqlalchemy import create_engine, TIMESTAMP
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

def connection():
    db_params = {
        'dbname': 'rl_trade',
        'user': 'bezzonov',
        'password': 'bezzonov_rl_trade',
        'host': '46.17.100.206',
        'port': '5432'
    }
    conn_str = f'postgresql+psycopg2://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["dbname"]}'
    return create_engine(conn_str)

def fetch_index_close_data(index, date_from):

    all_data = []
    start = 0
    limit = 100

    while True:
        params = {
            "start": start,
            "limit": limit
        }
        url = f'https://iss.moex.com/iss/history/engines/stock/markets/index/securities/{index}.json?from={date_from}'
        response = requests.get(url, params=params)
        response.raise_for_status()
        json_data = response.json()

        history = json_data.get("history")
        if not history:
            break

        columns = history.get("columns", [])
        rows = history.get("data", [])
        if not rows:
            break

        df = pd.DataFrame(rows, columns=columns)
        all_data.append(df)

        cursor = json_data.get("cursor", {})
        total = cursor.get("TOTAL", 0)

        start += limit
        if start >= total:
            break

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df = full_df[["TRADEDATE", "CLOSE"]]
        full_df["TRADEDATE"] = pd.to_datetime(full_df["TRADEDATE"])
        full_df = full_df.sort_values("TRADEDATE").reset_index(drop=True)
        full_df = full_df.rename(columns={"TRADEDATE" : "date", "CLOSE" : f"{index.lower()}"})
        return full_df
    else:
        return pd.DataFrame()

def get_one_index_info(index, dates_from):
    df_index = pd.DataFrame()
    for date in dates_from:
        df_part = fetch_index_close_data(index, date)
        df_index = pd.concat([df_index, df_part])
    df_index = df_index.drop_duplicates(subset=['date'])
    return df_index

async def periodic_task(interval_hours=3):
    engine = connection()
    indexes = ["IMOEX", "RTSI", "MOEXBC", "MOEXOG", "MOEXEU", "MOEXFN"]

    while True:
        today_str = datetime.today().strftime('%Y-%m-%d')
        today_date = pd.to_datetime(today_str)

        df_index_full_today = pd.DataFrame([today_date], columns=['date'])

        try:
            for ind in indexes:
                df_index_today = get_one_index_info(ind, [today_str])
                if df_index_today.empty:
                    logger.info(f"[{datetime.now()}] Нет данных по индексу {ind} за {today_str}")
                    continue
                df_index_today['date'] = pd.to_datetime(df_index_today['date'])
                df_index_full_today = df_index_full_today.merge(df_index_today, how='left', on='date')

            if df_index_full_today.drop(columns=['date']).isnull().all(axis=1).iloc[0]:
                logger.info(f"[{datetime.now()}] Нет данных за сегодня по всем индексам, пропуск обновления.")
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                yesterday_data = pd.read_sql_query(f"SELECT * FROM stock_market_indexes WHERE date = '{yesterday}'", connection())
                yesterday_data['date'] = datetime.today().date()
                yesterday_data.to_sql(
                        'stock_market_indexes',
                        con=engine,
                        if_exists='append',
                        index=False,
                        dtype={'date': TIMESTAMP}
                    )
                logger.info(f"[{datetime.now()}] Добавлены данные за вчера, обновим их, как появятся за сегодня.")

            else:
                query = f"SELECT date FROM stock_market_indexes WHERE date = '{today_str}'"
                existing = pd.read_sql(query, engine)

                if not existing.empty:
                    with engine.begin() as conn:
                        conn.execute(text(f"DELETE FROM stock_market_indexes WHERE date = DATE '{today_str}'"))
                    logger.info(f"[{datetime.now()}] Удалена старая запись за {today_str}")

                df_index_full_today.to_sql(
                    'stock_market_indexes',
                    con=engine,
                    if_exists='append',
                    index=False,
                    dtype={'date': TIMESTAMP}
                )
                logger.info(f"[{datetime.now()}] Данные за {today_str} обновлены в БД.")

        except SQLAlchemyError as e:
            logger.info(f"[{datetime.now()}] Ошибка работы с БД: {e}")
        except Exception as e:
            logger.info(f"[{datetime.now()}] Ошибка при обновлении данных: {e}")

        await asyncio.sleep(interval_hours * 3600)

if __name__ == "__main__":
    asyncio.run(periodic_task())