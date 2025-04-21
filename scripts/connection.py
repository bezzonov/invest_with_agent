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