from datetime import datetime, timedelta
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.main import check_and_make_directories
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from stable_baselines3.common.logger import configure
from finrl.config import INDICATORS
from stockstats import StockDataFrame as Sdf
from sqlalchemy import create_engine, TIMESTAMP, JSON
from pprint import pprint
from scripts.connection import connection
from scripts.data_filling import fill_data

import sys
sys.path.append("../FinRL-Library")
import itertools
import time
import json
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

table_name = 'hour_shares_data'
threshold_date = '2015-01-01'

# A2C
model_name_a2c = 'a2c'
model_kwargs_a2c = {'learning_rate':0.05, 'ent_coef': 0.01, 'gae_lambda':0.95,
                    'gamma':0.98, 'max_grad_norm':0.49, 'n_steps':4, 'normalize_advantage':True}
policy_kwargs_a2c = {'optimizer_kwargs': {'alpha': 0.90, 'eps': 1e-04, 'weight_decay': 0}}

# DDPG
model_name_ddpg = 'ddpg'
model_kwargs_ddpg= {'batch_size':128, 'buffer_size':50000, 'gamma':0.99, 'gradient_steps':2,
                    'learning_rate':0.01, 'learning_starts':500, 'tau':0.003, 'optimize_memory_usage': False}
policy_kwargs_ddpg={'n_critics': 1}

# PPO
model_name_ppo = 'ppo'
model_kwargs_ppo = {'batch_size':64, 'ent_coef':0.02, 'gae_lambda':0.95, 'gamma':0.99,
                    'learning_rate':0.003, 'max_grad_norm':0.5, 'n_epochs':10, 'n_steps':2048,
                    'normalize_advantage':True, 'sde_sample_freq':-1, 'vf_coef':0.5}
policy_kwargs_ppo={}

# TD3
model_name_td3 = 'td3'
model_kwargs_td3 = {'batch_size':100, 'buffer_size':50000, 'gamma':0.99, 'learning_rate':0.001,
                    'learning_starts':100, 'optimize_memory_usage':True, 'tau':0.005 }
policy_kwargs_td3={}

# SAC
model_name_sac = 'sac'
model_kwargs_sac = {"batch_size": 128, "buffer_size": 100000, 'gamma':0.99,
                    'gradient_steps':2, "learning_rate": 0.0001, "learning_starts": 100,
                    'optimize_memory_usage':False, "ent_coef": "auto_0.1", 'tau':0.005,
                    'target_entropy':-20, 'sde_sample_freq':-1}
policy_kwargs_sac = {'use_sde': False}

def extract_train_data(conn, selected, start_date, table_name, threshold_date):
    data = pd.read_sql_query(f"""
                            SELECT *
                            FROM {table_name}
                            WHERE time >= '{threshold_date}'
                            AND time < '{start_date}'
                            AND ticker in {selected}
                            """, conn)
    return data


def extract_trade_data(conn, selected, start_date, end_date, table_name):
    data = pd.read_sql_query(f"""
                            SELECT *
                            FROM {table_name}
                            WHERE time >= '{start_date}'
                            AND time <='{end_date}'
                            AND ticker in {selected}
                            """, conn)
    return data


def create_train_env(data, capital):
    stock_dimension = len(data.tic.unique())
    state_space =  len(INDICATORS)*stock_dimension + 1 + 2*stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": capital,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4}
    e_train_gym = StockTradingEnv(df = data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    return env_train


def train_model(env_train, model_name, model_kwargs, policy_kwargs, total_timesteps):
    start_time = datetime.now()
    agent = DRLAgent(env = env_train)
    model = agent.get_model(f"{model_name}", model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)
    date_model_init = datetime.now()
    trained_model = agent.train_model(model=model,
                                    tb_log_name=f"{model_name}",
                                    total_timesteps=total_timesteps)
    end_time = datetime.now()
    func_time = end_time - start_time
    return date_model_init, model_name, trained_model, total_timesteps, model_kwargs, policy_kwargs, func_time


def model_train_predict(selected, capital, start_date, end_date):
    if len(selected) == 1:
        selected = f'({selected[0]})'
    else:
        selected = tuple(selected)
    train_data = extract_train_data(connection(), selected, start_date, table_name, threshold_date)
    full_train_data = fill_data(train_data)
    full_train_data.rename(columns={'time': 'date', 'ticker': 'tic'}, inplace=True)
    full_train_data= full_train_data[['date',	'open',	'high',	'low',	'close', 'volume', 'tic']]
    fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_turbulence=True,
                     user_defined_feature = False)
    processed_train = fe.preprocess_data(full_train_data)
    env_train = create_train_env(processed_train, capital)

    date_model_init, model_name, trained_model, total_timesteps, model_kwargs, policy_kwargs, train_time = train_model(env_train, model_name, model_kwargs, policy_kwargs, total_timesteps)

    print(processed_train.head())
    return processed_train.head()
