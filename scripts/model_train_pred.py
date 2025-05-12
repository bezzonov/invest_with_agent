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
from babel.dates import format_date
from pypfopt.efficient_frontier import EfficientFrontier


import streamlit as st
from scripts.connection import connection
from scripts.data_filling import fill_data
from scripts.compare_fig import plot_compare_chart

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
threshold_date = '2018-01-01'

# # A2C
# model_name_a2c = 'a2c'
# model_kwargs_a2c = {'learning_rate':0.05, 'ent_coef': 0.01, 'gae_lambda':0.95,
#                     'gamma':0.98, 'max_grad_norm':0.49, 'n_steps':4, 'normalize_advantage':True}
# policy_kwargs_a2c = {'optimizer_kwargs': {'alpha': 0.90, 'eps': 1e-04, 'weight_decay': 0}}

# # DDPG
# model_name_ddpg = 'ddpg'
# model_kwargs_ddpg= {'batch_size':128, 'buffer_size':50000, 'gamma':0.99, 'gradient_steps':2,
#                     'learning_rate':0.01, 'learning_starts':500, 'tau':0.003, 'optimize_memory_usage': False}
# policy_kwargs_ddpg={'n_critics': 1}

# # PPO
# model_name_ppo = 'ppo'
# model_kwargs_ppo = {'batch_size':64, 'ent_coef':0.02, 'gae_lambda':0.95, 'gamma':0.99,
#                     'learning_rate':0.003, 'max_grad_norm':0.5, 'n_epochs':10, 'n_steps':2048,
#                     'normalize_advantage':True, 'sde_sample_freq':-1, 'vf_coef':0.5}
# policy_kwargs_ppo={}

# # TD3
# model_name_td3 = 'td3'
# model_kwargs_td3 = {'batch_size':100, 'buffer_size':50000, 'gamma':0.99, 'learning_rate':0.001,
#                     'learning_starts':100, 'optimize_memory_usage':True, 'tau':0.005 }
# policy_kwargs_td3={}

# # SAC
# model_name_sac = 'sac'
# model_kwargs_sac = {"batch_size": 128, "buffer_size": 100000, 'gamma':0.99,
#                     'gradient_steps':2, "learning_rate": 0.0001, "learning_starts": 100,
#                     'optimize_memory_usage':False, "ent_coef": "auto_0.1", 'tau':0.005,
#                     'target_entropy':-20, 'sde_sample_freq':-1}
# policy_kwargs_sac = {'use_sde': False}

def extract_train_data(conn, selected, start_date, table_name, threshold_date):
    data = pd.read_sql_query(f"""
                            SELECT *
                            FROM {table_name}
                            WHERE time >= '{threshold_date}'
                            AND time < '{start_date}'
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


def train_model(env_train, model_name, model_params):
    model_kwargs = {}
    policy_kwargs = {}
    for key, val in model_params.items():
        if key in ['learning_rate', 'ent_coef', 'gae_lambda', 'gamma', 'max_grad_norm', 'n_steps', 'normalize_advantage',
                   'batch_size', 'buffer_size', 'gradient_steps', 'learning_starts', 'vf_coef', 'tau', 'train_freq',
                   'target_entropy', 'optimize_memory_usage']:
            model_kwargs[key] = val
        elif key == 'total_timesteps':
            continue
        elif model_name == 'a2c':
            policy_kwargs =  {'optimizer_kwargs': {'alpha': model_params['alpha'], 'eps': model_params['eps']}}
        elif model_name == 'sac':
            model_kwargs['gradient_clip'] = 0.0001
            policy_kwargs['use_sde'] = False
        else:
            policy_kwargs[key] = val
    agent = DRLAgent(env = env_train)
    model = agent.get_model(f"{model_name}", model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)
    trained_model = agent.train_model(model=model,
                                    tb_log_name=f"{model_name}",
                                    total_timesteps=model_params['total_timesteps'])
    return trained_model


def extract_trade_data(conn, selected, start_date, end_date, table_name):
    data = pd.read_sql_query(f"""
                            SELECT *
                            FROM {table_name}
                            WHERE time >= '{start_date}'
                            AND time <='{end_date}'
                            AND ticker in {selected}
                            """, conn)
    return data


def create_trade_env(data, capital):
    stock_dimension = len(data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
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
        "reward_scaling": 1e-4
    }
    e_trade_gym = StockTradingEnv(df=data, turbulence_threshold = 70, **env_kwargs)
    return e_trade_gym

def mvo_strategy(processed_train, processed_trade, capital):

    stock_dimension = len(processed_trade.tic.unique())

    def process_df_for_mvo(df):
        df = df.sort_values(['date','tic'],ignore_index=True)[['date','tic','close']]
        fst = df
        fst = fst.iloc[0:stock_dimension, :]
        tic = fst['tic'].tolist()

        mvo = pd.DataFrame()

        for k in range(len(tic)):
            mvo[tic[k]] = 0

        for i in range(df.shape[0]//stock_dimension):
            n = df
            n = n.iloc[i * stock_dimension:(i+1) * stock_dimension, :]
            date = n['date'][i*stock_dimension]
            mvo.loc[date] = n['close'].tolist()

        return mvo

    def StockReturnsComputing(StockPrice, Rows, Columns):
        StockReturn = np.zeros([Rows-1, Columns])
        for j in range(Columns):        # j: Assets
            for i in range(Rows-1):     # i: Daily Prices
                StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100
        return StockReturn

    StockData = process_df_for_mvo(processed_train)
    TradeData = process_df_for_mvo(processed_trade)

    TradeData.to_numpy()

    arStockPrices = np.asarray(StockData)
    [Rows, Cols]=arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)
    meanReturns = np.mean(arReturns, axis = 0)
    covReturns = np.cov(arReturns, rowvar=False)
    np.set_printoptions(precision=3, suppress = True)

    ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array([capital * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])

    LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

    Portfolio_Assets = TradeData @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var Optimization"])

    return MVO_result

def model_train_predict(selected_shares, capital, start_date, end_date, selected_model, selected_params):
    time.sleep(1)
    if len(selected_shares) == 1:
        selected_shares = f"('{selected_shares[0]}')"
    else:
        selected_shares = tuple(selected_shares)

    with st.expander(label="Этапы обучения агента", expanded=True):

        train_data = extract_train_data(connection(), selected_shares, start_date, table_name, threshold_date)
        st.write(f"""✅ Торговые данные выгружены из базы данных c {format_date(min(train_data['time']), format='d MMMM yyyy', locale='ru')} года.""")
        full_train_data = fill_data(train_data)
        st.write(f"""✅ Добавлены данные о торгах выходного дня, кол-во выходных: {train_data['time'].nunique() // 7}.""")
        full_train_data.rename(columns={'time': 'date', 'ticker': 'tic'}, inplace=True)
        st.write(f"""✅ Собраны данные для обучения, имеющие размерность: {len(full_train_data)} x {len(full_train_data.columns)}.""")
        fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list = INDICATORS,
                        use_turbulence=True,
                        user_defined_feature = False)
        processed_train = fe.preprocess_data(full_train_data)
        processed_train.index= processed_train["date"].factorize()[0]
        st.write(f"""✅ Добавлены торговые индикаторы в качестве дополнительных параметров, всего добавлено {len(INDICATORS)} индикаторов.""")
        env_train = create_train_env(processed_train, capital)
        st.write(f"""✅ Подготовлено тренировочное окружение.""")
        selected_model = selected_model.split()[0].lower()
        trained_model = train_model(env_train, selected_model, selected_params)
        st.write(f"""✅ Агент завершил обучение.""")
    # ---------------------------------------------------------------------------------------------------
        trade_data = extract_trade_data(connection(), selected_shares, start_date, end_date, table_name)
        full_trade_data = fill_data(trade_data)
        full_trade_data.rename(columns={'time': 'date', 'ticker': 'tic'}, inplace=True)
        fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list = INDICATORS,
                        use_turbulence=False,
                        user_defined_feature = False)
        processed_trade = fe.preprocess_data(full_trade_data)
        processed_trade['turbulence'] = 0
        processed_trade.index= processed_trade["date"].factorize()[0]
        st.write(f"""✅ Подготовлены данные для торговли агентом.""")
        env_trade = create_trade_env(processed_trade, capital)
        st.write(f"""✅ Подготовлено торговое окружение.""")
        df_account_value, df_actions = DRLAgent.DRL_prediction(
                                                model=trained_model,
                                                environment=env_trade)
        st.success(f"""Агент окончательно определился с торговой стратегией.""")

    mvo = mvo_strategy(processed_train, processed_trade, capital)
    cp1 = capital
    cp2 = round(df_account_value['account_value'].tolist()[-1])
    cp3 = round(mvo['Mean Var Optimization'].tolist()[-1])

    if cp2 >= cp1 and cp2 >= cp3:
        st.markdown(f"### :green[{cp2}₽  ↑ {cp2-cp1}₽  (+{round(100*(cp2-cp3)/cp3, 1)}%)]",
                    help='Первое значение - баланс (руб.) в результате торговой стратегии, выбранной агентом. Второе значение - на сколько в рублях стратегия агента отличается от выбранного первоначального капитала. Третье значение - относительная разница эффективности стратегии агента от стратегии MVO.')
    elif cp2 >= cp1 and cp2 < cp3:
        st.markdown(f"### :green[{cp2}₽   ↑ {cp2-cp1}₽]  :red[({round(100*(cp2-cp3)/cp3, 1)}%)]",
                    help='Первое значение - баланс (руб.) в результате торговой стратегии, выбранной агентом. Второе значение - на сколько в рублях стратегия агента отличается от выбранного первоначального капитала. Третье значение - относительная разница эффективности стратегии агента от стратегии MVO.')
    elif cp2 < cp1 and cp2 >= cp3:
            st.markdown(f"### :red[{cp2}₽   ↓ {cp2-cp1}₽]  :green[(+{round(100*(cp2-cp3)/cp3, 1)}%)]",
                        help='Первое значение - баланс (руб.) в результате торговой стратегии, выбранной агентом. Второе значение - на сколько в рублях стратегия агента отличается от выбранного первоначального капитала. Третье значение - относительная разница эффективности стратегии агента от стратегии MVO.')
    elif cp2 < cp1 and cp2 < cp3:
        st.markdown(f"### :red[{cp2}₽   ↓ {cp2-cp1}₽  ({round(100*(cp2-cp3)/cp3, 1)}%)]",
                    help='Первое значение - баланс (руб.) в результате торговой стратегии, выбранной агентом. Второе значение - на сколько в рублях стратегия агента отличается от выбранного первоначального капитала. Третье значение - относительная разница эффективности стратегии агента от стратегии MVO.')
    # st.markdown(f"## {round(df_account_value['account_value'].tolist()[-1])}")
    # st.write('баланс (руб.) в результате торговой стратегии, выбранной агентом.')

    # st.write(mvo_strategy(processed_train, processed_trade, capital))
    # st.write(df_account_value)

    st.info('Mean-Variance Optimization (MVO) - это одна из самых мощных и широко используемых методик в современной инвестиционной теории, лежащая в основе эффективного управления портфелем. Она позволяет инвесторам оптимально распределять активы, чтобы максимизировать ожидаемую доходность при заданном уровне риска или минимизировать риск при заданной доходности.')

    st.plotly_chart(plot_compare_chart(selected_model,
                                       df_account_value,
                                       mvo),
                                       on_select="rerun")
    if cp2 >= cp3:
        st.write(f'Торговая стратегия агента {selected_model.upper()} окалась более эффективной по сравнению со стратегией MVO на {cp2-cp3} руб. ({round(100*(cp2-cp3)/cp3, 1)}%)')
    elif cp2 <= cp3:
            st.write(f'Торговая стратегия агента {selected_model.upper()} окалась менее эффективной по сравнению со стратегией MVO на {cp3-cp2} руб. ({round(100*(cp3-cp2)/cp3, 1)}%)')



    # return trained_model
