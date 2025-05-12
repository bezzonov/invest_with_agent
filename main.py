import streamlit as st
import pandas as pd
import time

from datetime import datetime, timedelta
from scripts.actual_shares_list import shares_list
from scripts.config import stock_info, model_info, model_params
from scripts.shares_fig import plot_price_chart
from scripts.connection import connection
from scripts.metrics import show_metrics, calc_metrics
from scripts.data_filling import fill_data
from scripts.model_train_pred import model_train_predict

st.set_page_config(
page_title="RL Trade Agent",
page_icon="📈",
layout="centered"
)


# Инициализация состояния
if "page" not in st.session_state:
    st.session_state.page = "main"
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = []
if "current_stock" not in st.session_state:
    st.session_state.current_stock = None
if "training_result" not in st.session_state:
    st.session_state.training_result = None

def main_page():
    st.title("📈| RL Trade Agent")
    st.markdown("#### Параметры торговли")

    stocks = None
    capital = None
    start_date, end_date = None, None
    selected_model = None
    selected_params = None

    if "selected_stocks" not in st.session_state:
        st.session_state.selected_stocks = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "training_result" not in st.session_state:
        st.session_state.training_result = None

    # Используем временную переменную для выбора в форме
    with st.form("stock_form"):
        selected = st.multiselect(
            "Акции",
            options=list(shares_list.keys()),
            format_func=lambda x: f"{shares_list[x]} ({x})",
            default=st.session_state.selected_stocks,
            placeholder ='Выберите акции'
        )

        capital = st.number_input("Капитал для торговли (руб.)", min_value=10000, max_value=5_000_000, step=10000, value=100_000)
        col1, col2 = st.columns(2)

        min_start_date = datetime.today() - timedelta(days=365*10)
        max_start_date = datetime.today() - timedelta(days=7)
        min_end_date = datetime.today() - timedelta(days=365*10) + timedelta(days=7)
        max_end_date = datetime.today()

        with col1:
            start_date = st.date_input("Дата начала",
                                       value=datetime.today() - timedelta(days=7),
                                       min_value=min_start_date.date(),
                                       max_value=max_start_date.date())
        with col2:
            end_date = st.date_input("Дата окончания",
                                     value=datetime.today(),
                                    min_value=min_end_date.date(),
                                    max_value=max_end_date.date())

        selected_model = st.selectbox(
            "Модель",
            options=list(model_info.keys()),
            index=list(model_info.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in model_info else 0
        )
        st.session_state.selected_model = selected_model

        submitted = st.form_submit_button("Подтвердить выбор")

# -----------------------------------------------------------------------------------------------------------------------------------

    if submitted:
        if not selected:
            st.error("Пожалуйста, выберите хотя бы одну акцию.")
        elif not selected_model:
            st.error("Пожалуйста, выберите модель.")
        elif start_date < min_start_date.date() or start_date > max_start_date.date():
            st.error(f"Дата начала должна быть в диапазоне с {min_start_date.strftime('%d.%m.%Y')} по {max_start_date.strftime('%d.%m.%Y')}")
        elif end_date < min_end_date.date() or end_date > max_end_date.date():
            st.error(f"Дата окончания должна быть в диапазоне с {min_end_date.strftime('%d.%m.%Y')} по {max_end_date.strftime('%d.%m.%Y')}")
        elif start_date > end_date:
            st.error("Дата начала не может быть позже даты окончания.")
        elif (end_date - start_date).days < 7:
            st.error("Выберите более длительный период для торговли (от 7 дней)")
        else:
            st.session_state.selected_stocks = selected
            st.session_state.capital = capital
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
            st.rerun()

    if st.session_state.selected_stocks:
        st.success("Параметры торговли сохранены!")
        stocks = st.session_state.selected_stocks
        for i in range(0, len(stocks), 2):
            cols = st.columns(2)
            ticker = stocks[i]
            name = shares_list[ticker]
            if cols[0].button(f"{name} ({ticker})", key=ticker):
                st.session_state.current_stock = ticker
                st.session_state.page = "detail"
                st.rerun()

            if i + 1 < len(stocks):
                ticker = stocks[i + 1]
                name = shares_list[ticker]
                if cols[1].button(f"{name} ({ticker})", key=ticker + "_right"):
                    st.session_state.current_stock = ticker
                    st.session_state.page = "detail"
                    st.rerun()

        show_metrics({'Капитал': f"{capital} руб.",
                      'Дата начала': start_date,
                      'Дата окончания': end_date,
                      })

        if st.button(f"{selected_model}"):
            st.session_state.page = "model_detail"
            st.rerun()
    else:
        st.info("Выберите параметры торговли.")

# ------------------------------------------------------
    st.markdown("#### Параметры модели")
    params = model_params[selected_model]

    for key, param in params.items():
        if key not in st.session_state:
            st.session_state[key] = param.get("value", param.get("options", [None])[0])

    with st.form("model_params_form"):
        cols = st.columns(2)
        col_index = 0

        for key, param in params.items():
            with cols[col_index]:
                if param["type"] == "number_input":
                    st.number_input(
                        param["label"],
                        min_value=param["min_value"],
                        max_value=param["max_value"],
                        step=param["step"],
                        format=param.get("format"),
                        help=param['help'],
                        key=key
                    )
                elif param["type"] == "slider":
                    st.slider(
                        param["label"],
                        min_value=param["min_value"],
                        max_value=param["max_value"],
                        step=param["step"],
                        format=param.get("format"),
                        help=param['help'],
                        key=key
                    )
                elif param["type"] == "selectbox":
                    st.selectbox(
                        param["label"],
                        options=param["options"],
                        help=param['help'],
                        key=key
                    )
            col_index = (col_index + 1) % 2

        submitted_params = st.form_submit_button("Подтвердить выбор")

    if submitted_params:
        selected_params = {key: st.session_state[key] for key in params.keys()}
        st.session_state.selected_params = selected_params
        st.success("Параметры модели сохранены!")
        df_params = pd.DataFrame.from_dict(selected_params, orient='index', columns=['Значение'])
        df_params.index.name = 'Параметр'
        st.dataframe(df_params, use_container_width=True)
    else:
        if "selected_params" in st.session_state:
            st.success("Параметры модели сохранены!")
            df_params = pd.DataFrame.from_dict(st.session_state.selected_params, orient='index', columns=['Значение'])
            df_params.index.name = 'Параметр'
            st.dataframe(df_params, use_container_width=True)
        else:
            st.info("Выберите параметры модели.")

    st.markdown("#### Торговля с агентом")

    # # Заглушка для stocks, замените на вашу логику выбора акций
    # if "stocks" not in st.session_state:
    #     st.session_state.stocks = None  # Или список выбранных акций

    # stocks = st.session_state.stocks

    if st.button("🚀 Начать обучение", key="big_train_button"):
        if stocks is None:
            st.error('Пожалуйста, выберите хотя бы одну акцию.')
        elif "selected_params" not in st.session_state:
            st.error('Пожалуйста, подтвердите выбор параметров модели.')
        else:
            with st.spinner(f"Агент выбирает лучшую торговую стратегию, процесс запущен в {datetime.now().time().strftime('%H:%M:%S')}."):
                st.info('В среднем агенту требуется около 7 минут на подбор лучшей торговой стратегии, однако время ожидания может увеличиться в зависимости от выбранных параметров.')
                try:
                    result = model_train_predict(stocks, capital, start_date, end_date, selected_model, st.session_state.selected_params)
                    st.session_state.training_result = result
                except MemoryError:
                    st.warnings('Что-то пошло не так, попробуйте снова.')

    # if "selected_params" in st.session_state:
    #     st.markdown("### Текущие параметры:")
    #     st.json(st.session_state.selected_params)




# ------------------------------------------------------------------------

def model_detail_page():
    st.markdown(f"### {st.session_state.selected_model}")
    st.text_area("О модели", model_info.get(st.session_state.selected_model, "Описание отсутствует."), height=400)

    if st.button("Назад"):
        st.session_state.page = "main"
        st.rerun()

def detail_page():
    ticker = st.session_state.current_stock
    st.markdown(f"### {shares_list[ticker]} ({ticker})")
    info = stock_info.get(ticker, "Информация отсутствует.")
    st.text_area("О компании", info[0], height=150)
    st.plotly_chart(plot_price_chart(connection(),
                                             ticker,
                                             'hour_shares_data',
                                             datetime.strftime(datetime.today() - timedelta(days=60),'%Y-%m-%d')),
                                             on_select="rerun"
                                               )
    show_metrics(calc_metrics(connection(),
                            ticker,
                            'hour_shares_data',
                             datetime.strftime(datetime.today() - timedelta(days=400),'%Y-%m-%d')
                            ))


    if st.button("← Назад"):
        st.session_state.page = "main"
        st.rerun()

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "detail":
    detail_page()
elif st.session_state.page == "model_detail":
    model_detail_page()