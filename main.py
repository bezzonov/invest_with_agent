import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from scripts.actual_shares_list import shares_list
from scripts.config import stock_info
from scripts.shares_fig import plot_price_chart
from scripts.connection import connection
from scripts.metrics import show_metrics, calc_metrics
from scripts.data_filling import fill_data

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

def main_page():
    st.markdown("### Параметры торговли")

    if "selected_stocks" not in st.session_state:
        st.session_state.selected_stocks = []

    # Используем временную переменную для выбора в форме
    with st.form("stock_form"):
        selected = st.multiselect(
            "Выберите акции",
            options=list(shares_list.keys()),
            format_func=lambda x: f"{shares_list[x]} ({x})",
            default=st.session_state.selected_stocks
        )

        capital = st.number_input("Капитал для торговли (руб.)", min_value=10000, max_value=5_000_000, step=10000, value=10_000)
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

        submitted = st.form_submit_button("Подтвердить выбор")

        if submitted:
            if not selected:
                st.error("Пожалуйста, выберите хотя бы одну акцию.")
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
        st.success("Выбранные параметры торговли:")
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
    else:
        st.info("Выберите акции, капитал и период торговли.")



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
                             datetime.strftime(datetime.today() - timedelta(days=365),'%Y-%m-%d')
                            ))


    if st.button("← Назад"):
        st.session_state.page = "main"
        st.rerun()

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "detail":
    detail_page()