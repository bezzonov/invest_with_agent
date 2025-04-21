import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from scripts.actual_shares_list import shares_list
from scripts.config import stock_info
from scripts.shares_fig import plot_price_chart
from scripts.connection import connection
from scripts.metrics import show_metrics, calc_metrics

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
    st.title("Выбор акций")

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
        submitted = st.form_submit_button("Подтвердить выбор")

        if submitted:
            st.session_state.selected_stocks = selected
            st.rerun()

    if st.session_state.selected_stocks:
        st.markdown("### Выбранные акции:")

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
    else:
        st.info("Пожалуйста, выберите хотя бы одну акцию.")

    st.markdown("##### Выбор агента в процессе")


def detail_page():
    ticker = st.session_state.current_stock
    st.title(f"{shares_list[ticker]} ({ticker})")
    info = stock_info.get(ticker, "Информация отсутствует.")
    st.text_area("О компании", info[0], height=200)
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