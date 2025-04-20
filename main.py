import streamlit as st
import pandas as pd
from scripts.actual_shares_list import shares_list


st.set_page_config(
page_title="RL Trade Agent",
page_icon="📈",
layout="centered"
)

# Пример подробной информации по акциям (можно расширить)
stock_info = {
    ticker: f"Подробная информация об акции {name} ({ticker}). Здесь можно разместить описание, финансовые показатели и т.п."
    for ticker, name in shares_list.items()
}

# Инициализация состояния
if "page" not in st.session_state:
    st.session_state.page = "main"
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = []
if "current_stock" not in st.session_state:
    st.session_state.current_stock = None

def main_page():
    st.title("Выбор акций")

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

    if st.session_state.selected_stocks:
        st.markdown("### Выбранные акции:")
        for ticker in st.session_state.selected_stocks:
            name = shares_list[ticker]
            # Кнопка для перехода на страницу акции
            if st.button(f"{name} ({ticker})", key=ticker):
                st.session_state.current_stock = ticker
                st.session_state.page = "detail"
                st.rerun()
    else:
        st.info("Пожалуйста, выберите хотя бы одну акцию.")

def detail_page():
    ticker = st.session_state.current_stock
    if ticker not in shares_list:
        st.error("Информация об акции недоступна.")
        if st.button("← Назад"):
            st.session_state.page = "main"
            st.rerun()
        return

    st.title(f"Информация об акции {shares_list[ticker]} ({ticker})")
    st.write(stock_info.get(ticker, "Информация отсутствует."))

    if st.button("← Назад"):
        st.session_state.page = "main"
        st.rerun()

# Навигация по страницам
if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "detail":
    detail_page()