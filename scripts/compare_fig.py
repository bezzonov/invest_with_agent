import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from scripts.connection import connection


def plot_compare_chart(selected_model, df1, df2):


    indexes_info = pd.read_sql_query("select * from stock_market_indexes", connection())
    indexes_info = indexes_info[(indexes_info['date'] >= min(df1['date'])) &
                                (indexes_info['date'] <= max(df1['date']))]
    start_index_row = indexes_info[indexes_info['date'] == min(df1['date'])]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df1['date'], y=[round(i) for i in df1['account_value']], name=f'{selected_model.upper()}', visible=True, line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df1['date'], y=[round(i) for i in df2['Mean Var Optimization']], name='MVO', visible=True, line=dict(color='blue'),
                                 fill='tonexty',
                                 fillcolor='rgba(61, 237, 77, 0.15)'))
    fig.add_trace(go.Scatter(x=indexes_info['date'], y=[round(100 * (i - df1['account_value'].values[0]) / df1['account_value'].values[0], 1) for i in df1['account_value']], name=f'{selected_model.upper()}', visible=False, line=dict(color='red')))
    fig.add_trace(go.Scatter(x=indexes_info['date'], y=[round(100 * (i - start_index_row['imoex'].values[0]) / start_index_row['imoex'].values[0], 1)  for i in indexes_info['imoex']], name='IMOEX', visible=False, line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=indexes_info['date'], y=[round(100 * (i - start_index_row['rtsi'].values[0]) / start_index_row['rtsi'].values[0], 1) for i in indexes_info['rtsi']], name='RTSI', visible=False, line=dict(color='green')))
    fig.add_trace(go.Scatter(x=indexes_info['date'], y=[round(100 * (i - start_index_row['moexbc'].values[0]) / start_index_row['moexbc'].values[0], 1) for i in indexes_info['moexbc']], name='MOEXBC', visible=False, line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=indexes_info['date'], y=[round(100 * (i - start_index_row['moexog'].values[0]) / start_index_row['moexog'].values[0], 1) for i in indexes_info['moexog']], name='MOEXOG', visible=False, line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=indexes_info['date'], y=[round(100 * (i - start_index_row['moexeu'].values[0]) / start_index_row['moexeu'].values[0], 1) for i in indexes_info['moexeu']], name='MOEXEU', visible=False, line=dict(color='magenta')))
    fig.add_trace(go.Scatter(x=indexes_info['date'], y=[round(100 * (i - start_index_row['moexfn'].values[0]) / start_index_row['moexfn'].values[0], 1) for i in indexes_info['moexfn']], name='MOEXFN', visible=False, line=dict(color='cyan')))



    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        label="Стратегии",
                        method="update",
                        args=[
                            {"visible": [True, True, False, False, False, False, False, False, False]},
                            {"yaxis": {"title": "Баланс портфеля (руб.)"}}
                        ],
                    ),
                    dict(
                        label="Индикаторы",
                        method="update",
                        args=[
                            {"visible": [False, False, True, True, True, True, True, True, True]},
                            {"yaxis": {"title": "%"}}
                        ],
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.1,
                xanchor="right",
                y=1.15,
                yanchor="top"
            ),
        ],
        title="Сравнительный анализ торговых стратегий и индикаторов",
        xaxis_title="Дата",
        yaxis_title="Баланс портфеля (руб.)",
        margin=dict(t=85),
        xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1)
    )

    return fig
