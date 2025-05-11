import streamlit as st
import plotly.graph_objects as go
import numpy as np

def plot_compare_chart(selected_model, df1, df2):
    x = np.linspace(0, 10, 100)
    y3 = np.sin(x) * 0.5
    y4 = np.cos(x) * 0.5 + 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df1['date'], y=df1['account_value'], name=f'{selected_model.upper()}', visible=True, line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df1['date'], y=df2['Mean Var Optimization'], name='MVO', visible=True, line=dict(color='blue'),
                                 fill='tonexty',
                                 fillcolor='rgba(0, 255, 0, 0.3)'))
    fig.add_trace(go.Scatter(x=x, y=y3, name='Линия 3', visible=False, line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x, y=y4, name='Линия 4', visible=False, line=dict(color='orange')))


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
                            {"visible": [True, True, False, False]},
                            {"yaxis": {"title": "Баланс портфеля (руб.)"}}
                        ],
                    ),
                    dict(
                        label="Индикаторы",
                        method="update",
                        args=[
                            {"visible": [False, False, True, True]},
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
        yaxis_title="Баланс портфеля (руб.) / %",
        margin=dict(t=85),
        xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1)
    )

    return fig
