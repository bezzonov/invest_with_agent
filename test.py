import streamlit as st
import pandas as pd
import plotly.express as px

def load_data():
    data = {
        'Сектор': ['Технологии', 'Технологии', 'Финансы', 'Финансы', 'Энергетика', 'Энергетика'],
        'Акция': ['Apple', 'Microsoft', 'Sberbank', 'VTB', 'Gazprom', 'Lukoil'],
        'Куплено, шт.': [100, 150, 200, 180, 120, 130],
        'Профит, руб.': [50000, 60000, 30000, 20000, 40000, 35000]
    }
    df = pd.DataFrame(data)
    return df

def create_treemap(df):

    fig = px.treemap(
        df,
        path=[px.Constant("Все секторы экономики"), 'Сектор', 'Акция'],
        values='Куплено, шт.',  # Use raw values for sizing
        color='Профит, руб.',
        color_continuous_scale='RdYlGn',
        hover_data={'Куплено, шт.': True, 'Профит, руб.': True},
        title='Дерево акций по секторам с отображением прибыли и количества купленных акций'
    )

    fig.update_traces(
        root_color="white",
        marker_line_width=0,
        tiling=dict(pad=0),
        hovertemplate='<b>%{label}</b><br>Куплено: %{value}<br>Профит: %{color:.0f} руб.<extra></extra>'
    )

    fig.update_layout(
        height=450,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=40, l=5, r=24, b=5),
        font=dict(size=14, family="Arial, sans-serif", color="#000000"),
        coloraxis_colorbar=dict(
            title="Профит, руб.",
            thickness=15,
            len=0.9,
            x=0.98,
            y=0.5
        )
    )

    return fig

def main():
    st.title("Treemap of Stocks by Sector")

    df = load_data()
    st.dataframe(df)

    fig = create_treemap(df)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
