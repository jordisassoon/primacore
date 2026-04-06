import plotly.graph_objects as go


custom_theme = dict(
    template="plotly_dark",  # Dark background theme
    font=dict(family="Arial", size=14, color="white"),
    title_font=dict(family="Arial", size=18, color="cyan"),
    legend=dict(font=dict(size=12, color="white")),
    margin=dict(l=60, r=60, t=80, b=60),
)


def scatter_predictions(df, x_col, y_col, title="Predictions (scatter)"):
    fig = go.Figure(go.Scatter(x=df[x_col], y=df[y_col], mode="markers"))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col, **custom_theme)
    fig.show()
    return fig


def line_predictions(df, x_col, y_col, title="Predictions (line)"):
    fig = go.Figure(go.Scatter(x=df[x_col], y=df[y_col], mode="lines"))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col, **custom_theme)
    fig.show()
    return fig


def spider_plot(df, title="Spider Plot"):
    fig = go.Figure(
        go.Scatterpolar(
            r=df.values,
            theta=df.index,
            fill="toself",
        )
    )
    fig.update_layout(
        title=title, polar=dict(radialaxis=dict(visible=True)), **custom_theme
    )
    fig.show()
    return fig
