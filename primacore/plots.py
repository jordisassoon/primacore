import plotly.graph_objects as go


def scatter_predictions(df, x_col, y_col, title="Predictions (scatter)"):
    fig = go.Figure(go.Scatter(x=df[x_col], y=df[y_col], mode="markers"))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
    fig.show()
    return fig


def line_predictions(df, x_col, y_col, title="Predictions (line)"):
    fig = go.Figure(go.Scatter(x=df[x_col], y=df[y_col], mode="lines"))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
    fig.show()
    return fig
