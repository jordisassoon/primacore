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


def plot_neighbors(
    sources: list[tuple[float, float]],
    neighbors: list[tuple[float, float]],
    distances: list[float],
    leftovers: list[tuple[float, float]],
    title="Neighbor Distances",
):
    fig = go.Figure()

    # Scatter plot with sources and neighbors connected, and leftovers also plotted in
    for source, neighbor, distance in zip(sources, neighbors, distances):
        fig.add_trace(
            go.Scatter(
                x=[source[0], neighbor[0]],
                y=[source[1], neighbor[1]],
                mode="lines+markers",
                line=dict(color="cyan", width=2),
                marker=dict(size=8, color="magenta"),
                name=f"Distance: {distance:.2f}",
            )
        )

    if leftovers:
        fig.add_trace(
            go.Scatter(
                x=[o[0] for o in leftovers],
                y=[o[1] for o in leftovers],
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Leftovers",
            )
        )

    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y", **custom_theme)
    fig.show()
    return fig
