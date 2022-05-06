from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from SVM_Parameters import SVM_Parameters
from random import randint

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colorscale = [[0, "#003366"], [0.5, "rgba(203, 203, 212, .1)"], [1, "#663300"]]

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    dcc.Store(id="seed", storage_type="memory", data="42"),
    html.Header(id="Header", children=[
        html.H1("Teste como o conjunto de dados ou os parâmetros podem influenciar o modelo de aprendizagem de máquina"),
    ]),
    html.Div(id="models", children=[
        html.Button("Nearest Neighbors", id="bt-knn", className="basic-button"),
        html.Button("Support Vector Machine", id="bt-svm", className="basic-button"),
        html.Button("Neural Network", id="bt-nn", className="basic-button", style={"margin-right": "30px"})
    ], className="container-flex"),
    html.Div(id="main", children=[
        html.Div(children=[
            html.H4("Dados"),
            html.P("Tipo"),
            dcc.Dropdown(
                id="data-type-dropdown",
                options=[
                    {"label": key.upper(), "value": key}
                    for key in ["blobs", "moons", "circles"]
                ],
                value="blobs",
                clearable=False,
                className="dropdown",
            ),
            html.P("Quantidade de pontos"),
            dcc.Input(value=100, id="n-input"),
            html.P("Ruído dos dados"),
            dcc.Input(value=0.1, id="std-input"),
            #html.P("% dos pontos pertencentes à classe amarela"),
            #dcc.Slider(0, 100, value=50, marks=None, tooltip={"placement": "bottom", "always_visible": True}, className="sliders", id="balance-slider"),
            html.Button("Gerar dados", id="bt-generate-data", className="basic-button", style={"margin-right": "0px", "margin-left": "0px"}),
        ], className="column", style={"width": "15%", "position": "relative"}),
        html.Div(children=[
            html.H4("Parâmetros"),
            html.P("Função Kernel"),
            dcc.Dropdown(
                id="kernel-dropdown",
                options=[
                    {"label": key.upper(), "value": key}
                    for key in ["linear", "poly", "rbf", "sigmoid"]
                ],
                value="linear",
                clearable=False,
                className="dropdown",
            ),
            html.P("Parâmetro de regularização (C)"),
            dcc.Input(value=1.0, id="c-input"),
            html.Button("Treinar modelo", id="bt-fit", className="basic-button", style={"margin-right": "0px", "margin-left": "0px", "margin-top": "12px"})
        ], className="column", style={"width": "15%", "position": "relative"}),
        html.Div(children=[
            html.H4("Modelo"),
            #html.P("Acurácia no treino:"),
            html.P("Acurácia do modelo:", id="model-accuracy-display"),
            dcc.Graph(id="scatter", config={"displayModeBar": False}),
        ], className="column", style={"width": "65%", "position": "relative"}),
        ], className="main-div"),
])

buttons = ["bt-knn", "bt-svm", "bt-nn"]
@app.callback(
    [
        Output("bt-knn", "className"),
        Output("bt-svm", "className"),
        Output("bt-nn", "className"),
    ],
    [
        Input("bt-knn", "n_clicks"),
        Input("bt-svm", "n_clicks"),
        Input("bt-nn", "n_clicks"),
    ],
)
def set_active(*args):
    ctx = callback_context
    
    if not ctx.triggered or not any(args):
        return [
            "basic-button" if x != 1 else "basic-button selected" for x in range(3)
        ]

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    styles = []
    for button in buttons:
        if button == button_id:
            styles.append("basic-button selected")
        else:
            styles.append("basic-button")

    return styles

@app.callback(
    [
        Output("scatter", "figure"),
        Output("model-accuracy-display", "children"),
        Output("seed", "data")
    ],
    [
        Input("bt-generate-data", "n_clicks"),
        Input("bt-fit", "n_clicks"),
        Input("seed", "data")
    ],
    [
        State("n-input", "value"),
        State("std-input", "value"),
        #State("balance-slider", "value"),
        State("kernel-dropdown", "value"),
        State("c-input", "value"),
        State("data-type-dropdown", "value")
    ]
)
def generate_fit_data(bt_generate_data, bt_fit_data, seed, n, std, kernel, c, data_type):
    # Obter o id do botão selecionado
    ctx = callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "bt-generate-data": seed = randint(0, 1000)
    # Gerar o dataset
    if data_type == "blobs": X, y = make_blobs(n_samples=int(n), centers=2, n_features=2, cluster_std=float(std)*10, random_state=int(seed))
    elif data_type == "circles": X, y = make_circles(n_samples=int(n), noise=float(std), random_state=int(seed))
    elif data_type == "moons": X, y = make_moons(n_samples=int(n), noise=float(std), random_state=int(seed))
    # Gerar o gráfico de saída
    fig = go.Figure()
    fig.update_layout(showlegend=False, margin=dict(l=16, r=16, t=16, b=16), xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False), plot_bgcolor="rgba(203, 203, 212, .3)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="DIN Alternate",
        xaxis_range=[X[:, 0].min() - 1, X[:, 0].max() + 1],
        yaxis_range=[X[:, 1].min() - 1, X[:, 1].max() + 1])
    if button_id == "bt-fit": # Clique em bt-fit
        model = SVC(kernel=kernel, C=c)
        model.fit(X, y)
        fig.add_trace(plot_svc_decision_function((min(X[:, 0]), max(X[:, 0])), (min(X[:, 1]), max(X[:, 1])), model))
    # Adicionar os pontos
    fig.add_trace(
        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=y, colorscale=colorscale), showlegend=False, name="")
    )
    if button_id == "bt-fit": return fig, "Acurácia do modelo: {0:.2f}".format(accuracy_score(y, model.predict(X))), no_update
    return fig, "Clique em treinar modelo para ver o desempenho dele", str(seed)
    
def plot_svc_decision_function(xlim, ylim, model, margin=1):
    # create grid to evaluate model
    x = np.linspace(xlim[0] - margin, xlim[1] + margin, 100)
    y = np.linspace(ylim[0] - margin, ylim[1] + margin, 100)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.decision_function(xy).reshape(xx.shape)
    # plot decision boundary and margins
    return go.Contour(x=x, y=y, z=Z, colorscale=colorscale,
            contours_coloring="heatmap", hoverinfo="none", showlegend=False, showscale=False
            #line_width=2,
        )
    #plt.contour(X, Y, P,  colors=color,
    #           levels=[-1, 0, 1], alpha=0.5,
    #           linestyles=['--', '-', '--'])
               
if __name__ == '__main__':
    app.run_server(debug=True)
