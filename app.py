from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context
from dash_daq import BooleanSwitch

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from SVM_Parameters import SVM_Parameters
from SVM_Batch import train_SVM_batch

from random import randint

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colorscale = [[0, "#003366"], [0.5, "rgba(203, 203, 212, .1)"], [1, "#663300"]]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "ML Playgrounds"

server = app.server

app.layout = html.Div([
    # Componentes de memória
    dcc.Store(id="seed-memory", storage_type="memory", data="42"),
    dcc.Store(id="data-type-memory", storage_type="memory", data="moons"),
    dcc.Store(id="n-memory", storage_type="memory", data="100"),
    dcc.Store(id="std-memory", storage_type="memory", data="0.1"),
    dcc.Store(id="bt-memory", storage_type="memory", data="bt-svm"),
    dcc.Store(id="step-memory", storage_type="memory", data="0"),
    # Cabeçalho
    html.Header(id="Header", children=[
        html.H1("Teste como o conjunto de dados ou os parâmetros podem influenciar o modelo de aprendizagem de máquina"),
    ]),
    # Seletor de algoritmo
    html.Div(id="models", children=[
        html.Button("Nearest Neighbors", id="bt-knn", className="basic-button"),
        html.Button("Support Vector Machine", id="bt-svm", className="basic-button"),
        html.Button("Decision Tree", id="bt-dtc",
            className="basic-button", style={"margin-right": "30px"})
    ], className="container-flex"),
    # Corpo do app
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
                value="moons",
                clearable=False,
                className="dropdown",
            ),
            html.P("Quantidade de pontos"),
            dcc.Input(value=100, id="n-input", className="input"),
            html.P("Ruído dos dados", id="std-info"),
            dcc.Input(value=0.1, id="std-input", className="input"),
            #html.P("% dos pontos pertencentes à classe amarela"),
            #dcc.Slider(0, 100, value=50, marks=None, tooltip={"placement": "bottom", "always_visible": True}, className="sliders", id="balance-slider"),
            html.Button("Gerar dados", id="bt-generate-data", className="basic-button", style={"margin-right": "0px", "margin-left": "0px"}),
        ], className="column", style={"width": "15%", "position": "relative"}),
        # Div de parâmetros do SVM
        html.Div(children=[
            html.H4("Parâmetros"),
            html.Div(children=[
                html.P("Batch SVM"),
                BooleanSwitch(disabled=True, color="#003366", id="batch-switch")
                ], style={
                    "display": "flex",
                    "justify-content": "space-between",
                    "align-items": "center"}),
            html.P("Tamanho de S (% do dataset)", id="s-info", style={"display": "none"}),
            dcc.Input(value=0.1, id="s-input", className="input", style={"display": "none"}),
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
            dcc.Input(value=1.0, id="c-input", className="input"),
            html.P("Grau (apenas para kernel poly)", id="degree-info"),
            dcc.Input(value=3, id="degree-input", className="input"),
            html.P("Gama (apenas para kernel rbf, poly, ou sigmoid)", id="gamma-info"),
            html.P("Use scale, auto, ou um float qualquer"),
            dcc.Input(value="scale", id="gamma-input", className="input"),
            html.Button("Treinar modelo", id="bt-fit-svm", className="basic-button", style={"margin-right": "0px", "margin-left": "0px", "margin-top": "12px"}),
            html.P("Treinar modelo", id="train-info", style={"display": "none"}),
            html.Div(children=[
                html.Button("↺", id="bt-fit-reset", className="basic-button on", style={"margin-left": "0px", "display": "none"}),
                html.Button("▶", id="bt-fit-step", className="basic-button on", style={"display": "none"}),
                html.Button("▶▶", id="bt-fit-skip", className="basic-button on", style={"display": "none"})
                ], style={
                    "display": "flex",
                    "justify-content": "space-between",
                    "align-items": "center"}),
                html.P("Passo: 0", id="step-display", style={"display": "none"}),
        ], id="parameters-svm-column", className="column", style={"width": "15%", "position": "relative"}),
        # Div de parâmetros do K-NN
        html.Div(children=[
            html.H4("Parâmetros"),
            html.P("Vizinhos"),
            dcc.Input(value=5, id="k-input", className="input"),
            html.P("Pesos"),
            dcc.Dropdown(
                id="weights-dropdown",
                options=[
                    {"label": key.upper(), "value": key}
                    for key in ["uniform", "distance"]
                ],
                value="uniform",
                clearable=False,
                className="dropdown",
            ),
            html.P("P"),
            html.P("P=1 equivale a distância de Manhattan, P=2 Euclidiana, e para outro P qualquer é usada Minkowski"),
            dcc.Input(value=2, id="p-input", className="input"),
            html.Button("Treinar modelo", id="bt-fit-knn", className="basic-button", style={"margin-right": "0px", "margin-left": "0px", "margin-top": "12px"})
        ], id="parameters-knn-column", className="column", style={"width": "15%", "position": "relative", "display": "none"}),
        # Div de parâmetros da árvore de decisão
        html.Div(children=[
            html.H4("Parâmetros"),
            html.P("Critério de divisão"),
            dcc.Dropdown(
                id="criterion-dropdown",
                options=[
                    {"label": key.upper(), "value": key}
                    for key in ["gini", "entropy"]
                ],
                value="gini",
                clearable=False,
                className="dropdown",
            ),
            html.P("Estratégia de divisão"),
            dcc.Dropdown(
                id="splitter-dropdown",
                options=[
                    {"label": key.upper(), "value": key}
                    for key in ["best", "random"]
                ],
                value="best",
                clearable=False,
                className="dropdown",
            ),
            html.P("Profundidade máxima da árvore"),
            html.P("Utilize None para ilimitado"),
            dcc.Input(value="None", id="depth-input", className="input"),
            html.P("Mínimo de amostras para se dividir um nó"),
            dcc.Input(value=2, id="samples-input", className="input"),
            html.P("Mínimo de amostras em um nó folha"),
            dcc.Input(value=1, id="leaf-input", className="input"),
            html.Button("Treinar modelo", id="bt-fit-dtc", className="basic-button", style={"margin-right": "0px", "margin-left": "0px", "margin-top": "12px"})
        ], id="parameters-dtc-column", className="column", style={"width": "15%", "position": "relative", "display": "none"}),
        html.Div(children=[
            html.H4("Modelo"),
            #html.P("Acurácia no treino:"),
            html.P(id="model-accuracy-display"),
            dcc.Graph(id="scatter", config={"displayModeBar": False}),
        ], className="column", style={"width": "65%", "position": "relative"}),
        ], className="main-div"),
])

buttons = ["bt-knn", "bt-svm", "bt-dtc"]
@app.callback(
    [
        Output("bt-knn", "className"),
        Output("bt-svm", "className"),
        Output("bt-dtc", "className"),
    ],
    [
        Input("bt-knn", "n_clicks"),
        Input("bt-svm", "n_clicks"),
        Input("bt-dtc", "n_clicks"),
    ],
)
def set_active(*args):
    ctx = callback_context
    
    if not ctx.triggered or not any(args): return ["basic-button" if x != 1 else "basic-button selected" for x in range(3)]

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    styles = []
    for button in buttons:
        if button == button_id: styles.append("basic-button selected")
        else: styles.append("basic-button")

    return styles

# Callback do clique no botão de gerar dados ou treinar modelo
@app.callback(
    [
        Output("scatter", "figure"),
        Output("model-accuracy-display", "children"),
        Output("seed-memory", "data"),
        Output("data-type-memory", "data"),
        Output("n-memory", "data"),
        Output("std-memory", "data"),
        Output("step-display", "children"),
        Output("step-memory", "data")
    ],
    [
        Input("bt-generate-data", "n_clicks"),
        Input("bt-fit-svm", "n_clicks"),
        Input("bt-fit-knn", "n_clicks"),
        Input("bt-fit-dtc", "n_clicks"),
        Input("bt-fit-step", "n_clicks"),
        Input("bt-fit-skip", "n_clicks"),
        Input("bt-fit-reset", "n_clicks"),
    ],
    [
        # Memória de parâmetros dos dados
        State("seed-memory", "data"),
        State("data-type-memory", "data"),
        State("n-memory", "data"),
        State("std-memory", "data"),
        # Parâmetros do SVM
        State("kernel-dropdown", "value"),
        State("c-input", "value"),
        State("gamma-input", "value"),
        State("degree-input", "value"),
        # Parâmetros do KNN
        State("k-input", "value"),
        State("weights-dropdown", "value"),
        State("p-input", "value"),
        # Parâmetros da Árvore de Decisão
        State("criterion-dropdown", "value"),
        State("splitter-dropdown", "value"),
        State("depth-input", "value"),
        State("samples-input", "value"),
        State("leaf-input", "value"),
        # Estado do input dos parâmetros dos dados
        State("n-input", "value"),
        State("std-input", "value"),
        State("data-type-dropdown", "value"),
        # Dados e parâmetros do batch SVM
        State("step-memory", "data"),
        State("s-input", "value")
    ]
)
def generate_fit_data(bt_generate, bt_fit_svm, bt_fit_knn, bt_fit_dtc, bt_fit_step, bt_fit_skip, bt_fit_reset, mem_seed, mem_data_type, mem_n, mem_std, kernel_input, c_input, gamma_input, degree_input, k_input, knn_weights, p_input, criterion_input, splitter_input, depth_input, split_input, leaf_input, n_input, std_input, data_type_input, mem_steps, s_input):
    seed = randint(0, 1000)
    n_steps = 0
    # Definir se o callback foi ativado por um botão
    ctx = callback_context
    # Obter o id do botão selecionado (ou não, caso não tenha sido um que disparou o callback)
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "bt-generate-data": X, y = get_dataset(data_type_input, n_input, std_input, seed)
    else: X, y = get_dataset(mem_data_type, mem_n, mem_std, mem_seed)
    # Gerar o gráfico de saída
    fig = generate_figure(X)
    # Gerar o modelo
    if button_id == "bt-fit-svm":
        if gamma_input != "auto" and gamma_input != "scale": gamma_input = float(gamma_input)
        model = SVC(kernel=kernel_input, C=float(c_input), gamma=gamma_input, degree=float(degree_input))
        model.fit(X, y)
        fig.add_trace(plot_svc_decision_function((min(X[:, 0]), max(X[:, 0])), (min(X[:, 1]), max(X[:, 1])), model))
    elif button_id == "bt-fit-knn":
        model =  KNeighborsClassifier(n_neighbors=int(k_input), weights=knn_weights, p=int(p_input))
        model.fit(X, y)
        fig.add_trace(plot_decision_function((min(X[:, 0]), max(X[:, 0])), (min(X[:, 1]), max(X[:, 1])), model))
    elif button_id == "bt-fit-dtc":
        if depth_input == "None": depth = None
        else: depth = int(depth_input)
        model =  DecisionTreeClassifier(criterion=criterion_input, splitter=splitter_input, max_depth=depth, min_samples_split=int(split_input), min_samples_leaf=int(leaf_input))
        model.fit(X, y)
        fig.add_trace(plot_decision_function((min(X[:, 0]), max(X[:, 0])), (min(X[:, 1]), max(X[:, 1])), model))
    elif button_id == "bt-fit-step":
        steps = int(mem_steps) + 1
        model, n_steps, S_X, U_X, S_y, U_y = train_SVM_batch(X, y, float(s_input), steps, kernel_input, c_input, gamma_input, degree_input)
        fig.add_trace(plot_svc_decision_function((min(X[:, 0]), max(X[:, 0])), (min(X[:, 1]), max(X[:, 1])), model))
    elif button_id == "bt-fit-skip":
        model, n_steps, S_X, U_X, S_y, U_y  = train_SVM_batch(X, y, float(s_input), float("inf"), kernel_input, c_input, gamma_input, degree_input)
        fig.add_trace(plot_svc_decision_function((min(X[:, 0]), max(X[:, 0])), (min(X[:, 1]), max(X[:, 1])), model))
    if button_id == "bt-fit-step" or button_id == "bt-fit-skip":
        for trace in plot_svm_batch_points(S_X, S_y, U_X, U_y): fig.add_trace(trace)
        return fig, "Acurácia do modelo: {0:.2f}%".format(accuracy_score([1 if yi == 1 else -1 for yi in y], model.predict(X)) * 100), no_update, no_update, no_update, no_update, "Passos: {}".format(n_steps), str(n_steps)
    # Adicionar os pontos do dataset ao gráfico
    fig.add_trace(
        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=y, colorscale=colorscale), showlegend=False, name="")
    )
    if button_id == "bt-fit-svm" or button_id == "bt-fit-knn" or button_id == "bt-fit-dtc": return fig, "Acurácia do modelo: {0:.2f}%".format(accuracy_score(y, model.predict(X)) * 100), no_update, no_update, no_update, no_update, no_update, "0"
    
    return fig, "Clique em treinar modelo para ver o desempenho dele", str(seed), str(data_type_input), int(n_input), float(std_input), "Passos: {}".format(n_steps), "0"
    
@app.callback(Output("std-info", "children"), Input("data-type-dropdown", "value"))
def update_std_info(data_type):
    if data_type == "blobs": return "Desvio padrão das classes"
    return "Ruído dos dados"
    
@app.callback([Output("gamma-info", "children"), Output("degree-info", "children")], Input("kernel-dropdown", "value"))
def update_svm_parameters_info(kernel_type):
    if kernel_type == "linear": return "Gama (não aplicável)", "Grau (não aplicável)"
    if kernel_type == "poly": return "Gama", "Grau"
    return "Gama", "Grau (não aplicável)"

@app.callback(
    [
        Output("parameters-svm-column", "style"),
        Output("parameters-knn-column", "style"),
        Output("parameters-dtc-column", "style"),
        Output("bt-memory", "data"),
    ],
    [
        Input("bt-knn", "n_clicks"),
        Input("bt-svm", "n_clicks"),
        Input("bt-dtc", "n_clicks"),
        Input("bt-memory", "data"),
    ],
)
def update_parameters(bt_knn, bt_svm, bt_nn, bt_memory):
    # Definir se o callback foi ativado por um botão
    ctx = callback_context
    # Caso entre no if abaixo é por ter sido o startup do aplicativo
    if not ctx.triggered or not any([bt_knn, bt_svm, bt_nn, bt_memory]): return {"width": "15%", "position": "relative"}, {"width": "15%", "position": "relative", "display": "none"}, {"width": "15%", "position": "relative", "display": "none"}, no_update
    # Obter o id do botão selecionado (ou não, caso não tenha sido um que disparou o callback)
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    # Foi clicado no mesmo botão
    if button_id == bt_memory: return no_update, no_update, no_update, no_update
    if button_id == "bt-svm": return {"width": "15%", "position": "relative"}, {"width": "15%", "position": "relative", "display": "none"}, {"width": "15%", "position": "relative", "display": "none"}, "bt-svm"
    if button_id == "bt-knn": return {"width": "15%", "position": "relative", "display": "none"}, {"width": "15%", "position": "relative"}, {"width": "15%", "position": "relative", "display": "none"}, "bt-knn"
    if button_id == "bt-dtc": return {"width": "15%", "position": "relative", "display": "none"}, {"width": "15%", "position": "relative", "display": "none"}, {"width": "15%", "position": "relative"}, "bt-dtc"
    return no_update, no_update, no_update, no_update

@app.callback(
    [
        Output("s-info", "style"),
        Output("s-input", "style"),
        Output("bt-fit-reset", "style"),
        Output("bt-fit-step", "style"),
        Output("bt-fit-skip", "style"),
        Output("bt-fit-svm", "style"),
        Output("train-info", "style"),
        Output("step-display", "style"),
    ],
    [
        Input("batch-switch", "on"),
    ],
)
def update_batch_parameters(is_on):
    none_display = {"display": "none"}
    if is_on: return {}, {}, {}, {}, {}, none_display, {}, {}
    else: return none_display, none_display, none_display, none_display, none_display, {"margin-right": "0px", "margin-left": "0px", "margin-top": "12px"}, none_display, none_display
    
# Função para gerar uma figura seguindo os padrões do dashboard
def generate_figure(X):
    fig = go.Figure()
    fig.update_layout(showlegend=True, margin=dict(l=16, r=16, t=16, b=16), xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False), plot_bgcolor="rgba(203, 203, 212, .3)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="DIN Alternate",
        xaxis_range=[X[:, 0].min() - 1, X[:, 0].max() + 1],
        yaxis_range=[X[:, 1].min() - 1, X[:, 1].max() + 1])
    
    return fig
    
# Função para gerar o dataset
def get_dataset(data_type, n, std, seed):
    if data_type == "blobs": X, y = make_blobs(n_samples=int(n), centers=2, n_features=2, cluster_std=float(std), random_state=int(seed))
    elif data_type == "circles": X, y = make_circles(n_samples=int(n), noise=float(std), random_state=int(seed))
    elif data_type == "moons": X, y = make_moons(n_samples=int(n), noise=float(std), random_state=int(seed))
    
    return X, y
    
# Visualização dos pontos do SVM batch
def plot_svm_batch_points(S_X, S_y, U_X, U_y):
    trace_specs = [
        [S_X, S_y, -1, 'S', 'circle'],
        [S_X, S_y, 1, 'S', 'circle'],
        [U_X, U_y, -1, 'U', 'square'],
        [U_X, U_y, 1, 'U', 'square']
    ]

    return [
        go.Scatter(
            x=X[y==label, 0], y=X[y==label, 1],
            name=f'{split} Split, Label {label}',
            mode='markers', marker_symbol=marker,
            marker=dict(color=y)
        )
        for X, y, label, split, marker in trace_specs
    ]

def plot_svc_decision_function(xlim, ylim, model, margin=1, size=100):
    x = np.linspace(xlim[0] - margin, xlim[1] + margin, size)
    y = np.linspace(ylim[0] - margin, ylim[1] + margin, size)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.decision_function(xy).reshape(xx.shape)
    # plot decision boundary and margins
    return go.Contour(x=x, y=y, z=Z, colorscale=colorscale,
            contours_coloring="heatmap", hoverinfo="none", showlegend=False, showscale=False, opacity=0.8
            #line_width=2,
        )
               
def plot_decision_function(xlim, ylim, model, margin=1, size=100):
    x = np.linspace(xlim[0] - margin, xlim[1] + margin, size)
    y = np.linspace(ylim[0] - margin, ylim[1] + margin, size)
    xx, yy = np.meshgrid(x, y)
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    return go.Contour(x=x, y=y, z=Z, colorscale=colorscale,
            contours_coloring="heatmap", hoverinfo="none", showlegend=False, showscale=False, opacity=0.8
            #line_width=2,
    )
    
if __name__ == '__main__':
    app.run_server(debug=True)
