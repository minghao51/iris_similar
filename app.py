# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import numpy as np
from Scripts.helper import *
from dash.dependencies import Input, Output, State
import plotly.express as px
import base64
import io 
import json


external_stylesheets = [dbc.themes.BOOTSTRAP]



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Link", href="#")),
        # html.Button('Data Quality Check', id='data-quality-val', n_clicks=0),
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Menu",
            children=[
                dbc.DropdownMenuItem("Entry 1"),
                dbc.DropdownMenuItem("Entry 2"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Entry 3"),
            ],
        ),
    ],
    brand="Iris Neighbors",
    brand_href="#",
    sticky="top",
)

body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Controls"),
                        html.Label('Number of Similar Neighbors'),
                        html.Div(dcc.Slider(
                            id='neighbor_slider',
                            min=0,
                            max=30,
                            marks={
                                10: '10',
                                20: '20',
                                30: '30'
                            },
                            value=10,
                            step=None,
                        )),
                        html.Label('Data Selection'),
                        html.Div(dcc.Dropdown(
                            id='data_ctl',
                            options=[
                                {'label': 'All', 'value': 'All'},
                                {'label': 'Iris-setosa', 'value': 'Iris-setosa'},
                                {'label': 'Iris-versicolor', 'value': 'Iris-versicolor'},
                                {'label': 'Iris-virginica', 'value': 'Iris-virginica'},
                            ],
                            value='All'
                        )),
                        html.Label('Distance Measure'),
                        html.Div(dcc.Dropdown(
                            id='distance_ctl',
                            options=[
                                {'label': 'Euclidean-Space', 'value': 'Euclidean-Space'},
                                {'label': 'PCA-Space', 'value': 'PCA-Space'}
                            ],
                            value='Euclidean-Space'
                        )),
                        html.Label('Dimension'),
                        html.Div(dcc.Dropdown(
                            id='dim_ctl',
                            options=[
                                {'label': '2', 'value': 2},
                                {'label': '3', 'value': 3}
                            ],
                            value=3
                        )),
                        html.Label('Visualisation'),
                        html.Div(dcc.Dropdown(
                            id='visual_ctl',
                            options=[
                                {'label': 'Scatter', 'value': 'scatter'},
                                {'label': 'Parrallel Coordinates', 'value': 'paral_coor'},
                                {'label': 'RadViz', 'value': 'radviz'},
                            ],
                            value='scatter'
                        )),
                        html.H2("Input"),
                        html.Label('sepal length cm'),
                        html.Div(dcc.Input(
                            id='sepal_length',
                            placeholder='sepal length cm',
                            type='number',
                            value='5.1'
                        )),
                        html.Label('sepal width cm'),
                        html.Div(dcc.Input(
                            id='sepal_width',
                            placeholder='sepal width cm',
                            type='number',
                            value='3.5'
                        )),
                        html.Label('petal length cm'),
                        html.Div(dcc.Input(
                            id='petal_length',
                            placeholder='petal length cm',
                            type='number',
                            value='1.4' 
                        )),
                        html.Label('petal width cm'),
                        html.Div(dcc.Input(
                            id='petal_width',
                            placeholder='petal width cm',
                            type='number',
                            value='0.2'
                        )), 
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.H2("Graph"),
                        # dcc.Graph(id="graph"),
                        html.Div(
                            children=[dcc.Loading(id='indicator-graphic',type='graph', className='')]
                        ),
                    ]
                ),
            ]
        )
    ],
    className="mt-6",
)

modal = html.Div(
    [   
        dcc.Store(id='quality_data', storage_type='session'),
        dbc.Button("Data Quality", id="open"),
        dbc.Modal(
            [
                dbc.ModalHeader("Data Quality Result"),
                dbc.ModalBody("This is the content of the modal", id="modal_body"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", className="ml-auto")
                ),
            ],
            id="modal",
        ),
    ]
)

app.layout =  html.Div([navbar, body, modal])

@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal_body", "children"),
    [Input("quality_data", "data")]
)
def modal_children(data):
    with open('data_quality.output') as f:
        lines = f.readlines()
    # lines = lines[0].replace('\n', '<br>')
    return lines

@app.callback(
    [Output('indicator-graphic', 'children'), Output('quality_data', 'data')],
    [Input('neighbor_slider', 'value'),
     Input('data_ctl', 'value'),
     Input('distance_ctl', 'value'),
     Input('dim_ctl', 'value'),
     Input('visual_ctl', 'value'),
     Input('sepal_length', 'value'),
     Input('sepal_width', 'value'),
     Input('petal_length', 'value'),
     Input('petal_width', 'value')]
     )
def update_graph(neighbor_slider, 
                data_ctl, distance_ctl,
                dim_ctl, visual_ctl,
                sepal_length, sepal_width,
                petal_length, petal_width):

    
    Top_n_counts=int(neighbor_slider)
    export_ctl='False'

    #Input Data
    input_x=[[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]]

    # loading data
    df, target, features = loading_data()

    check_data_quality = data_quality(df)

    # Data Selection
    df = selection_data(df, data_ctl)
    
    # Identifying neigbors
    Y,X = prep_model(df, target)

    # dist measure
    df_final = dist_measure(X, Y, input_x, distance_ctl, dim_ctl)

    # columns from dimension control 
    if dim_ctl == 3:
        scatter_x = list(df_final.columns[1:4])
    elif dim_ctl == 2:
        scatter_x = list(df_final.columns[1:3])

    df_similar = df_final.loc[(df_final['similarity_ranked']<=neighbor_slider-1) & (df_final['Top_n_similar']!='input')]
    df_plot = df_final.loc[(df_final['Top_n_similar']=='input')]

    # Extraction
    output = df_similar.iloc[:,0:dim_ctl+2]

    # Visual
    print(f'Visualisation with for method : {visual_ctl}')

    ### Parallel Coordinates
    if visual_ctl == 'scatter':
        
        # 2d scatter plot
        if dim_ctl == 2:
            fig = px.scatter(df_final, x=scatter_x[0], y=scatter_x[1],
                        color='Top_n_similar', size_max=18,
                        symbol='Class', opacity=0.5)
        # 3d scatter plot
        elif dim_ctl == 3:
            fig = px.scatter_3d(df_final, x=scatter_x[0], y=scatter_x[1], z=scatter_x[2],
                        color='Top_n_similar', size_max=18,
                        symbol='Class', opacity=0.5)

        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), title="Scatter Plot")
        fig.layout.font = dict(family="Helvetica")

        return [
            html.Div(dcc.Graph(figure = fig)),
            html.Div(
                dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in output.columns],
                data=output.to_dict('records'),
                export_columns='all',
                export_headers='display',
                export_format='csv'))
        ], check_data_quality

     ### Radviz
    if visual_ctl == 'radviz':
        scatter_x.append('Top_n_similar')
   
        radviz_fig, ax = plt.subplots( nrows=1, ncols=1 )

        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # Plot
        ax = radviz(df_final[scatter_x], "Top_n_similar", color =['Red','Orange','Blue','Green'], alpha=0.5)
        ax.title.set_text('Radviz Plot of the Features')

        tmp_path = 'tmp/MultiDimension_Radviz.png'
        radviz_fig.savefig(tmp_path, bbox_inches='tight')

        tmp_base64 = base64.b64encode(open(tmp_path, 'rb').read()).decode('ascii')

        return [
            html.Div(html.Img(src='data:image/png;base64,{}'.format(tmp_base64)),  className='db'),            html.Div(
                dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in output.columns],
                data=output.to_dict('records'),
                export_columns='all',
                export_headers='display',
                export_format='csv'))
        ], check_data_quality

    ### Raw
    if visual_ctl == 'paral_coor':
        parallel_x = scatter_x.copy()
        parallel_x.append('similar_alpha')
        # parallel plot
        fig = px.parallel_coordinates(df_final[parallel_x],
                                    color="similar_alpha", 
                                    dimensions=scatter_x,
                                    color_continuous_scale=px.colors.diverging.Tealrose, 
                                    color_continuous_midpoint=2)

        # Layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),  title="Parallel Coord Plot",)
        fig.layout.font = dict(family="Helvetica")

        # Output
        return [
            html.Div(dcc.Graph(figure = fig)),
            html.Div(html.P('Where similar alpha = 3 are the input measurement; similar alpha = 2 are the most similar iris flower; the rest are the other iris flower')),
            html.Div(
                dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in output.columns],
                data=output.to_dict('records'),
                export_columns='all',
                export_headers='display',
                export_format='csv'))
        ], check_data_quality

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True, port=8050)