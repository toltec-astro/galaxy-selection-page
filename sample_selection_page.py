"""
Created on Thu Mar  4 14:38:20 2021

@author: Caleigh Ryan

Version 2.1 of Sample definition page

Implements galaxy_class to read in sample
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from astropy.table import Table, Column
from dash.dependencies import Input, Output
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.utils.data import get_pkg_data_filename
import dash_bootstrap_components as dbc
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
import galaxy_class as gc
import os


#Create app 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Get Paths to data (csv and fits)
# TODO: add dgs fits
dustopedia_csv = os.path.join('Samples','dustopedia_sample.csv')

kingfish_csv = os.path.join('Samples','kingfish_sample.csv')
kingfish_fits = os.listdir('Samples\\Kingfish_FITS\\Spire\\KINGFISH_SPIRE_v3.0_updated\\KINGFISH_SPIRE_v3.0_updated')

dgs_csv = os.path.join('Samples','dgs_sample.csv')

# Create galaxy objects for each sample
# TODO: add dgs fits
dustopedia = gc.Sample('Dustopedia',dustopedia_csv)
kingfish = gc.Sample('Kingfish',kingfish_csv,kingfish_fits)
dgs = gc.Sample('DGS', dgs_csv)

# Create names list for fits dropdown, and dictionary mapping to galaxy objects
names = []
for name in kingfish.data['Object Name']:
    if type(name) is str:
        names.append({'label': name, 'value': name})
samples = {'Kingfish': kingfish, 'DGS': dgs, 'Dustopedia': dustopedia} 

def get_galaxy_names(selected_sample='Kingfish'):
    galaxy_names = []
    sample = samples[selected_sample]
    for name in sample.data['Object Name']:
        if type(name) is str:
            galaxy_names.append({'label': name, 'value': name})
    return galaxy_names

# Options for scatterplot and histogram axes
axes_options = ['Object Name','RA','DEC','Redshift','Distance','Log Stellar Mass','Metallicity','Diameter (arcsec)']

# Options for sample names dropdown
sampleNames = ['Kingfish', 'DGS', 'Dustopedia']



#App setup
"""
app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='sample',
                options=[{'label':i[0], 'value':i[1]} for i in samples],
                value='Kingfish'
            )
        ]),
        dbc.Col([
            dcc.Dropdown(
                id='galaxy-name',
                options=[{'label':i, 'value':i} for i in names],
                value='NGC 0007'
            )
        ]),
        dbc.Col([
            dcc.Dropdown(
                id='band',
                options={'500 band':500, '350 band':350, '250 band':250},
                value='500'
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            dcc.Graph(
                id='fits-image'
            )
        ], width=5),
        dbc.Col([], width=2)
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(className="h-50"),
            dcc.Dropdown(
                id='y-dropdown',
                options=[{'label':i, 'value':i} for i in temp_dust_df],
                value='DEC',
                style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},
            ),
        ], width=2),
        dbc.Col([
            dcc.Graph(id='scatter'),
        ], width=5),
        dbc.Col([
            dcc.Graph(id='sky-map')
        ], width=5)
    ]),
    dbc.Row([
        dbc.Col([
        ], width=2),
        dbc.Col([
            dcc.Dropdown(
                id='x-dropdown',
                options=[{'label':i, 'value':i} for i in temp_dust_df],
                value='RA',
                style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            )
        ], width=5),
        dbc.Col([
            # Sky Map switch units go here
        ], width=5)
    ]),
    dbc.Row([
        dbc.Col([
        ], width=2),
        dbc.Col([
            dcc.Graph(id='histogram'),
        ], width=5)
    ]),
    dbc.Row([
        dbc.Col([
        ], width=2),
        dbc.Col([
            dcc.Dropdown(
                id='hist-param',
                options=[{'label':i, 'value':i} for i in temp_dust_df],
                value='Redshift'
            )
        ], width=5)
    ])
])

app.layout = html.Div([
    dbc.Row([
        dbc.Col([ 
        ]),
        dbc.Col([
            dcc.Dropdown(
                id='sample',
                #options=[{'label':i[0], 'value':i[1]} for i in samples],
                options=[{'Kingfish': kingfish}],
                value='Kingfish'
            ),
            dcc.Dropdown(
                id='galaxy-name',
                options=[{'label':i, 'value':i} for i in names],
                value='NGC 0007'
            ),
            dcc.Dropdown(
                id='band',
                options={'500 band':500, '350 band':350, '250 band':250},
                value='500'
            )
        ]),
        dbc.Col([  
        ])
    ]),
    dbc.Row([
        dbc.Col([
            ]),
        dbc.Col([
            dcc.Graph(
                id='fits-image'
            )
        ]),
        dbc.Col([
            ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='y-dropdown',
                options=[{'label':i, 'value':i} for i in temp_dust_df],
                value='DEC',
                style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},
            ),
            dcc.Dropdown(
                id='x-dropdown',
                options=[{'label':i, 'value':i} for i in temp_dust_df],
                value='RA',
                style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            )
        ]),
        dbc.Col([
        ]),
        dbc.Col([  
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='scatter')
        ]),
        dbc.Col([
            dcc.Graph(id='sky-map')
        ]),
        dbc.Col([
            #Sky map pick telescope goes here
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='hist-param',
                options=[{'label':i, 'value':i} for i in temp_dust_df],
                value='Redshift'
            )
        ]),
        dbc.Col([
            # Sky Map switch units go here
        ]),
        dbc.Col([])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='histogram')
        ]),
        dbc.Col([ 
        ]),
        dbc.Col([])
    ])
])
"""

header = html.Div(dbc.Row(dbc.Col(html.Div(
    html.H2("Sample Selection",
            style={
                'margin-top': '50px',
                'margin-left': '40%',
                'margin-bottom': '50px'
                })))))

row1 = html.Div(dbc.Row([
            dbc.Col(html.Div(
                dcc.Dropdown(
                    id='sample',
                    options=[{'label': i, 'value': i} for i in sampleNames],
                    value='Kingfish'
                )
            ), style={'width': '33%'}),
            dbc.Col(html.Div(
                dcc.Dropdown(
                    id='galaxy-name',
                    options=get_galaxy_names(),
                    value=names[1]['value']
                )
            ), style={'width': '33%'}),
            dbc.Col(html.Div(
                dcc.Dropdown(
                id='band',
                    options=[{'label':'500 band', 'value':'500 band'}, 
                            {'label':'350 band','value':'350 band'}, 
                            {'label':'250 band', 'value':'250 band'}],
                    value='250 band'
                )
            ), style={'width': '33%'})
        ], style={'display': 'flex'})
    )

row2 = html.Div(dbc.Row(
        dbc.Col(html.Div(dcc.Graph(id='fits-image'))),
        style={'display': 'flex','margin-left':'20%'})
    )

row3 = html.Div(dbc.Row(
        dbc.Col(html.Div(dcc.Graph(id='sky-map'))),
        style={'display': 'flex','margin-left':'20%'})
    )

row4 = html.Div(dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(html.Div(
                        dcc.Dropdown(
                            id='x-dropdown',
                            options=[{'label':i, 'value':i} for i in axes_options],
                            value='RA'
                        )
                    ), style={'width': '50%'}),
                    dbc.Col(html.Div(
                        dcc.Dropdown(
                            id='y-dropdown',
                            options=[{'label':i, 'value':i} for i in axes_options],
                            value='DEC'
                        )
                    ), style={'width': '50%'}),   
                ], style={'display': 'flex'}),
                dbc.Row(
                    dbc.Col(html.Div(dcc.Graph(id='scatter')),
                        style={'display':'flex'})
                )
            ], style={'width': '50%'}),
            dbc.Col([
                dbc.Row(
                    dbc.Col(html.Div(
                        dcc.Dropdown(
                            id='hist-param',
                            options=[{'label':i, 'value':i} for i in axes_options],
                            value='Redshift'
                        )
                    ))
                ),
                dbc.Row(
                    dbc.Col(html.Div(dcc.Graph(id='histogram')))
                )
            ], style={'width': '50%'})
        ], style={'display': 'flex'})
    )
sky_map_text = html.Div(dbc.Row(dbc.Col(html.Div(
    dcc.Markdown('Select points from the All Sky Map to plot the image of the galaxy above.')
))))

scatter_hist_text = html.Div(dbc.Row(dbc.Col(html.Div(
    dcc.Markdown('Use the box select tool on the scatter plot and histogram to filter galaxies on the All Sky Map.')
))))

app.layout = html.Div(children=[
    header,
    row1,
    row2,
    sky_map_text,
    row3,
    scatter_hist_text,
    row4
])


@app.callback(
    [Output('galaxy-name','options'),
    Output('galaxy-name', 'value'),
    Output('sample', 'value')],
    [Input('sample','value'), 
    Input('sky-map','clickData')]
)
def updateNamesDropdown(selected_sample, skyClickData):
    event = dash.callback_context.triggered[0]["prop_id"].split('.')[0]
    if (event == 'sky-map'):
        if (skyClickData['points'][0]['x'] in samples['Kingfish'].data['RA'].unique()):
            kingfish = samples['Kingfish']
            index = kingfish.data.index
            row = index[kingfish.data['RA'] == skyClickData['points'][0]['x']]
            value = kingfish.data['Object Name'][row].values[0]

            new_names = get_galaxy_names('Kingfish')
            return new_names, value, 'Kingfish'
        elif (skyClickData['points'][0]['x'] in samples['DGS'].data['RA'].unique()):
            # index = dgs.data.index
            # row = index[dgs.data['RA'] == skyClickData['points'][0]['x']]
            # value = dgs.data['Object Name'][row].values[0]
            return [{'label': 'No DGS Fits', 'value': 'dgs'}],'No DGS Fits', 'DGS'
        else:
            return [{'label': 'No Dustopedia Fits', 'value': 'dustopedia'}],'No Dustopedia Fits', 'Dustopedia'
    else: 
        if selected_sample == 'Kingfish':
            new_names = get_galaxy_names(selected_sample)
            return new_names, new_names[1]['value'], selected_sample
        elif selected_sample == 'DGS':
            return [{'label': 'No DGS Fits', 'value': 'DGS'}], 'DGS', selected_sample
        else:
            return [{'label': 'No Dustopedia Fits', 'value': 'Dustopedia'}], 'Dustopedia', selected_sample

@app.callback(
    Output('fits-image','figure'),
    [Input('sample','value'),
     Input('galaxy-name','value'),
     Input('sky-map', 'clickData'),
     Input('band', 'value')]
     #Input('pacs-or-spire', 'value')]
    )
def updateFitsImage(selected_sample, name, skyClickData, band):  #add camera as last attribute
    event = dash.callback_context.triggered[0]["prop_id"].split('.')[0]
    if event == 'sky-map':
        if (skyClickData['points'][0]['x'] in samples['Kingfish'].data['RA'].unique()):
            kingfish = samples['Kingfish']
            index = kingfish.data.index
            row = index[kingfish.data['RA'] == skyClickData['points'][0]['x']]
            if (kingfish.data[band][row].values[0] != None):
                img_path = kingfish.data[band][row].values[0]
                img_data = get_pkg_data_filename(img_path)  
                img = fits.getdata(img_data, ext=0)
                fig = px.imshow(img, zmax=img.max())
            else: fig = {}
        elif (skyClickData['points'][0]['x'] in samples['DGS'].data['RA'].unique()):
            # index = dgs.data.index
            # row = index[dgs.data['RA'] == skyClickData['points'][0]['x']]
            # img_path = dgs.data[band][row].values[0]
            fig = {}
        else:
            fig = {}
    else:
        sample = samples[selected_sample]
        if (name in sample.data['Object Name'].unique()):
            index = sample.data.index
            row = index[sample.data['Object Name'] == name]
            img_path = sample.data[band][sample.data['Object Name'] == name].values[0]
            img_data = get_pkg_data_filename(img_path)
            img = fits.getdata(img_data, ext=0)
            fig = px.imshow(img, zmax=img.max())
        else: fig = {}
    return fig

@app.callback(
    Output('sky-map', 'figure'),
    [Input('scatter', 'selectedData'),
     Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value'),
     Input('hist-param', 'value'),
     Input('histogram', 'selectedData')]
    )
def update_sky_map(selectedData,param1,param2,histparam,histSelectedData):
    event = dash.callback_context.triggered[0]["prop_id"].split('.')[0]
    if(selectedData == None and histSelectedData == None):
        fig = go.Figure(data=[go.Scatter(x=dustopedia.data[param1], 
                    y=dustopedia.data[param2], 
                    mode='markers',
                    marker=dict(color='#FC8D62'),
                    name='Dustopedia')])
        fig.add_trace(go.Scatter(x=kingfish.data[param1], 
                    y=kingfish.data[param2],
                    mode='markers',
                    marker=dict(color='#66C2A5'),
                    name='Kingfish')) 
        fig.add_trace(go.Scatter(x=dgs.data[param1], 
                    y=dgs.data[param2],
                    mode='markers',
                    marker=dict(color='#8DA0CB'),
                    name='DGS'))
        fig.update_layout(
                title='All Sky Map',
                xaxis_title="RA",
                yaxis_title="DEC"
            )
    elif(event == 'x-dropdown' or event == 'y-dropdown' or event == 'hist-param'):
        dash.no_update
    elif((selectedData != None and 'range' in selectedData) or (histSelectedData != None and 'range' in histSelectedData)):
        if(selectedData != None):
            xmin,xmax = selectedData['range']['x'][0:2]
            ymin,ymax = selectedData['range']['y'][0:2]
        else: 
            xmin,xmax = (min(kingfish.data[param1].min(),dustopedia.data[param1].min(),dgs.data[param1].min()),
                         max(kingfish.data[param1].max(),dustopedia.data[param1].max(),dgs.data[param1].max()))
            ymin,ymax = (min(kingfish.data[param2].min(),dustopedia.data[param2].min(),dgs.data[param2].min()),
                         max(kingfish.data[param2].max(),dustopedia.data[param2].max(),dgs.data[param2].max()))
            
        if(histSelectedData != None):
            histmin,histmax = histSelectedData['range']['x'][0:2]
        else:
            histmin,histmax = (min(kingfish.data[histparam].min(),dustopedia.data[histparam].min(),dgs.data[histparam].min()),
                               max(kingfish.data[histparam].max(),dustopedia.data[histparam].max(),dgs.data[histparam].max()))
            
        kingMask = ((kingfish.data[param1] >= xmin) & 
                (kingfish.data[param1] < xmax) &
                (kingfish.data[param2] >= ymin) &
                (kingfish.data[param2] < ymax) &
                (kingfish.data[histparam] < histmax) &
                (kingfish.data[histparam] >= histmin))
        dustMask = ((dustopedia.data[param1] >= xmin) & 
                (dustopedia.data[param1] < xmax) &
                (dustopedia.data[param2] >= ymin) &
                (dustopedia.data[param2] < ymax) &
                (dustopedia.data[histparam] < histmax) &
                (dustopedia.data[histparam] >= histmin))
        dgsMask = ((dgs.data[param1] >= xmin) & 
                (dgs.data[param1] < xmax) &
                (dgs.data[param2] >= ymin) &
                (dgs.data[param2] < ymax) &
                (dgs.data[histparam] < histmax) &
                (dgs.data[histparam] >= histmin))
        fig = go.Figure(data=[go.Scatter(x=dustopedia.data[dustMask]['RA'], 
                        y=dustopedia.data[dustMask]['DEC'],
                        mode='markers',
                        marker=dict(color='#FC8D62'),
                        name='Dustopedia')])
        fig.add_trace(go.Scatter(x=kingfish.data[kingMask]['RA'], 
                        y=kingfish.data[kingMask]['DEC'],
                        mode='markers',
                        marker=dict(color='#66C2A5'),
                        name='Kingfish'))
        fig.add_trace(go.Scatter(x=dgs.data[dgsMask]['RA'],
                        y=dgs.data[dgsMask]['DEC'],
                        mode='markers',
                        marker=dict(color='#8DA0CB'),
                        name='DGS'))
        fig.update_layout(
                title='All Sky Map',
                xaxis_title="RA",
                yaxis_title="DEC"
            )
    return fig

@app.callback(
    Output('scatter', 'figure'),
    [Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value')]
    )
def update_scatter(param1, param2):
    fig = go.Figure(data=[go.Scatter(x=dustopedia.data[param1], 
                    y=dustopedia.data[param2], 
                    mode='markers',
                    marker=dict(color='#FC8D62'),
                    name='Dustopedia')])
    fig.add_trace(go.Scatter(x=kingfish.data[param1], 
                    y=kingfish.data[param2],
                    mode='markers',
                    marker=dict(color='#66C2A5'),
                    name='Kingfish')) 
    fig.add_trace(go.Scatter(x=dgs.data[param1], 
                    y=dgs.data[param2],
                    mode='markers',
                    marker=dict(color='#8DA0CB'),
                    name='DGS'))
    fig.update_layout(
                xaxis_title=param1,
                yaxis_title=param2
            )
    return fig

@app.callback(
    Output('histogram', 'figure'),
    Input('hist-param', 'value')
    )
def update_hist(param):
    fig = go.Figure(data=[go.Histogram(x=dustopedia.data[param],
                    marker=dict(color='#FC8D62'),
                    name='Dustopedia',
                    nbinsx=int(np.sqrt(len(dustopedia.data[param]))))])
    fig.add_trace(go.Histogram(x=kingfish.data[param], 
                    marker=dict(color='#66C2A5'),
                    name='Kingfish',
                    nbinsx=int(np.sqrt(len(dustopedia.data[param])))))
    fig.add_trace(go.Histogram(x=dgs.data[param], 
                    marker=dict(color='#8DA0CB'),
                    name='DGS',
                    nbinsx=int(np.sqrt(len(dustopedia.data[param])))))
    fig.update_layout(barmode='stack',
                    xaxis_title=param )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

