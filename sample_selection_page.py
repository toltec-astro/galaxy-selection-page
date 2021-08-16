"""
Created on Thu Mar  4 14:38:20 2021

@author: Caleigh Ryan

Version 2.1 of Sample definition page

Implements galaxy_class to read in sample
"""

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
# from astropy.utils.data import get_pkg_data_filename
import dash_bootstrap_components as dbc
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from galaxy_selection_page import galaxy_class as gc
import os
import flask
from pathlib import Path


if flask.current_app:
    # flask server already exists, which is created within dasha.
    # from dasha.web.extensions.dasha import dash_app as app
    from dasha.web.extensions.dasha import dash_app as app
    from dasha.web.extensions.cache import cache
    cache_func = cache.memoize()
    app.config.external_stylesheets.append(dbc.themes.LUX)
else:
    app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
    server = app.server
    # need to make available the parent as package
    # this is already done in dasha_app.py if run from DashA
    sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())
    cache_func = functools.lru_cache

#Create app 
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Get Paths to data (csv and fits)
datadir = Path(__file__).parent.joinpath('data')
dustopedia_csv = datadir.joinpath('Samples/dustopedia_sample.csv').as_posix()

kingfish_csv = datadir.joinpath('Samples/kingfish_sample.csv').as_posix()
kingfish_fits = os.listdir(datadir.joinpath("Samples/Kingfish_FITS/Spire/KINGFISH_SPIRE_v3.0_updated_updated/KINGFISH_SPIRE_v3.0_updated_updated").as_posix())

dgs_csv = datadir.joinpath('Samples','dgs_sample.csv').as_posix()
#dgs_fits = os.listdir("C:\Fall_2020_Wilson_Lab\SN_Page\Samples\DGS_FITS\Renamed_FITs")
dgs_fits = os.listdir(datadir.joinpath("Samples/DGS_FITS/Renamed_FITs").as_posix())

# Create galaxy objects for each sample
dustopedia = gc.Sample('Dustopedia',dustopedia_csv)
kingfish = gc.Sample('Kingfish',kingfish_csv,kingfish_fits)
dgs = gc.Sample('DGS', dgs_csv, dgs_fits)

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
    Output('sample', 'value'),
    Input('sky-map', 'clickData')  
)

def updateSampleDropdown(skyClickData):
    event = dash.callback_context.triggered[0]["prop_id"].split('.')[0]
    if (event == 'sky-map'):
        if (skyClickData['points'][0]['x'] in samples['Kingfish'].data['RA'].unique()):
            return 'Kingfish'
        elif (skyClickData['points'][0]['x'] in samples['DGS'].data['RA'].unique()):
            return 'DGS'
        else:
            return 'Dustopedia'
    else: return dash.no_update

@app.callback(
    [Output('galaxy-name','options'),
    Output('galaxy-name', 'value')],
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
            return new_names, value
        elif (skyClickData['points'][0]['x'] in samples['DGS'].data['RA'].unique()):
            dgs = samples['DGS']
            index = dgs.data.index
            row = index[dgs.data['RA'] == skyClickData['points'][0]['x']]
            value = dgs.data['Object Name'][row].values[0]
            new_names = get_galaxy_names('DGS')
            return new_names, value
        else:
            return [{'label': 'No Dustopedia Fits', 'value': 'dustopedia'}],'No Dustopedia Fits'
    else: 
        if selected_sample == 'Kingfish':
            new_names = get_galaxy_names(selected_sample)
            return new_names, new_names[1]['value'] 
        elif selected_sample == 'DGS':
            new_names = get_galaxy_names(selected_sample)
            return new_names, new_names[1]['value'] 
        else:
            return [{'label': 'No Dustopedia Fits', 'value': 'Dustopedia'}], 'Dustopedia' 

@app.callback(
    Output('fits-image','figure'),
    [Input('sample','value'),
     Input('galaxy-name','value'),
     Input('sky-map', 'clickData'),
     Input('band', 'value')]
    )
def updateFitsImage(selected_sample, name, skyClickData, band): 
    event = dash.callback_context.triggered[0]["prop_id"].split('.')[0]
    if event == 'sky-map':
        if (skyClickData['points'][0]['x'] in samples['Kingfish'].data['RA'].unique()):
            kingfish = samples['Kingfish']
            index = kingfish.data.index
            row = index[kingfish.data['RA'] == skyClickData['points'][0]['x']]
            if (kingfish.data[band][row].values[0] != None):
                color_label, x, y = kingfish.update_herschel('Kingfish', name, band)
                img_path = kingfish.data[band][row].values[0]
                img_data = img_path
                img = fits.getdata(img_data, ext=0)
                fig = px.imshow(img, x=x, y=y, zmax=img.max(), labels = {'color': color_label})
                fig.update_xaxes(title='x (arcsec)')
                fig.update_yaxes(title='y (arcsec)')
            else: fig = {}
        elif (skyClickData['points'][0]['x'] in samples['DGS'].data['RA'].unique()):
            dgs = samples['DGS']
            index = dgs.data.index
            row = index[dgs.data['RA'] == skyClickData['points'][0]['x']]
            if (dgs.data[band][row].values[0] != None):
                color_label, x, y = dgs.update_herschel('DGS',name, band)
                img_path = dgs.data[band][row].values[0]
                img_data = img_path
                img = fits.getdata(img_data, ext=0)
                fig = px.imshow(img, x=x, y=y, zmax=img.max(), labels = {'color': color_label})
                fig.update_xaxes(title='x (arcsec)')
                fig.update_yaxes(title='y (arcsec)')
            else: fig = {}
        else: fig = {}
    else:
        if selected_sample == 'Kingfish':
            sample = samples[selected_sample]
            index = sample.data.index
            row = index[sample.data['Object Name'] == name]
            if (sample.data[band][row].values[0] != None):
                kingfish = samples['Kingfish']
                color_label, x, y = kingfish.update_herschel(selected_sample, name, band)
                img_path = sample.data[band][sample.data['Object Name'] == name].values[0]
                img_data = img_path
                img = fits.getdata(img_data, ext=0)
                fig = px.imshow(img, x=x, y=y, zmax=img.max(), labels = {'color': color_label})
                fig.update_xaxes(title='x (arcsec)')
                fig.update_yaxes(title='y (arcsec)')
            else: fig = {}
        elif  selected_sample == 'DGS':
            sample = samples[selected_sample]
            index = sample.data.index
            row = index[sample.data['Object Name'] == name]
            if (sample.data[band][row].values[0] != None):
                dgs = samples['DGS']
                print(dgs.data[band][row].values[0])
                color_label, x, y = dgs.update_herschel(selected_sample, name, band)
                img_path = sample.data[band][sample.data['Object Name'] == name].values[0]
                img_data = img_path
                img = fits.getdata(img_data, ext=0)
                fig = px.imshow(img, x=x, y=y, zmax=img.max(), labels = {'color': color_label})
                fig.update_xaxes(title='x (arcsec)')
                fig.update_yaxes(title='y (arcsec)')
            else: fig = {}
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
        fig = go.Figure(data=[go.Scatter(x=dustopedia.data['RA'], 
                    y=dustopedia.data['DEC'], 
                    mode='markers',
                    marker=dict(color='#FC8D62'),
                    name='Dustopedia')])
        fig.add_trace(go.Scatter(x=kingfish.data['RA'], 
                    y=kingfish.data['DEC'],
                    mode='markers',
                    marker=dict(color='#66C2A5'),
                    name='Kingfish')) 
        fig.add_trace(go.Scatter(x=dgs.data['RA'], 
                    y=dgs.data['DEC'],
                    mode='markers',
                    marker=dict(color='#8DA0CB'),
                    name='DGS'))
        fig.update_layout(
                title='All Sky Map',
                xaxis_title="RA",
                yaxis_title="DEC"
            )
    elif(event == 'x-dropdown' or event == 'y-dropdown' or event == 'hist-param'):
        return dash.no_update
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

