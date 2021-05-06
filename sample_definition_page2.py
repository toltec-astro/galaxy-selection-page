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


#Get Paths to data and open names column
dustopedia_csv = r"C:\Fall_2020_Wilson_Lab\SN_Page\Samples\dustopedia_sample.csv"
#dust_names = Column(ascii.read(dustopedia_csv,format='csv'),'Object Name')

kingfish_csv = r"C:\Fall_2020_Wilson_Lab\SN_Page\Samples\kingfish_sample.csv"
kingfish_fits = os.listdir('Samples\\Kingfish_FITS\\Spire\\KINGFISH_SPIRE_v3.0_updated\\KINGFISH_SPIRE_v3.0_updated')

#Create galaxy objects for each sample
dustopedia = gc.Sample('Dustopedia',dustopedia_csv)
kingfish = gc.Sample('Kingfish',kingfish_csv,kingfish_fits)

#names list for fits plotting, add kingfish and dgs
names = []
for name in kingfish.data['Object Name']:
    if type(name) is str:
        names.append({'label': name, 'value': name})

temp_dust_df = pd.read_csv(dustopedia_csv)
samples = {'Kingfish': kingfish}  #add dgs


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



app.layout = html.Div(children=[
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='sample',
                #options=[{'label':i[0], 'value':i[1]} for i in samples],
                options=[{'label': 'Kingfish', 'value':'Kingfish'}],
                value='Kingfish'
            )
        ]),
        html.Div([
            dcc.Dropdown(
                id='galaxy-name',
                options=names,
                value=names[1]['value']
            )
        ]),
        html.Div([
            dcc.Dropdown(
                id='band',
                options=[{'label':'500 band', 'value':'500 band'}, 
                         {'label':'350 band','value':'350 band'}, 
                         {'label':'250 band', 'value':'250 band'}],
                value='500 band'
            )
        ])
    ]),
    html.Div([
        dcc.Graph(
            id='fits-image'
        )
    ]),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='y-dropdown',
                options=[{'label':i, 'value':i} for i in dustopedia.data],
                value='DEC',
                style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},
            )
        ]),
        html.Div([
            dcc.Dropdown(
                id='x-dropdown',
                options=[{'label':i, 'value':i} for i in dustopedia.data],
                value='RA',
                style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            )
        ])
    ]),
    html.Div([
        html.Div([
            dcc.Graph(id='scatter')
        ]),
        html.Div([
            dcc.Graph(id='sky-map')
        ])
    ]),
    html.Div([
       dcc.Dropdown(
           id='hist-param',
           options=[{'label':i, 'value':i} for i in temp_dust_df],
           value='Redshift'
       )
    ]),
    html.Div([
        dcc.Graph(id='histogram')
    ])
])


@app.callback(
    Output('galaxy-name','options'),
    [Input('sample','value')]
)
def updateNamesDropdown(selected_sample):
    sample = samples[selected_sample]
    new_names = sample.data['Object Name']
    names_dict = []
    for name in new_names:
        if type(name) is str:
            names_dict.append({'label': name, 'value': name})
    return names_dict

@app.callback(
    Output('fits-image','figure'),
    [Input('sample','value'),
     Input('galaxy-name','value'),
     Input('sky-map', 'selectedData'),
     Input('band', 'value')]
     #Input('pacs-or-spire', 'value')]
    )
def updateFitsImage(selected_sample, name, selectedName, band):  #add camera as last attribute
    sample = samples[selected_sample]
    index = sample.data.index
    if selectedName == None:
        row = index[sample.data['Object Name'] == name]
        img_path = sample.data[band][sample.data['Object Name'] == name].values[0]
        #img_path = sample.data.query('Object Name==' + name)[band]
        img_data = get_pkg_data_filename(img_path)
        img = fits.getdata(img_data, ext=0)
        #x = np.linspace(-100, 100, img.shape[1])
        #y = np.linspace(-100, 100, img.shape[0])
        fig = px.imshow(img, zmax=img.max())
    else:
        name = selectedName['Object Name']
        row = index[sample.data['Object Name'] == name]
        data = get_pkg_data_filename(sample.data[band][row])
        img = fits.getdata(data, ext=0)
        #x = np.linspace(-100, 100, data.shape[1])
        #y = np.linspace(-100, 100, data.shape[0])
        fig = px.imshow(img, zmax=img.max())
    return fig

@app.callback(
    Output('sky-map', 'figure'),
    [Input('scatter', 'selectedData'),
     Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value'),
     Input('hist-param', 'value'),
     Input('histogram', 'selectedData')]
    )
def update_sky_map(selectedData,xparam,yparam,histparam,histSelectedData):
    event = dash.callback_context.triggered[0]["prop_id"]
    if(selectedData == None and histSelectedData == None):
        fig2 = px.scatter(dustopedia.data, 'RA', 'DEC')
        fig = px.scatter(kingfish.data, 'RA', 'DEC')
        fig.add_trace(fig2.data[0])
        #fig.add_trace(px.scatter(dgs))
    elif((selectedData != None and 'range' in selectedData) or (histSelectedData != None and 'range' in histSelectedData)):
        if(selectedData != None):
            xmin,xmax = selectedData['range']['x'][0:2]
            ymin,ymax = selectedData['range']['y'][0:2]
        else: 
            xmin,xmax = (min(kingfish.data[xparam],dustopedia.data[xparam]),
                         max(kingfish.data[xparam],dustopedia.data[xparam]))
            ymin,ymax = (min(kingfish.data[yparam],dustopedia.data[yparam]),
                         max(kingfish.data[yparam],dustopedia.data[yparam]))
            
        if(histSelectedData != None):
            histmin,histmax = histSelectedData['range']['x'][0:2]
        else:
            histmin,histmax = (min(kingfish.data[histparam],dustopedia.data[histparam]),
                               max(kingfish.data[histparam],dustopedia.data[histparam]))
            
        kingMask = ((kingfish.data[xparam] >= xmin) & 
                (kingfish.data[xparam] < xmax) &
                (kingfish.data[yparam] >= ymin) &
                (kingfish.data[yparam] < ymax) &
                (kingfish.data[histparam] < histmax) &
                (kingfish.data[histparam] >= histmin))
        dustMask = ((dustopedia.data[xparam] >= xmin) & 
                (dustopedia.data[xparam] < xmax) &
                (dustopedia.data[yparam] >= ymin) &
                (dustopedia.data[yparam] < ymax) &
                (dustopedia.data[histparam] < histmax) &
                (dustopedia.data[histparam] >= histmin))
        fig = px.scatter(dustopedia.data[dustMask], 'RA', 'DEC')
        fig.add_trace(px.scatter(kingfish.data[kingMask], 'RA', 'DEC'))
    #elif (selectedDat)
    else:
        xPoints = selectedData['lassoPoints']['x']
        xdata = pd.Series(list(set(xPoints).intersection(set(temp_dust_df[xparam]))))
        yPoints = selectedData['lassoPoints']['y']
        ydata = pd.Series(list(set(yPoints).intersection(set(temp_dust_df[yparam]))))
        #mask = (for(i in range(yPoints)): 
        #            yPoints[i] in temp_dust_df.loc[xparam,xPoints[i]]
        #        )
        fig = []
        
    return fig

@app.callback(
    Output('scatter', 'figure'),
    [Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value')]
    )
def update_scatter(param1, param2):
    fig2 = px.scatter(kingfish.data, x=param1, y=param2, color='Color',
                     color_discrete_sequence=['DarkOrchid'])
    fig = px.scatter(dustopedia.data, x=param1, y=param2,color='Color',
                      color_discrete_sequence=['CornflowerBlue']) 
    fig.add_trace(fig2.data[0])
    return fig

@app.callback(
    Output('histogram', 'figure'),
    Input('hist-param', 'value')
    )
def update_hist(param):
    fig = px.histogram(kingfish.data, param)
    fig2 = px.histogram(dustopedia.data, param)
    fig.add_trace(fig2.data[0])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

