#! /usr/bin/env python
"""
Created on Wed May 26 10:42:52 2021

@author: Caleigh Ryan

Create class to hold header, header data you need in functions, 
pass to get flux (make wcs for all 3 bands in getFlux())

Convert pix on all 3 to Ra/Dec (allpic2world), convert back to pixels for each band (different)
"""
import plotly.graph_objs as go
import os
import dash

import sys
sys.path.insert(0, r"C:\Fall_2020_Wilson_Lab\SN_page\galaxy-selection-page")

from sample_selection_class import sample_definition_class

from dash.dependencies import Output, Input
from dash import no_update, exceptions
import functools
# This import grabs the main class setup components
from dasha.web.templates import ComponentTemplate
from dasha.web.extensions.cache import cache
from tollan.utils.log import timeit, get_logger

# We get the usual dash bootstrap components in a different way in dasha
import dash_bootstrap_components as dbc
import dash_core_components as dcc

from dasha.web.templates.common import LiveUpdateSection
import dash_html_components as html
# This is the main class that includes everything.  All code should be ran
# from inside this class.  The name is yours to choose.

class sample_definition_page(ComponentTemplate):
    _component_cls = dbc.Container

    # fluid controls if the components will resize and rearrange when the
    # window is resized.
    fluid = True
    logger = get_logger()
    
    sc = sample_definition_class()

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    # Set up all of your components and call-backs in this function.  You can
    # also define other functions and call them in here.  Note that it takes
    # the dash 'app' as input for the call-backs.
    def setup_layout(self, app):
        app.config.external_stylesheets.append(dbc.themes.LUX)
        container = self

        # Create two rows for header and body
        header_container, body = container.grid(2,1)
        
        # Create the header as a child of the header_container.
        header = header_container.child(LiveUpdateSection(
            title_component=html.H2("Galaxy Selection",
                                    style={
                                        'margin-top': '50px',
                                        'margin-bottom': '50px'
                                        }),
            interval_options=[2000, 5000],
            interval_option_value=5000))

        # Row containing the sample, galaxy, herschel band, toltec band, map type dropdowns
        fits_dropdowns_container = body.child(dbc.Row, 
                                    style={
                                    'margin-top': '50px',
                                    'margin-bottom': '50px'
                                    })
        
        sample_dropdown = fits_dropdowns_container.child(dbc.Col, style={'display': 'inline-block', 'width': '20%'}).child(
                                    dcc.Dropdown,
                                    style={})
        sample_dropdown.options = self.sc.get_sample_names()
        sample_dropdown.value = 'Kingfish'

        galaxy_dropdown = fits_dropdowns_container.child(dbc.Col, style={'display': 'inline-block', 'width': '20%'}).child(
                                    dcc.Dropdown,
                                    style={})
        galaxy_dropdown.options = self.sc.get_galaxy_names()
        galaxy_dropdown.value = self.sc.get_galaxy_names()[1]['value']

        herschel_band_dropdown = fits_dropdowns_container.child(dbc.Col, style={'display': 'inline-block', 'width': '20%'}).child(
                                    dcc.Dropdown,
                                    style={})
        herschel_band_dropdown.options = self.sc.get_herschel_bands()
        herschel_band_dropdown.value = '250 band'

        toltec_band_dropdown = fits_dropdowns_container.child(dbc.Col, style={'display': 'inline-block', 'width': '20%'}).child(
                                    dcc.Dropdown,
                                    style={})
        toltec_band_dropdown.options = self.sc.get_toltec_bands()
        toltec_band_dropdown.value = '1.1 mm'

        signal_uncertainty_button = fits_dropdowns_container.child(dbc.Col, style={'display': 'inline-block', 'width': '20%'}).child(
                                    dcc.RadioItems,
                                    style={
                                    'width': '35%'
                                    })
        signal_uncertainty_button.options = self.sc.get_sig_unc()
        signal_uncertainty_button.value = 'S/N'

        # Row containing herschel and toltec fits images
        herschel_toltec_fits = body.child(dbc.Row, 
                                    style={})

        herschel_fits_image = herschel_toltec_fits.child(dbc.Col, 
                                    style = {'width': '50%', 'display': 'inline-block'}).child(
                                        dcc.Graph, figure=go.Figure(),
                                        style={
                                        # 'width': '34%',
                                        # 'margin-left': '30%'
                                    })
        
        toltec_fits_image = herschel_toltec_fits.child(dbc.Col, 
                                    style = {'width': '50%', 'display': 'inline-block'}).child(
                                        dcc.Graph, figure=go.Figure(),
                                        style={
                                        # 'width': '34%',
                                        # 'margin-left': '30%'
                                    })
        
        # Row containing fits mapping options
        options_checklist = body.child(dbc.Row).child(dbc.Checklist,
                                                       options=[
                {"label": "Show Map Area", "value": "show map"},
                {"label": "Show Array", "value": "show array"},
                {"label": "Log Scale", "value": "log_scale"}
                ],
                inline=True,
                style={"margin-right": "0px",
                    "margin-left": "450px",
                    "margin-bottom": "50"})
        
        # Row containing fitting options
        fitting_inputs = body.child(dbc.Row,
                                    style={
                                    'margin-top': '50px',
                                    'margin-bottom': '50px',
                                    'width': '100%'
                                    }).child(dbc.InputGroup, size='sm',
                                    style={
                                    "margin-right": "20px",
                                    "margin-left": "20px"
                                    })

        
        #Change integration time
        fitting_inputs.child(dbc.InputGroupAddon("Time (hrs)", addon_type="prepend"))
        fit_time_input = fitting_inputs.child(dbc.Input, value=1.0)

        #Change map area
        fitting_inputs.child(dbc.InputGroupAddon("Area (deg^2)", addon_type="prepend"))
        fit_area_input = fitting_inputs.child(dbc.Input, value=0.1)

        #Change atmosphere factor (1-7)
        fitting_inputs.child(dbc.InputGroupAddon("Atm Factor (1-7)", addon_type="prepend"))
        fit_atm_input = fitting_inputs.child(dbc.Input, value=1.0)

        #Change spectral index for greybody
        fitting_inputs.child(dbc.InputGroupAddon("Spectral Index", addon_type="prepend"))
        fit_beta_input = fitting_inputs.child(dbc.Input, value=2.0)

        # Container for the row containing the flux fit and table
        flux_fit_and_data_table = body.child(dbc.Row,
                                    style={
                                    'margin-top': '50px',
                                    'margin-bottom': '50px',
                                    'width': '100%'
                                    })
        
        flux_fit = flux_fit_and_data_table.child(dbc.Col, style={'width': '50%', 'display': 'inline-block'}).child(dbc.Col).child(    
                                    dcc.Graph, figure=go.Figure(),
                                    style={})
        tbl = dbc.Table.from_dataframe(self.sc.get_empty_fit_table())
        flux_fit_table = flux_fit_and_data_table.child(dbc.Col, style={'width': '50%', 'display': 'inline-block'}).child(dbc.Col).child(    
                                    dbc.Table, tbl,
                                    striped=True,
                                    bordered=True,
                                    hover=True
                                )
        
        # Container for the row containing the all sky map
        all_sky_map_row = body.child(dbc.Row,
                                    style={
                                    'margin-top': '50px',
                                    'margin-bottom': '50px',
                                    'width': '100%'
                                    })
        
        #Container for row containing all sky map and hover preview image
        all_sky_map = all_sky_map_row.child(dbc.Col, style={'width': '60%', 'display':'inline-block'}).child(dbc.Col).child(   
                                    dcc.Graph, figure=go.Figure(), config={'modeBarButtonsToRemove': ['lasso2d']},
                                    style={})
        hover_fits_image = all_sky_map_row.child(dbc.Col, style={'width': '30%', 'display': 'inline-block'}).child(dbc.Col).child(    
                                    dcc.Graph, figure=go.Figure(),
                                    style={})

        # Row containing scatter and histogram parameter dropdowns
        scatter_hist_param = body.child(dbc.Row,
                                    style={
                                    'margin-top': '50px',
                                    'margin-bottom': '50px',
                                    'width': '100%'
                                    })
        
        scatter_button_container = scatter_hist_param.child(dbc.Col,style={'display': 'inline-block', 'width': '50%'}).child(dbc.Row)

        xparam_dropdown = scatter_button_container.child(dbc.Col, style={'display': 'inline-block', 'width': '50%'}).child(
                                    dcc.Dropdown,
                                    style={})
        xparam_dropdown.options = self.sc.get_axes_options()
        xparam_dropdown.value = 'RA'

        yparam_dropdown = scatter_button_container.child(dbc.Col, style={'width':'50%','display': 'inline-block'}).child(
                                    dcc.Dropdown,
                                    style={})
        yparam_dropdown.options = self.sc.get_axes_options()
        yparam_dropdown.value = 'DEC'

        hist_button_container = scatter_hist_param.child(dbc.Col, style={'display': 'inline-block', 'width': '50%'})
                                    
        histparam_dropdown = hist_button_container.child(dbc.Row).child(dbc.Col).child(
                                    dcc.Dropdown, 
                                    style={})
        histparam_dropdown.options = self.sc.get_axes_options()
        histparam_dropdown.value = 'Redshift'

        # Row containing scatter plot and histogram
        scatter_hist_graphs = body.child(dbc.Row,
                                    style={
                                    'margin-top': '50px',
                                    'margin-bottom': '50px',
                                    'width': '100%'
                                    })

        scatter_plot_container = scatter_hist_graphs.child(dbc.Col,
                                    style={
                                    'display': 'inline-block',
                                    'width': '50%'
                                    })
        
        scatter_plot = scatter_plot_container.child(dbc.Row).child(
                                    dcc.Graph, figure=go.Figure(),
                                    config={'modeBarButtonsToRemove': ['lasso2d']},
                                    style={})

        
        hist_container = scatter_hist_graphs.child(dbc.Col,
                                    style={
                                    'display': 'inline-block',
                                    'width': '50%'
                                    })
        hist_plot = hist_container.child(dbc.Row).child(dbc.Col).child(
                                    dcc.Graph, figure=go.Figure(),
                                    config={'modeBarButtonsToRemove': ['lasso2d']},
                                    style={})

        
        # Callback functions to update fits image and dropdowns
        def update_sample_name_dropdown(skyClickData):
            """
            Takes in clickData from all sky map and updates sample in dropdown.
            """
            sample = self.sc.update_sample_dropdown(skyClickData)
            return sample
        app.callback(
            Output(sample_dropdown.id, 'value'),
            Input(all_sky_map.id, 'clickData')
            )(functools.partial(update_sample_name_dropdown))

        def update_galaxy_names_dropdown(sampleName, skyClickData):
            """
            Uses input from either the sample dropdown or all sky map clickData
            to update the list of galaxy names. 
            """
            new_names, galaxyValue = self.sc.update_galaxy_names_dropdown(sampleName,skyClickData)
            return new_names, galaxyValue
        app.callback(
            [Output(galaxy_dropdown.id,'options'),
            Output(galaxy_dropdown.id, 'value')],
            [Input(sample_dropdown.id,'value'), 
            Input(all_sky_map.id,'clickData')]
            )(functools.partial(update_galaxy_names_dropdown))

        def update_herschel_fits_image(sampleName, galaxyName, skyClickData, herschelBand, area_deg2, options):
            """
            Uses the selected sample and galaxy names to update the herschel image. If 
            the event includes clickData, it will use the selected herschel band to update
            the image. It will also add a box for the map area if selected.
            """
            event = dash.callback_context.triggered[0]['prop_id']
            figure = self.sc.update_herschel_fits_image(sampleName, galaxyName, skyClickData, herschelBand, area_deg2, options, event)
            return figure
        app.callback(
            [Output(herschel_fits_image.id, "figure")],
            [Input(sample_dropdown.id, "value"),
             Input(galaxy_dropdown.id, "value"),
             Input(all_sky_map.id, "clickData"),
             Input(herschel_band_dropdown.id, "value"),
             Input(fit_area_input.id, "value"),
             Input(options_checklist.id, "value"),
             ] # Input(herschel_map_select.id, "value") this determines signal or uncertainty map
            )(functools.partial(update_herschel_fits_image))

        def update_flux_fit(galaxyName, herschelBand, herschelClickData, sampleName, toltecBand, time, area_deg2, atm, beta, options):
            """
            Takes all imput parameters and outputs the flux fit figure, flux fit 
            table, and toltec image.
            """
            sample = self.sc.samples[sampleName]
            event = dash.callback_context.triggered[0]['prop_id']
            print(event)
            if 'clickData' in event or 'dropdown3' in event or 'input' in event:
                fit_fig, toltec_fits_image, fit_table = sample.update_fit(galaxyName, sampleName, herschelBand, herschelClickData, sample, toltecBand, time, area_deg2, atm, beta, options)
                return [fit_fig, toltec_fits_image, fit_table]
            else:
                return [dash.no_update, dash.no_update, dash.no_update]
        app.callback(
            [Output(flux_fit.id, "figure"),
            Output(toltec_fits_image.id, "figure"),
            Output(flux_fit_table.id, 'children')],
            [Input(galaxy_dropdown.id, "value"),
            Input(herschel_band_dropdown.id, "value"),
            Input(herschel_fits_image.id, "clickData"),
            Input(sample_dropdown.id, 'value'),
            Input(toltec_band_dropdown.id, "value"),
            Input(fit_time_input.id, "value"),
            Input(fit_area_input.id, "value"),
            Input(fit_atm_input.id, "value"),
            Input(fit_beta_input.id, "value"),
            Input(options_checklist.id, "value")]
            )(functools.partial(update_flux_fit))

        def update_scatter(xparam, yparam):
            """
            Changes x and y axes values based on dropdowns.
            """
            figure = self.sc.update_scatter(xparam, yparam)
            return figure  
        app.callback(
            [Output(scatter_plot.id, "figure")],
            [Input(xparam_dropdown.id, "value"),
             Input(yparam_dropdown.id, "value")]
            )(functools.partial(update_scatter))

        def update_hist(param):
            """
            Updates histogram exis based on dropdown.
            """
            figure = self.sc.update_hist(param)
            return figure 
        app.callback(
            [Output(hist_plot.id, "figure")],
            [Input(histparam_dropdown.id, "value")]
            )(functools.partial(update_hist))

        def update_sky_map(selectedData,param1,param2,histparam,histSelectedData):
            """
            Filters the all sky map accorning to the selected data from the
            scatter plot and histogram.
            """
            event = dash.callback_context.triggered[0]["prop_id"] 
            figure = self.sc.update_sky_map(selectedData,param1,param2,histparam,histSelectedData,event)
            return figure
        app.callback(
            [Output(all_sky_map.id, "figure")],
            [Input(scatter_plot.id, "selectedData"),
             Input(xparam_dropdown.id, "value"),
             Input(yparam_dropdown.id, "value"),
             Input(histparam_dropdown.id, "value"),
             Input(hist_plot.id, "selectedData")]
            )(functools.partial(update_sky_map))
        
        def update_hover_fits(hoverData):
            """
            Updates the small hover fits image based on point hovered over
            on the all sky map.
            """
            event = dash.callback_context.triggered[0]["prop_id"] 
            figure = self.sc.update_hover_fits(hoverData,event)
            return figure 
        app.callback(
            [Output(hover_fits_image.id, "figure")],
            [Input(all_sky_map.id, "hoverData")
            ])(functools.partial(update_hover_fits))
        

# This declares and runs the above class.  Change the name to match the class
# above
extensions = [
{
    'module': 'dasha.web.extensions.dasha',
    'config': {
        'template': sample_definition_page,
        'title_text': 'Sample Selection',
        }
    },
]