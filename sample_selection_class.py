"""
This is a class to read in the samples, create galaxy objects for each class, 
store options for dropdowns/buttons/radioitems, and hold the callback 
functions for the dash page
"""
import os
import sys
sys.path.insert(0, r"C:\Fall_2020_Wilson_Lab\SN_page\galaxy-selection-page")

from galaxy_class import Sample as gc
import plotly.graph_objs as go
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits, ascii
import numpy as np
import dash
import pandas as pd
import plotly.express as px
import copy

class sample_definition_class():
    # Get Paths to data (csv and fits)
    # TODO: add dgs fits
    def __init__(self):
        self.samples = self.read_in_data()
    
    # Create galaxy objects for each sample
    def read_in_data(self):
        dustopedia_csv = os.path.join('Samples','dustopedia_sample.csv')

        kingfish_csv = os.path.join('Samples','kingfish_sample.csv')
        kingfish_fits = os.listdir("C:\Fall_2020_Wilson_Lab\SN_Page\Samples\Kingfish_FITS\Spire\KINGFISH_SPIRE_v3.0_updated\KINGFISH_SPIRE_v3.0_updated")

        dgs_csv = os.path.join('Samples','dgs_sample.csv')
        dgs_fits = os.listdir("C:\Fall_2020_Wilson_Lab\SN_Page\Samples\DGS_FITS\Renamed_FITs")

        dustopedia = gc('Dustopedia',dustopedia_csv)
        kingfish = gc('Kingfish',kingfish_csv,kingfish_fits)
        dgs = gc('DGS', dgs_csv, dgs_fits)

        samples = {'Kingfish': kingfish, 'DGS': dgs, 'Dustopedia': dustopedia}
        return samples

    # Galaxy names list for fits dropdown
    def get_galaxy_names(self,sampleName='Kingfish'):
        galaxy_names = []
        sample = self.samples[sampleName]
        for name in sample.data['Object Name']:
            if type(name) is str:
                galaxy_names.append({'label': name, 'value': name})
        return galaxy_names
    
    # Sample names list for fits dropdown
    def get_sample_names(self):
        sample_names = [{'label': 'Kingfish', 'value': 'Kingfish'},
                        {'label': 'DGS', 'value': 'DGS'},
                        {'label': 'Dustopedia', 'value': 'Dustopedia'}]
        return sample_names
    
    #Band options for fits dropdown
    # def get_bands(self):
    #     bands = [{'label': '500 band', 'value': '500 band'},
    #             {'label': '350 band', 'value': '350 band'},
    #             {'label': '250 band', 'value': '250 band'}]
    #     return bands
    
    def get_herschel_bands(self):
        bands = [{'label': '500 band', 'value': '500 band'},
                {'label': '350 band', 'value': '350 band'},
                {'label': '250 band', 'value': '250 band'}]
        return bands

    def get_toltec_bands(self):
        bands = [{'label': '2.0 mm', 'value': '2.0 mm'},
                {'label': '1.4 mm', 'value': '1.4 mm'},
                {'label': '1.1 mm', 'value': '1.1 mm'}]
        return bands
    
    def get_sig_unc(self):
        sig_unc = [{'label': 'S/N', 'value': 'S/N'},
                {'label': 'Uncertainty', 'value': 'Uncertainty'}]
        return sig_unc

    # Options for scatterplot and histogram axes dropdowns
    def get_axes_options(self):
        axes_options = [{'label': 'Object Name', 'value': 'Object Name'},
                        {'label': 'RA (deg)', 'value': 'RA'},
                        {'label': 'DEC (deg)', 'value': 'DEC'},
                        {'label': 'Redshift', 'value': 'Redshift'},
                        {'label': 'Distance (Mpc)', 'value': 'Distance'},
                        {'label': 'Log Stellar Mass (M<sub>&#9737;</sub>)', 'value': 'Log Stellar Mass'},
                        {'label': 'Metallicity (12+log(O/H))', 'value': 'Metallicity'},
                        {'label': 'Diameter (arcsec)', 'value': 'Diameter (arcsec)'}]
        return axes_options
    
    #Options for all sky map checkboxes
    # TODO: Add checkboxes to all sky map to determine which samples are displayed
    def get_sky_map_checkboxes(self):
        sky_map_checkboxes = [{'label': 'Dustopedia', 'value': 'Dustopedia'},
                            {'label': 'Kingfish', 'value': 'Kingfish'},
                            {'label': 'DGS', 'value': 'DGS'}]
        return sky_map_checkboxes
    
    # Empty table for fit
    def get_empty_fit_table(self):
        fit_table = pd.DataFrame(
                {
                "Parameter": ["Fit Status", "Temperature (K)", "Scaling Constant",
                            "1.1 Depth (mJy/beam)", "1.4 Depth (mJy/beam)",
                            "2.0 Depth (mJy/beam)"],
                "Value": ['Not Fit', "N/A", "N/A", "N/A", "N/A", "N/A"]
                }
            )
        return fit_table

    def update_sample_dropdown(self, skyClickData):
        event = dash.callback_context.triggered[0]["prop_id"] 
        if ('clickData' in event):  #  and 'graph1' in event
            if (skyClickData['points'][0]['x'] in self.samples['Kingfish'].data['RA'].unique()):
                return 'Kingfish'
            elif (skyClickData['points'][0]['x'] in self.samples['DGS'].data['RA'].unique()):
                return 'DGS'
            else:
                return 'Dustopedia'
        else: return dash.no_update

    def update_galaxy_names_dropdown(self, sampleName, skyClickData):
        event = dash.callback_context.triggered[0]["prop_id"]
        if ('clickData' in event):
            if (skyClickData['points'][0]['x'] in self.samples['Kingfish'].data['RA'].unique()):
                kingfish = self.samples['Kingfish']
                index = kingfish.data.index
                row = index[kingfish.data['RA'] == skyClickData['points'][0]['x']]
                value = kingfish.data['Object Name'][row].values[0]
                new_names = self.get_galaxy_names('Kingfish')
                return new_names, value
            elif (skyClickData['points'][0]['x'] in self.samples['DGS'].data['RA'].unique()):
                dgs = self.samples['DGS']
                index = dgs.data.index
                row = index[dgs.data['RA'] == skyClickData['points'][0]['x']]
                value = dgs.data['Object Name'][row].values[0]
                new_names = self.get_galaxy_names('DGS')
                return new_names, value
            else:
                return [{'label': 'No Dustopedia Fits', 'value': 'dustopedia'}],'No Dustopedia Fits'
        else: 
            if sampleName == 'Kingfish':
                new_names = self.get_galaxy_names(sampleName)
                return new_names, new_names[1]['value'] 
            elif sampleName == 'DGS':
                new_names = self.get_galaxy_names(sampleName)
                return new_names, new_names[1]['value'] 
            else:
                return [{'label': 'No Dustopedia Fits', 'value': 'Dustopedia'}], 'Dustopedia'
        
    # def update_herschel(self, sampleName, galaxyName, herschelBand):  #area_deg2, options, map_type removed for now
        '''
        Update fits figure for selected galaxy and band.
        '''
        # #find current galaxy index
        # index = self.samples[sampleName].data.index
        # row = index[self.samples[sampleName].data['Object Name'] == galaxyName]

        # if galaxyName in ['NGC 6822','NGC 1705'] or sampleName == 'DGS':
        #     pos = 1
        # else:
        #     pos = 0

        # img_path = copy.deepcopy(self.samples[sample].data[band][row].values[0])
        # img = fits.open(img_path, memmap=False)
        # color_label = "Mjy/sr"
        
        # #Change to arcsec centered at (0,0)
        # img_x, img_y = self.center_arcsec(img,pos)
        # sample = self.samples[sampleName]
        # return sample.update_herschel(self, sampleName, galaxyName, herschelBand)

    # def center_arcsec(self, img, pos):
    #     '''
    #     Rescale image to arcseconds
    #     '''
    #     header = img[0].header
    #     img = img[pos].data

    #     nx, ny = img.shape
        
    #     cdelt1 = header['CDELT1']*3600
    #     cdelt2 = header['CDELT2']*3600
        
    #     crpix1 = header['CRPIX1']
    #     crpix2 = header['CRPIX2']

    #     x = (np.array(range(0,nx)) - crpix1)*cdelt1
    #     y = (np.array(range(0,ny)) - crpix2)*cdelt2

    #     return x, y

    def update_herschel_fits_image(self, sampleName, galaxyName, skyClickData, herschelBand, area_deg2, options, event):  
        if 'clickData' in event:   #'dropdown1' not in event and 'dropdown0' not in event
            if (skyClickData['points'][0]['x'] in self.samples['Kingfish'].data['RA'].unique()):
                kingfish = self.samples['Kingfish']
                index = kingfish.data.index
                row = index[kingfish.data['RA'] == skyClickData['points'][0]['x']]
                if (kingfish.data[herschelBand][row].values[0] != None):
                    galaxyName = kingfish.data['Object Name'][row].values[0]
                    color_label, x, y = kingfish.update_herschel('Kingfish', galaxyName, herschelBand)
                    img_path = kingfish.data[herschelBand][row].values[0]
                    img_data = get_pkg_data_filename(img_path)  
                    img = fits.getdata(img_data, ext=0)
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(x=x, y=y, z=img, zmax=img.max(),zmin=img.min(), colorbar = {'title': color_label}))
                    fig.update_xaxes(title='x (arcsec)')
                    fig.update_yaxes(title='y (arcsec)')
                else: return [{}]
            elif (skyClickData['points'][0]['x'] in self.samples['DGS'].data['RA'].unique()):
                dgs = self.samples['DGS']
                index = dgs.data.index
                row = index[dgs.data['RA'] == skyClickData['points'][0]['x']]
                if (dgs.data[herschelBand][row].values[0] != None):
                    galaxyName = dgs.data['Object Name'][row].values[0]
                    color_label, x, y = dgs.update_herschel('DGS', galaxyName, herschelBand)
                    img_path = dgs.data[herschelBand][row].values[0]
                    img_data = get_pkg_data_filename(img_path)  
                    img = fits.getdata(img_data, ext=0)
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(x=x, y=y, z=img, zmax=img.max(),zmin=img.min(), colorbar = {'title': color_label}))
                    fig.update_xaxes(title='x (arcsec)')
                    fig.update_yaxes(title='y (arcsec)')
                else: return [{}]
            else: return [{}]
        else:
            if sampleName == 'Kingfish' or sampleName == 'DGS':
                sample = self.samples[sampleName]
                index = sample.data.index
                row = index[sample.data['Object Name'] == galaxyName]
                if (sample.data[herschelBand][row].values[0] != None):
                    color_label, x, y = sample.update_herschel(sampleName, galaxyName, herschelBand)
                    img_path = sample.data[herschelBand][sample.data['Object Name'] == galaxyName].values[0]
                    img_data = get_pkg_data_filename(img_path)
                    img = fits.getdata(img_data, ext=0)
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(x=x, y=y, z=img, zmax=img.max(),zmin=img.min(), colorbar = {'title': color_label}))
                    fig.update_xaxes(title='x (arcsec)')
                    fig.update_yaxes(title='y (arcsec)')
                else: return [{}]
            else: return [{}]
        if options!=None:
            #Add trace for map area
            if 'show map' in options:
                sx, sy = self.samples[sampleName].setup_box(img, area_deg2)
                fig.add_shape(
                    type="rect",
                    x0=np.min(sx),
                    y0=np.min(sy),
                    x1=np.max(sx),
                    y1=np.max(sy),
                    line=dict(
                        color="Red",
                    ),
                )
        return [fig]
    
    def update_sky_map(self,selectedData,param1,param2,histparam,histSelectedData,event): 
        dustopedia = self.samples['Dustopedia']
        dgs = self.samples['DGS']
        kingfish = self.samples['Kingfish']
        if('dropdown3' in event or 'dropdown2' in event or 'dropdown1' in event):
            return dash.no_update
        elif(selectedData == None and histSelectedData == None):
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
            fig.update_xaxes(showspikes=True)
            fig.update_yaxes(showspikes=True)
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
            fig = go.Figure(data=[go.Scatter(x=self.samples['Dustopedia'].data[dustMask]['RA'], 
                            y=self.samples['Dustopedia'].data[dustMask]['DEC'],
                            mode='markers',
                            marker=dict(color='#FC8D62'),
                            name='Dustopedia')])
            fig.add_trace(go.Scatter(x=self.samples['Kingfish'].data[kingMask]['RA'], 
                            y=self.samples['Kingfish'].data[kingMask]['DEC'],
                            mode='markers',
                            marker=dict(color='#66C2A5'),
                            name='Kingfish'))
            fig.add_trace(go.Scatter(x=self.samples['DGS'].data[dgsMask]['RA'],
                            y=self.samples['DGS'].data[dgsMask]['DEC'],
                            mode='markers',
                            marker=dict(color='#8DA0CB'),
                            name='DGS'))
            fig.update_layout(
                    title='All Sky Map',
                    xaxis_title="RA",
                    yaxis_title="DEC"
                )
            fig.update_xaxes(showspikes=True)
            fig.update_yaxes(showspikes=True)
        return [fig]

    def update_hover_fits(self, hoverData,event):
        herschelBand = '250 band'
        if 'hoverData' in event:
            if (hoverData['points'][0]['x'] in self.samples['Kingfish'].data['RA'].unique()):
                    kingfish = self.samples['Kingfish']
                    index = kingfish.data.index
                    row = index[kingfish.data['RA'] == hoverData['points'][0]['x']]
                    if (kingfish.data[herschelBand][row].values[0] != None):
                        galaxyName = kingfish.data['Object Name'][row].values[0]
                        color_label, x, y = kingfish.update_herschel('Kingfish', galaxyName, herschelBand)
                        img_path = kingfish.data[herschelBand][row].values[0]
                        img_data = get_pkg_data_filename(img_path)  
                        img = fits.getdata(img_data, ext=0)
                        fig = go.Figure()
                        fig.add_trace(go.Heatmap(x=x, y=y, z=img, zmax=img.max(),zmin=img.min(), colorbar = {'title': color_label}))
                        fig.update_xaxes(title='x (arcsec)')
                        fig.update_yaxes(title='y (arcsec)')
                    else: fig = {}
            elif (hoverData['points'][0]['x'] in self.samples['DGS'].data['RA'].unique()):
                dgs = self.samples['DGS']
                index = dgs.data.index
                row = index[dgs.data['RA'] == hoverData['points'][0]['x']]
                if (dgs.data[herschelBand][row].values[0] != None):
                    galaxyName = dgs.data['Object Name'][row].values[0]
                    color_label, x, y = dgs.update_herschel('DGS', galaxyName, herschelBand)
                    img_path = dgs.data[herschelBand][row].values[0]
                    img_data = get_pkg_data_filename(img_path)  
                    img = fits.getdata(img_data, ext=0)
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(x=x, y=y, z=img, zmax=img.max(),zmin=img.min(), colorbar = {'title': color_label}))
                    fig.update_xaxes(title='x (arcsec)')
                    fig.update_yaxes(title='y (arcsec)')
                else: fig = {}
            else:
                fig = {}
        else:
            fig = {}
        return [fig]

    def update_scatter(self, xparam, yparam):
        fig = go.Figure(data=[go.Scatter(x=self.samples['Dustopedia'].data[xparam], 
                        y=self.samples['Dustopedia'].data[yparam], 
                        mode='markers',
                        marker=dict(color='#FC8D62'),
                        name='Dustopedia')])
        fig.add_trace(go.Scatter(x=self.samples['Kingfish'].data[xparam], 
                        y=self.samples['Kingfish'].data[yparam],
                        mode='markers',
                        marker=dict(color='#66C2A5'),
                        name='Kingfish')) 
        fig.add_trace(go.Scatter(x=self.samples['DGS'].data[xparam], 
                        y=self.samples['DGS'].data[yparam],
                        mode='markers',
                        marker=dict(color='#8DA0CB'),
                        name='DGS'))
        fig.update_layout(
                    xaxis_title=xparam,
                    yaxis_title=yparam
                )
        return [fig]

    def update_hist(self,param):
        fig = go.Figure(data=[go.Histogram(x=self.samples['Dustopedia'].data[param],
                        marker=dict(color='#FC8D62'),
                        name='Dustopedia',
                        nbinsx=int(np.sqrt(len(self.samples['Dustopedia'].data[param]))))])
        fig.add_trace(go.Histogram(x=self.samples['Kingfish'].data[param], 
                        marker=dict(color='#66C2A5'),
                        name='Kingfish',
                        nbinsx=int(np.sqrt(len(self.samples['Kingfish'].data[param])))))
        fig.add_trace(go.Histogram(x=self.samples['DGS'].data[param], 
                        marker=dict(color='#8DA0CB'),
                        name='DGS',
                        nbinsx=int(np.sqrt(len(self.samples['DGS'].data[param])))))
        fig.update_layout(barmode='stack',
                        xaxis_title=param )
        return [fig]