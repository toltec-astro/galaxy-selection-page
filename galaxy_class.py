# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:42:18 2021

@author: Caleigh Ryan

Definition of class to create objects containing images and other relevant 
information for Herschel DGS and KINGFISH galaxies. It creates an astropy table 
for all of these elements, and includes functions to act on them for 
necessary unit conversions. It also includes the necessary functions for fitting
the TolTEC bands and creating the signal to noise map.
"""
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits, ascii
from astropy import units as u
from astroquery.ned import Ned
import pandas as pd
from astropy.wcs import WCS
import urllib.parse
import os
import numpy as np
import copy
import plotly.express as px
import plotly.graph_objs as go
import dash
import dash_bootstrap_components as dbc
from lmfit import Model, Parameter, report_fit
from .TolTEC import TolTEC

from pathlib import Path


datadir = Path(__file__).parent.joinpath('data').as_posix()


class Sample:
    def __init__(self, sampleName, csv, fits=None):
        self.sampleName = sampleName
        self.data = pd.read_csv(csv)

        self.w_micron = {'500 band': 500, '350 band': 350, '250 band': 250}
        self.w_toltec_mm = {'1.1 mm': 1.1, '1.4 mm': 1.4, '2.0 mm': 2.0}

        self.herschel_bands = ['500 band', '350 band', '250 band']

        self.toltec_bands = [1.1, 1.4, 2.0]
        self.toltec_map_from_index = {1.1: 0, 1.4: 1, 2.0: 2}
        self.herschel_index_from_w = {500: 0, 350: 1, 250: 2}

        # Beam sizes from equation Michael sent : beamsize = 2pi*(FWHM[arcsec]/2)^2 for each band
        # They are in arcseconds^2
        self.spire_beam_sizes = {'250 band': 2*np.pi*(17.6/2)**2, '350 band': 2*np.pi*(23.9/2)**2, '500 band': 2*np.pi*(35.2/2)**2}
        self.toltec_beam_sizes = {1.1: 2*np.pi*(5.0/2)**2, 1.4: 2*np.pi*(6.3/2)**2, 2.0: 2*np.pi*(9.5/2)**2}

        #unit conversions
        self.deg2_per_sr = 3282.8
        self.c = 3e8  # m/s
        self.arcsec_2 = 3600**2
        self.Mjy_sr_to_Jy_arcsec2 = 1e6/self.deg2_per_sr/self.arcsec_2
        self.w_micron_to_nu = self.c*1e6  # microm/s
        self.w_mm_to_nu = self.c*1e3  # mm/s
        self.Jy_arcsec2_to_erg_sr = 1e-23*self.arcsec_2*self.deg2_per_sr
        # self.Jy_to_cgs = u.Jy.to_system(u.cgs) 
        self.Jy_to_cgs = 10**-23 # g/s^2
        
        #Initial guess for fits
        self.T_guess = 50.0
        self.Const_guess = 25 # 10**23 # 1000000.0#e-4
        
        #Default fit status
        self.fit_status = 'Not Fit'

        if fits is None:
             fits = list()
        self.fits = list(fits)
        self.add_column('250 band')
        self.add_column('350 band')
        self.add_column('500 band')
        self.fixed_fits = self.fix_FITS_names(sampleName)
        self.add_column('Color')
        self.add_column('Fits Paths')
        self.add_column('Fixed Name Fits')
        
        for i in range(len(self.data['Object Name'])):
            self.add_data('Color',i,1)
            
        if(len(fits) > 0):   
            for i, row in self.data.iterrows():
                galaxyName = row["Object Name"]
                fixed_name_fits, fits_path = self.get_galaxy_fits(galaxyName, sampleName)
                self.data.at[i, "Fits Paths"] = fits_path
                self.data.at[i, "Fixed Name Fits"] = fixed_name_fits
                self.loadSpire(fits_path, fixed_name_fits, galaxyName, sampleName)
                
    # Change FITS file names to NED names and removes path which is not to a fits file
    def fix_FITS_names(self, sampleName):
        fits = []
        if sampleName == 'Kingfish':
            for i in self.fits[1:]:
                # galaxyName = i.split('_')[0]
                # newName = Ned.query_object(galaxyName)['Object Name'][0]
                # newNameFixed = newName.split(' ')[0]
                # for index in range(1,len(newName.split(' '))):
                #     newNameFixed = newName + '_' + newName.split(' ')[index]
                # fits.append(newName + '_' + i.split('_')[1] + '_' + i.split('_')[2] + '_' + i.split('_')[3] + '_' + i.split('_')[4])
                try:
                     i = os.path.basename(i)
                     name = i.split('_')[0]
                     newName = Ned.query_object(name)['Object Name'][0]
                     newNameFixed = newName.split(' ')[0]
                     for index in range(1,len(newName.split(' '))):
                         newNameFixed = newName + '_' + newName.split(' ')[index]
                     fits.append(newName + '_' + i.split('_')[1] + '_' +
                             i.split('_')[2] + '_' + i.split('_')[3] + '_' +
                             i.split('_')[4])
                except Exception:
                     print(f'unable to fix name for {i}')
                     pass
        elif sampleName == 'DGS':
            for i in self.fits:
                # galaxyName = i.split('_')[0]
                # newName = Ned.query_object(galaxyName)['Object Name'][0]
                # newNameFixed = newName.split(' ')[0]
                # for index in range(1,len(newName.split(' '))):
                #     newNameFixed = newName + '_' + newName.split(' ')[index]
                # fits.append(newName + '_' + i.split('_')[1] + '_' + i.split('_')[2])
                i = os.path.basename(i)
                try:
                    name = i.split('_')[0]
                    newName = Ned.query_object(name)['Object Name'][0]
                    newNameFixed = newName.split(' ')[0]
                    for index in range(1,len(newName.split(' '))):
                        newNameFixed = newName + '_' + newName.split(' ')[index]
                    fits.append(newName + '_' + i.split('_')[1] + '_' +
                            i.split('_')[2])
                except Exception:
                    print(f'unable to fix name for {i}')
                    pass
        return fits
            
    # Adds new columns
    def add_column(self,colname):
        self.data[colname] = [None for _ in range(len(self.data['Object Name']))]
        
    # Adds data to column
    def add_data(self,column,row,data):
        self.data.loc[row,[column]] = data
        
    def get_galaxy_fits(self,galaxyName,sampleName):
        fits = []
        old_fits = []
        if sampleName == 'Kingfish':
            for i in range(len(self.fixed_fits)):
                if (str(galaxyName) in self.fixed_fits[i]):
                    fits.append(self.fixed_fits[i])
                    old_fits.append(self.fits[i+1])
        elif sampleName == 'DGS':
            for i in range(len(self.fixed_fits)):
                if (str(galaxyName) in self.fixed_fits[i]):
                    fits.append(self.fixed_fits[i])
                    old_fits.append(self.fits[i])
        return fits, old_fits
    
    # Correctly reads in all associated FITs for Spire galaxy
    def loadSpire(self,fits_paths, fixed_name_fits, galaxyName, sampleName):     
        hdu = 0
        if sampleName == 'Kingfish':
            # path = os.path.join('Samples/Kingfish_FITS/Spire/KINGFISH_SPIRE_v3.0_updated/KINGFISH_SPIRE_v3.0_updated')
            path = os.path.join(datadir, 'Samples/Kingfish_FITS/Spire/KINGFISH_SPIRE_v3.0_updated_updated/KINGFISH_SPIRE_v3.0_updated_updated')
        elif sampleName == 'DGS':
            # path = os.path.join("Samples/DGS_FITS/Renamed_FITs")
            path = os.path.join(datadir, "Samples/DGS_FITS/Renamed_FITs")
        for i in fits_paths:
            if(('spire250' in i) and ('scan.fits' in i)):
                self.add_data('250 band',self.data['Object Name'] == galaxyName,os.path.join(path, i))
            elif(('spire350' in i) and ('scan.fits' in i)):
                self.add_data('350 band',self.data['Object Name'] == galaxyName,os.path.join(path, i))
            elif(('spire500' in i) and ('scan.fits' in i)):
                self.add_data('500 band',self.data['Object Name'] == galaxyName,os.path.join(path, i))
    
    #Correctly reads in all associated FITs for PACS galaxy
    # def loadPacs(self):
    #     pass
    
    def center_arcsec(self, img, pos):
        '''
        Rescale image to arcseconds
        '''
        try:
            header = img[0].header
            data = img[1].data
            nx, ny = data.shape
        except:
            header = img[0].header
            data = img[0].data
            nx, ny = data.shape

        # header = img[pos].header
        # img = img[pos].data
        # nx, ny = img.shape
        
        cdelt1 = np.abs(header['cdelt1'])*3600 # np.abs
        cdelt2 = np.abs(header['cdelt2'])*3600 # np.abs
        # pixel_size = header['pfov']  # arcseconds, DGS don't have this

        # ny = header['naxis1']
        # nx = header['naxis2']
        
        crpix1 = header['crpix1']
        crpix2 = header['crpix2']

        x = (np.arange(nx) - crpix2)/cdelt2
        y = (np.arange(ny) - crpix1)/cdelt1

        return x, y

    def update_herschel(self, sampleName, galaxyName, herschelBand):  #area_deg2, options, map_type removed for now
        '''
        Update fits figure for selected galaxy and band.
        '''
        #find current galaxy index
        row = self.data['Object Name'] == galaxyName

        if galaxyName in ['NGC 6822','NGC 1705']:
            pos = 1
        elif sampleName == 'DGS':
            pos = 1
        else:
            pos = 0

        img_path = copy.deepcopy(self.data[herschelBand][row].values[0])
        img = fits.open(img_path, memmap=False)
        color_label = "Mjy/sr"
        
        #Change to arcsec centered at (0,0)
        img_x, img_y = self.center_arcsec(img, pos)
        return color_label, img_y, img_x

    def get_flux(self, x, y, herschelBand, galaxyName, sampleName):
        '''
        Get's flux for clicked pixel (x,y) which is in the units of the
        map
        '''
        row = self.data['Object Name'] == galaxyName

        img_path = copy.deepcopy(self.data[herschelBand][row].values[0])
        img = fits.open(img_path, memmap=False)

        try:
            pos = 0
            header = img[0].header
            wcs = WCS(header)
            testx,testy = img[pos].shape
        except:
            pos = 1
            header = img[0].header
            wcs = WCS(header)
            testx,testy = img[pos].shape
        # if name in ['NGC6822','NGC1705']:
        #     pos = 1
        # elif sampleName == 'DGS':
        #     pos = 1
        # else:
        #     pos = 0

        # header = img[pos].header
        # wcs = WCS(header)
        # cdelt1 = header['cdelt1']*3600
        # cdelt2 = header['cdelt2']*3600
        
        crpix1 = header['crpix1']
        crpix2 = header['crpix2']
        
        cdelt1 = 3600.*np.abs(header['cdelt1']) # np.abs(
        cdelt2 = 3600.*np.abs(header['cdelt2']) # np.abs(
        # pixel_size = header['pfov']  # arcseconds
        
        # crpix1 = header['crpix1']
        # crpix2 = header['crpix2']

        #Needs to rescale back to pixels
        sx = x*cdelt2 + crpix2
        sy = y*cdelt1 + crpix1
        
        #Use WCS to get ra and dec
        ra, dec = wcs.all_pix2world(sx, sy, 0, ra_dec_order = True)

        fluxes = []
        # error_fluxes = []

        #Loop through images, converting RA and Dec to pixel positon for each map and storing the values
        for i in [self.data['500 band'][row], self.data['350 band'][row], self.data['250 band'][row]]:
            img = fits.open(i.values[0],memmap=False)
            wcs = WCS(img[0].header)
            py, px = wcs.all_world2pix(ra, dec, 0, ra_dec_order = True)
            # epx, epy = wcs.all_world2pix(self.era, self.edec, 0)

            py = int(np.round(py))
            px = int(np.round(px))
            print(f'PX {px}')
            print(f'PY {py}')

            # epx = int(np.round(epx))
            # epy = int(np.round(epy))

            # data indexed by row column
            fluxes.append(img[pos].data[py,px])
            # error_fluxes[i] = self.unc[i]['img'][epx, epy]
        return fluxes

    def bb(self, f, T):
        '''
        #Blackbody function (testing)
        '''
        c = 3e10 # cm s-1
        kb = 1.38e-16 # cm^2 g s^-2 K^-1
        h = 6.626e-27 # cm^2 g s^-1
        
        nume = (2*h*f**3)
        denom = c**2
        factor = (h*f)/(kb*T)
        result = (nume/denom)*(np.exp(factor) - 1)**-1 
        return result # return in cgs units

    def greybody(self, f, T, N_H2):
        '''
        Greybody function.  Uses self.beta
        '''
        beta = self.beta
        # nu = self.w_micron_to_nu/self.w_micron[herschelBand] # Hz
        # nu = f # Hz

        k0 = 1.37 # cm^2/g
        nu0 = 3*10**14/1000  # Hz 
        mu = 2.8
        mh = 1.67*10**(-21) # g

        tau = k0*((f/nu0)**self.beta)*mu*mh*(10**N_H2)*0.01 # cm^2 * N_H2 for units, should work out to be unitless
        tau = tau.astype(float)
        # return in MJy/sr, and ignore omega part of equation
        #return (1-np.exp(-tau))*self.bb(f,T)
        temp = (1-np.exp(-tau))*self.bb(f,T)
        mbb = temp*(10**23)/1e6
        try:
            return mbb.astype(float)
        except:
            return mbb

    def fit(self, x, y, herschelBand, galaxyName, sampleName, time_hours, area_deg2, atmFactor, beta):
        '''
        Get flux and fit it with greybody
        '''
        self.beta = beta
        self.herschelBand = herschelBand
        map_flux = self.get_flux(x, y, herschelBand, galaxyName, sampleName)
        #convert flux from MJy/sr to cgs as model input [500, 350, 250]
        # flux = (np.array(map_flux)*10**6)*self.Jy_to_cgs
        freq = np.array([self.w_micron_to_nu/self.w_micron[i] for i in ['500 band', '350 band', '250 band']], dtype=float)
        # Make it so numpy warnings will raise errors & get caught by try except
        np.seterr(invalid='raise')
        print(f'MAP FLUXES = {map_flux}')
        try:
            pmod = Model(self.greybody,independent_vars=['f'] ,
                                            T=Parameter('T',value=self.T_guess, min=5, max=100),
                                            N_H2=Parameter('N_H2',value=self.Const_guess, min=10,max=30), #, min=1e-8
                                            )

            result = pmod.fit(data=map_flux, 
                            f=freq, method='differential_evolution'
                            )

            T_fit = result.values['T']
            const_fit = result.values['N_H2']

            #Make arrays with fitted parameters for line on plot
            BB, nu_fit = self.make_planck_from_fit(T_fit,const_fit, herschelBand, beta)

            #Get TolTEC fluxes with fitted parameters
            toltec_fit_fluxes = self.calc_toltec_from_fit(T_fit,const_fit,herschelBand,beta)

            # Test taking the log of fluxes to catch error for plotting later
            np.log10(BB.astype(float))
            np.log10(toltec_fit_fluxes)
            np.log10(map_flux)

            self.fit_status = 'Success'
            return BB, toltec_fit_fluxes, map_flux, nu_fit, T_fit, const_fit

        except FloatingPointError:
            self.fit_status = 'Failure'
            return None, None, None, None, None, None
        except:
            self.fit_status = 'Failure'
            return None, None, None, None, None, None

# TODO: get nu_min and nu_max the way Michael does
    def make_planck_from_fit(self, T_fit, const_fit, herschelBand, beta, npts=50):
        '''
        Make arrays with fitted parameters for line on plot
        '''
        # nu_min = 6.0e11
        # nu_max = 1.2e12
        # nu_min = 1.0e11
        # nu_max = 1.2e12
        nu_min = self.w_mm_to_nu/2.0
        nu_max = self.w_mm_to_nu/0.25

        self.beta = beta
        self.herschelBand = herschelBand

        nu_fit = np.linspace(nu_min, nu_max, num=npts)
        BB = self.greybody(nu_fit, T_fit, const_fit)
        return BB, nu_fit

    def calc_toltec_from_fit(self, T_fit, const_fit, herschelBand, beta):
        '''
        Get TolTEC fluxes with fitted parameters
        '''
        self.beta = beta
        self.herschelBand = herschelBand

        flux11 = self.greybody(1/1.1*self.w_mm_to_nu, T_fit, const_fit)
        flux14 = self.greybody(1/1.4*self.w_mm_to_nu, T_fit, const_fit)
        flux20 = self.greybody(1/2.0*self.w_mm_to_nu, T_fit, const_fit)

        toltec_fit_fluxes = [flux11, flux14, flux20]
        return toltec_fit_fluxes
    
    def get_depth(self, area_deg2, time_hours, atmFactor=1.0):
        '''
        Use TolTEC class from TolTEC.py to calculate depth based on
        selected area, integration time, and atmFactor.  In units of
        mJy/beam where beam is the TolTEC beams.
        '''
        toltec = TolTEC(atmFactor)

        depth = toltec.depth_mJy(area_deg2, time_hours)
        return depth
    
    def setup_box(self, img, area_deg2):
        '''
        Create x and y arrays for map box area in arcseconds
        '''
        box_side_x = np.sqrt(area_deg2)*3600#/self.obs[self.indx].data[w]['cdelt1']
        box_side_y = np.sqrt(area_deg2)*3600#/self.obs[self.indx].data[w]['cdelt2']

        # Why do we get cx, cy here if we just set them to zero in the next step?
        try:
            cx = img[1].header['RA']
            cy = img[1].header['DEC']
        except:
            cx = img[0].header['RA']
            cy = img[0].header['DEC']

        cx, cy = 0,0#self.obs[self.indx].data[w]['wcs'].all_world2pix(cx/3600,cy/3600,0)

        sx = [cx - box_side_x/2, cx - box_side_x/2, cx + box_side_x/2,
              cx + box_side_x/2, cx - box_side_x/2]
        sy = [cy - box_side_y/2, cy + box_side_y/2, cy + box_side_y/2,
              cy - box_side_y/2, cy - box_side_y/2]
        return sx, sy

    def update_fit(self, galaxyName, sampleName, herschelBand, herschelClickData, sample, toltecBand, time_hours, area_deg2, atmFactor, beta, options):   # , map_type
        '''
        Updates fit figure, toltec figure, and table based on inputs from
        Dash page
        '''
        #Make sure wavlength is a float
        w = float(self.w_micron[herschelBand])
        
        #Get index for galaxy
        row = self.data['Object Name'] == galaxyName

        #Make sure toltec wavelength is a float
        tw = float(self.w_toltec_mm[toltecBand])
        # tw = np.where(np.array(self.config['meta']['TolTEC']['w']) == tw)[0][0]

        #Get clicked position.  Return no_update if nothing clicked to prevent errors
        try:
            x = herschelClickData['points'][0]['x']
            y = herschelClickData['points'][0]['y']
        except:
            return [dash.no_update, dash.no_update, dash.no_update]
        
        if galaxyName in ['NGC 6822','NGC 1705']:
                pos = 1
        elif sampleName == 'DGS':
            pos = 1
        else:
            pos = 0
        
        img_path = copy.deepcopy(self.data[herschelBand][row].values[0])
        img = fits.open(img_path, memmap=False)

        #Convert input params to floats
        time_hours = float(time_hours)
        area_deg2 = float(area_deg2)
        atmFactor = float(atmFactor)
        beta = float(beta)

        #Fit the currently selected pixel
        # self.obs[self.indx].fit(x, y, w, time_hours, area_deg2, atmFactor, beta)
        BB, toltec_fit_fluxes, map_flux, nu_fit, T_fit, const_fit = self.fit(x, y, herschelBand, galaxyName, sampleName, time_hours, area_deg2, atmFactor, beta)

        #Create fit figure
        np.seterr(divide='raise')
        fit_fig = go.Figure()
        if self.fit_status == 'Success':
            #Make TolTEC Map (not plotted)
            toltec_maps, header = self.make_toltec_map(w, toltec_fit_fluxes, map_flux, row, herschelBand, pos)

            #Make TolTEc S/N map
            toltec_snr_maps, depth = self.make_toltec_snr_map(toltec_maps, area_deg2, time_hours, row, herschelBand, pos, atmFactor)

            #Switch to log scale
            fit_fig.update_yaxes(title='log10(MJy/sr)')  # , type='log'

            #Add trace for Herschel points.  Use uncertainty from unc map for error bars
            fit_fig.add_trace(go.Scatter(x=np.array([500,350,250]),
                        y=np.log10(map_flux),
                        mode='markers',
                        name='Herschel'))
                        # ,
                        # error_y=dict(
                        #       type='data',
                        #       symmetric=True,
                        #       array=self.obs[self.indx].error_fluxes_list,
                        #       visible=True)))
            #Add trace for fit line.  Calculated in sncalc2
            fit_fig.add_trace(go.Scatter(x=1e6*3e8/np.array(nu_fit),
                                        y=np.log10(BB.astype(float)),
                                        mode='lines',
                                        name='Fit Line'))
            #Add trace for TolTEC flux.  Calculated in sncalc2
            fit_fig.add_trace(go.Scatter(x=np.array([1.1,1.4,2.0])*1e3,
                                        y=np.log10(toltec_fit_fluxes),
                                        mode='markers',
                                        name='TolTEC Flux',
                                        marker=dict(size=10)))
            #Update x axis
            fit_fig.update_xaxes(title='wavelength [microns]')

            #Convert TolTEC image scale to arcsec centered ata (0,0)
            index = self.toltec_map_from_index[tw]
            img_y, img_x = self.center_arcsec(img, pos)
            timg = toltec_snr_maps[:,:,index]
            toltec_fig = go.Figure()
            toltec_fig.add_trace(go.Heatmap(x=img_x, y=img_y, z=timg, zmax=timg.max(),
                                            zmin=timg.min(), colorbar = {'title': 'S/N'})
                                            )

            #Add scatter point at clicked position
            toltec_fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                            marker_symbol='square',
                                            marker=dict(
                                                color='LightGreen',
                                                line=dict(
                                                    color='Black',
                                                    width=2
                                                ))))

            toltec_fig.update_xaxes(title='x (arcsec)')
            toltec_fig.update_yaxes(title='y (arcsec)')
            toltec_fig.update_layout(showlegend=False)

            if options!=None:
                #Add map size
                if 'show map' in options:
                    sx, sy = self.setup_box(img, area_deg2)
                    toltec_fig.add_shape(
                        type="rect",
                        x0=np.min(sx),
                        y0=np.min(sy),
                        x1=np.max(sx),
                        y1=np.max(sy),
                        line=dict(
                            color="Red",
                        )
                    )
        else:
            fit_fig = {}
            toltec_fig = {}
        # TODO: get all00 design csv from Michael
            # if options!=None:
            #     #Show array FOV
            #     if 'show array' in options:
            #         array_scale_x = 2*60#/self.obs[self.indx].data[w]['cdelt1']
            #         array_scale_y = 2*60#/self.obs[self.indx].data[w]['cdelt2']
            #         a1100x = self.a1100x*array_scale_x
            #         a1100y = self.a1100y*array_scale_y

            #         toltec_fig.add_trace(go.Scatter(x=a1100x,
            #                                  y=a1100y,
            #                                  mode='markers',
            #                                  name='TolTEC Array',
            #                       marker=dict(symbol='x',
            #                                   size=1,
            #                                   color='Red',
            #                       #line=dict(color='LightGreen',width=2)
            #                       )))
        #Table has dark background by default
        dark = True

        fit_table = pd.DataFrame(
        {
            "Parameter": ["Fit Status", "Temperature (K)", "Scaling Constant",
                        "1.1 Depth (mJy/beam)", "1.4 Depth (mJy/beam)",
                        "2.0 Depth (mJy/beam)"],
            "Value": [self.fit_status, "N/A", "N/A", "N/A", "N/A", "N/A"]
            }
        )
        #If fit runs, update table.
        if self.fit_status == 'Success':
            fit_table['Value'][0] = self.fit_status
            fit_table['Value'][1] = '%.2f' % (T_fit)
            fit_table['Value'][2] = '%.2e' % (const_fit)
            fit_table['Value'][3] = "N/A"
            fit_table['Value'][4] = "N/A"
            fit_table['Value'][5] = "N/A"
            fit_table['Value'][3] = '%.4f' % (depth[0])
            fit_table['Value'][4] = '%.4f' % (depth[1])
            fit_table['Value'][5] = '%.4f' % (depth[2])
            dark = False

        #Else change colormode to Dark
        elif self.fit_status == 'Failure' or self.fit_status == 'Not Fit':
            fit_table['Value'][0] = self.fit_status
            fit_table['Value'][1] = "N/A"
            fit_table['Value'][2] = "N/A"
            fit_table['Value'][3] = "N/A"
            fit_table['Value'][4] = "N/A"
            fit_table['Value'][5] = "N/A"
            dark = True

        tbl = dbc.Table.from_dataframe(fit_table,dark=dark)

        #return fit, toltec, and table all at once
        return fit_fig, toltec_fig, tbl  #toltec_fig
    
    def make_toltec_map(self, w, toltec_fit_fluxes, fluxes, row, herschelBand, pos):
        '''
        Creates 2D array of each TolTEC map matching the clicked map's
        dimensions.  Takes the clicked map and scales to the TolTEC fluxes
        based on the TolTEC fit fluxes.
        '''
        w = self.herschel_index_from_w[w]
        img_path = copy.deepcopy(self.data[herschelBand][row].values[0])
        img = fits.open(img_path, memmap=False)
        toltec_nx, toltec_ny = img[pos].data.shape
        toltec_maps = np.zeros([toltec_nx, toltec_ny, 3])
        toltec_ref_pix = img[pos].header['cdelt1']*img[pos].header['cdelt2']

        for i in range(3):
            flux_ratio = toltec_fit_fluxes[i]/fluxes[w]
            toltec_maps[:,:,i] = img[pos].data*flux_ratio

        return toltec_maps, img[pos].header

    def make_toltec_snr_map(self, toltec_maps, area_deg2, time_hours, row, herschelBand, pos, atmFactor=1.0):
        '''
        Uses the TolTEC map and depth calculation to create S/N map:
        flux/depth = S/N
        '''
        img_path = copy.deepcopy(self.data[herschelBand][row].values[0])
        img = fits.open(img_path, memmap=False)
        toltec_nx, toltec_ny = img[pos].data.shape
        # depth in mJy, multiply by 10**3 to get Jy
        depth = np.array(self.get_depth(area_deg2, time_hours, atmFactor))*10**3

        toltec_snr_maps = np.zeros([toltec_nx, toltec_ny, 3])
        toltec_ref_pix = img[pos].header['cdelt1']*img[pos].header['cdelt2']

        for i in range(3): 
            w = self.toltec_bands[i]
            beam_area = self.toltec_beam_sizes[w]
            toltec_snr_maps[:,:,i] = toltec_maps[:,:,i]#*beam_area#/self.toltec_ref_pix
            depth_i = (depth[i]/beam_area)/self.Mjy_sr_to_Jy_arcsec2
            toltec_snr_maps[:,:,i] = toltec_snr_maps[:,:,i]/(depth_i)
        return toltec_snr_maps, depth

        

