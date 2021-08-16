# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:42:18 2021

@author: Caleigh Ryan

Definition of class to create objects containing images and other relevant 
information for Herschel DGS and KINGFISH galaxies. It creates an astropy table 
for all of these elements, and includes functions to act on them for 
necessary unit conversions.
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
import dash
from lmfit import Model, Parameter, report_fit
from pathlib import Path


datadir = Path(__file__).parent.joinpath('data').as_posix()


class Sample:
    def __init__(self, sample, csv, fits=None):
        self.sample = sample
        self.data = pd.read_csv(csv)

        #unit conversions
        self.deg_per_sr = 3282.8
        self.c = 3e8
        self.arcsec_2 = 3600**2
        self.Mjy_sr_to_Jy_arcsec = 1e6/self.deg_per_sr/self.arcsec_2
        self.w_micron_to_nu = self.c*1e6
        self.w_mm_to_nu = self.c*1e3
        self.Jy_arcsec_to_erg_sr = 1e-23*self.arcsec_2*self.deg_per_sr
        
        #Initial guess for fits
        self.T_guess = 25.0
        self.Const_guess = 1000000.0#e-4
        
        #Default fit status
        self.fit_status ='Not Fit'

        if fits is None:
            fits = list()
        self.fits = fits       #paths to all files in fits directory
        self.add_column('250 band')
        self.add_column('350 band')
        self.add_column('500 band')
        self.fixed_fits = self.fix_FITS_names(sample)
        self.add_column('Color')
        self.add_column('Fits Paths')
        self.add_column('Fixed Name Fits')
        
        for i in range(len(self.data['Object Name'])):
            self.add_data('Color',i,1)
            
        if(len(fits) > 0):   
            for i, row in self.data.iterrows():
                name = row["Object Name"]
                fixed_name_fits, fits_path = self.get_galaxy_fits(name,sample)
                self.data.at[i, "Fits Paths"] = fits_path
                self.data.at[i, "Fixed Name Fits"] = fixed_name_fits
                self.loadSpire(fits_path, fixed_name_fits, name, sample)
                
    # Change FITS file names to NED names and removes path which is not to a fits file
    def fix_FITS_names(self, sample):
        fits = []
        print(f'{self.fits=}')
        if sample == 'Kingfish':
            for i in self.fits[1:]:
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
        elif sample == 'DGS':
            for i in self.fits:
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
        #self.data[column][row] = data
        self.data.loc[row,[column]] = data
        
    def get_galaxy_fits(self,name,sample):
        fits = []
        old_fits = []
        if sample == 'Kingfish':
            for i in range(len(self.fixed_fits)):
                if (str(name) in self.fixed_fits[i]):
                    fits.append(self.fixed_fits[i])
                    old_fits.append(self.fits[i+1])
        elif sample == 'DGS':
            for i in range(len(self.fixed_fits)):
                if (str(name) in self.fixed_fits[i]):
                    fits.append(self.fixed_fits[i])
                    old_fits.append(self.fits[i])
        return fits, old_fits
    
    # Correctly reads in all associated FITs for Spire galaxy
    def loadSpire(self,fits_paths, fixed_name_fits, name, sample):     
        hdu = 0
        if sample == 'Kingfish':
            path = os.path.join(datadir, 'Samples/Kingfish_FITS/Spire/KINGFISH_SPIRE_v3.0_updated_updated/KINGFISH_SPIRE_v3.0_updated_updated')
        elif sample == 'DGS':
            path = os.path.join(datadir, "Samples/DGS_FITS/Renamed_FITs")
        for i in fits_paths:
            if(('250' in i) and ('scan.fits' in i)):
                self.add_data('250 band',self.data['Object Name'] == name,os.path.join(path, i))
            elif(('350' in i) and ('scan.fits' in i)):
                self.add_data('350 band',self.data['Object Name'] == name,os.path.join(path, i))
            elif(('500' in i) and ('scan.fits' in i)):
                self.add_data('500 band',self.data['Object Name'] == name,os.path.join(path, i))
    
    #Correctly reads in all associated FITs for PACS galaxy
    # def loadPacs(self):
    #     pass
    
    def center_arcsec(self, img, pos):
        '''
        Rescale image to arcseconds
        '''
        header = img[0].header
        img = img[pos].data
        nx, ny = img.shape
        
        cdelt1 = header['cdelt1']
        cdelt2 = header['cdelt2']
        
        crpix1 = header['crpix1']
        crpix2 = header['crpix2']

        x = (np.array(range(0,nx)) - crpix1)*cdelt1
        y = (np.array(range(0,ny)) - crpix2)*cdelt2

        return x, y

    def update_herschel(self, sample, name, band):  #area_deg2, options, map_type removed for now
        '''
        Update fits figure for selected galaxy and band.
        '''
        #find current galaxy index
        row = self.data['Object Name'] == name

        if name in ['NGC6822','NGC1705']:
            pos = 1
        elif sample == 'DGS':
            pos = 1
        else:
            pos = 0

        img_path = copy.deepcopy(self.data[band][row].values[0])
        img = fits.open(img_path, memmap=False)
        color_label = "Mjy/sr"
        
        #Change to arcsec centered at (0,0)
        img_x, img_y = self.center_arcsec(img,pos)

        return color_label, img_y, img_x

    def greybody(self, f, T, const):
        '''
        Graybody function.  Uses self.beta
        '''
        f0 = self.config['meta']['Herschel']['SPIRE']['nu'][0]
        return const*self.bb(f, T)*(f/f0)**self.beta

    def fit(self, x, y, w, beta):
        '''
        Get flux and fit it with graybody
        '''
        self.beta = beta
        self.get_flux(y, x, w)

        try:
            pmod = Model(self.greybody,independent_vars=['f'],
                                         T=Parameter('T',value=self.T_guess, min=1),
                                         const=Parameter('const',value=self.Const_guess, min=1e-8))
            result = pmod.fit(data=self.fluxes_list,
                                              f=self.config['meta']['Herschel']['SPIRE']['nu'])

            print(result.values)
            self.T_fit = result.values['T']
            self.const_fit = result.values['const']
            '''
            popt, pcov = curve_fit(self.greybody, self.config['meta']['Herschel']['SPIRE']['nu'],
                               self.fluxes_list)#, p0=self.p0, bounds=([0, 1e-10],[1000, 5e-5]))

            self.T_fit = popt[0]
            self.const_fit = popt[1]
            '''
            #Make arrays with fitted parameters for line on plot
            self.make_planck_from_fit()
            #Get TolTEC fluxes with fitted parameters
            self.calc_toltec_from_fit()

            self.fit_status = 'Success'

        except:
            self.fit_status = 'Failure'

    def make_planck_from_fit(self, npts=50):
        '''
        Make arrays with fitted parameters for line on plot
        '''
        nu_min = min(np.min(self.config['meta']['Herschel']['SPIRE']['nu']),
                                 np.min(self.config['meta']['TolTEC']['nu']))
        nu_max = max(np.max(self.config['meta']['Herschel']['SPIRE']['nu']),
                                 np.max(self.config['meta']['TolTEC']['nu']))

        self.nu_fit = np.linspace(nu_min, nu_max, npts)
        self.BB = self.greybody(self.nu_fit, self.T_fit, self.const_fit)

    def calc_toltec_from_fit(self):
        '''
        Get TolTEC fluxes with fitted parameters
        '''
        self.flux11 = self.greybody(self.config['meta']['TolTEC']['nu'][0],
                                       self.T_fit, self.const_fit)
        self.flux14 = self.greybody(self.config['meta']['TolTEC']['nu'][1],
                                       self.T_fit, self.const_fit)
        self.flux20 = self.greybody(self.config['meta']['TolTEC']['nu'][2],
                                       self.T_fit, self.const_fit)

        self.toltec_fit_fluxes = [self.flux11, self.flux14, self.flux20]
        
        #Sanity check for beta and fit
        #beta = (np.log10(self.flux11/self.flux20)/(np.log10(self.config['meta']['TolTEC']['nu'][0]/self.config['meta']['TolTEC']['nu'][2]))) - 2
        #print(beta)
        print(self.toltec_fit_fluxes)
        print(self.config['meta']['TolTEC']['nu'])
        

