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

class Sample:
    def __init__(self, sample, csv, fits=[]):
        self.sample = sample
        self.data = pd.read_csv(csv)
        self.fits = fits        #paths to all files in fits directory
        self.add_column('250 band')
        self.add_column('350 band')
        self.add_column('500 band')
        self.fixed_fits = self.fix_FITS_names()
        self.add_column('Color')
        self.add_column('Fits Paths')
        self.add_column('Fixed Name Fits')
        
        for i in range(len(self.data['Object Name'])):
            self.add_data('Color',i,1)
            
        if(len(fits) > 0):
            
            '''
            for name in self.data['Object Name']:
                self.data.loc[self.data['Object Name'] == name, ['Fits Paths']] = self.get_galaxy_fits(name)[0]
                self.data.loc[self.data['Object Name'] == name, ['Fixed Name Fits']] = self.get_galaxy_fits(name)[1]
                self.loadSpire(self.get_galaxy_fits(name)[1],self.get_galaxy_fits(name)[0],name)
            '''
            
            for i, row in self.data.iterrows():
                name = row["Object Name"]
                fixed_name_fits, fits_path = self.get_galaxy_fits(name)
                self.data.at[i, "Fits Paths"] = fits_path
                self.data.at[i, "Fixed Name Fits"] = fixed_name_fits
                self.loadSpire(fits_path, fixed_name_fits, name)
                
                
        """
        if(len(fits) > 0):
            index = self.data.index
            for name in self.data['Object Name']:
                #name = self.data['Object Name'][i]
                #name = self.data.at('Object Name',i)
                if type(name) == str: 
                    if sample == 'Kingfish':
                        self.loadSpire(self.get_galaxy_fits(name),name)
      """
    #Change FITS file names to NED names and removes path which is not to a fits file
    def fix_FITS_names(self):
        fits = []
        for i in self.fits[1:]:
            name = i.split('_')[0]
            newName = Ned.query_object(name)['Object Name'][0]
            newNameFixed = newName.split(' ')[0]
            for index in range(1,len(newName.split(' '))):
                newNameFixed = newName + '_' + newName.split(' ')[index]
            fits.append(newName + '_' + i.split('_')[1] + '_' +
                        i.split('_')[2] + '_' + i.split('_')[3] + '_' +
                        i.split('_')[4])
        return fits
            
    #Adds new columns
    def add_column(self,colname):
        self.data[colname] = [None for _ in range(len(self.data['Object Name']))]
        
    #Adds data to column
    def add_data(self,column,row,data):
        #self.data[column][row] = data
        self.data.loc[row,[column]] = data
        
    def get_galaxy_fits(self,name):
        fits = []
        old_fits = []
        for i in range(len(self.fixed_fits)):
            if (str(name) in self.fixed_fits[i]):
                fits.append(self.fixed_fits[i])
                old_fits.append(self.fits[i+1])
        return fits, old_fits
    
    #Correctly reads in all associated FITs for Spire galaxy
    def loadSpire(self,fits_paths, fixed_name_fits, name):     
        hdu = 0
        path = "C://Fall_2020_Wilson_Lab//SN_Page//Samples//Kingfish_FITS//Spire//KINGFISH_SPIRE_v3.0_updated//KINGFISH_SPIRE_v3.0_updated"
        for i in fits_paths:
            if(('250' in i) and ('scan.fits' in i)):
                #self.add_data('250 band',name,i)
                band_250 = get_pkg_data_filename(path + '//' + i)
                #band_250_data = fits.open(i,memmap=False)
                #band250_img = band_250[hdu].data
                #wcs_250 = WCS(band_250_data[hdu].header)
                band250_img = fits.getdata(band_250, ext=0)
                self.add_data('250 band',self.data['Object Name'] == name,path + '//' + i)
                #self.add_data('250 band WCS',name,wcs_250)
            elif(('350' in i) and ('scan.fits' in i)):
                #self.add_data('350 band',name,i)
                band_350 = get_pkg_data_filename(path + '//' + i)
                #band_350_data = fits.open(i,memmap=False)
                #band350_img = band_350[hdu].data
                #wcs_350 = WCS(band_350_data[hdu].header)
                band350_img = fits.getdata(band_350, ext=0)
                self.add_data('350 band',self.data['Object Name'] == name,path + '//' + i)
                #self.add_data('350 band WCS',name,wcs_350)
            elif(('500' in i) and ('scan.fits' in i)):
                #self.add_data('500 band',name,i)
                band_500 = get_pkg_data_filename(path +'//' + i)
                #band_500_data = fits.open(i,memmap=False)
                #band500_img = band_500[hdu].data
                #wcs_500 = WCS(band_500_data[hdu].header)
                #band500_img = fits.getdata(band_500, ext=0)
                self.add_data('500 band',self.data['Object Name'] == name,path + '//' + i)
                #self.add_data('500 band WCS',name,wcs_500)
    
    #Correctly reads in all associated FITs for PACS galaxy
    def loadPacs(self):
        pass


    #Fill in distances for Dustopedia and add flag column
    #def dustopediaDistances(self,t):
    #    for i in 
        
        
        
    
    