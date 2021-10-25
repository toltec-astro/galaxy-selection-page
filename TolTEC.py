import numpy as np

#map depth, area, or time
class TolTEC:

    #instantiate
    def __init__(self, atmFactor=1.):
        self.atmFactor = atmFactor
        self.FOV_arcmin2 = np.pi*2.**2
        self.ynorm = 0.5e-4
        self.vnorm = 500.      #km/s
        self.taunorm = 0.005
        self.MS1p1 = 12./atmFactor
        self.MS1p4 = 20./atmFactor
        self.MS2p0 = 69./atmFactor
    
    #time to map to a given depth 
    def time_hours(self, depth_mJy, area_deg2):
        """Returns time in Hours
           Inputs: depth in mJy, area in deg^2"""
        t1p1 = area_deg2/self.MS1p1/(depth_mJy**2)
        t1p4 = area_deg2/self.MS1p4/(depth_mJy**2)
        t2p0 = area_deg2/self.MS2p0/(depth_mJy**2)
        return t1p1, t1p4, t2p0

    #area one could map to a given depth in a given time
    def area_deg2(self, depth_mJy, time_hours):
        area1p1 = time_hours*self.MS1p1*(depth_mJy**2)
        area1p4 = time_hours*self.MS1p4*(depth_mJy**2)
        area2p0 = time_hours*self.MS2p0*(depth_mJy**2)  
        return area1p1, area1p4, area2p0
    
    #1-sigma depth in given time over given area
    def depth_mJy(self, area_deg2, time_hours):
        depth1p1 = np.sqrt(area_deg2/(time_hours*self.MS1p1))
        depth1p4 = np.sqrt(area_deg2/(time_hours*self.MS1p4))
        depth2p0 = np.sqrt(area_deg2/(time_hours*self.MS2p0))
        return depth1p1, depth1p4, depth2p0

    #time to map given y with 5sigma/pixel at 150GHz (2mm)
    def time_tSZ_mins(self,y,area_arcmin2,sigmaPerPix):
        return 138.*self.atmFactor/7.*\
            area_arcmin2/(2.*self.FOV_arcmin2)*\
            (self.ynorm/y)**2*\
            (sigmaPerPix/5.)**2

    #time to map given v with 2sigma/pixel at 220GHz (1.4mm)
    def time_kSZ_hours(self,v,area_arcmin2,taue,sigmaPerPix):
        return 115.*self.atmFactor/7.*\
            area_arcmin2/(2.*self.FOV_arcmin2)*\
            (self.taunorm/taue)**2 *\
            (self.vnorm/v)**2 *\
            (sigmaPerPix/2.)**2