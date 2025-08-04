'''
Created on March, 17th 2025
WORKS but gives GeodataFrame bakc
@author: hog
'''

import geopandas as gpd
import pandas as pd
import numpy as np
import re

from statsmodels.tsa.stattools import kpss, adfuller
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
from matplotlib.colors import ListedColormap


from pandas.api.types import is_numeric_dtype

from matplotlib import pyplot as plt


class TSGeoDataFrame(gpd.GeoDataFrame):

    # the index needs to be of type dt_dats
    # test = gmdf.data[gmdf.dt_dats]
    # test = test.loc[:,:gmdf.find_first_cycle()]
    # # def __init__(self, gm_dataframe,
    # #                     cycle  = 6):
    #
    # self.dt_dats                = []
    # self.nodats                 = []
    # self.dt_dats_asDays         = [0]
    # self.dt_dats_padded         = []
    # self.dt_dats_diffs          = []
    # self.dt_dats_padded_asDays  = []
    #
    # self.__cycle                = cycle 
    
    def __init__(self, *args,   dt_dats                 = None,
                                nodats                  = None,
                                dt_dats_asDays          = None,
                                dt_dats_padded          = None,
                                dt_dats_diffs           = None,
                                dt_dats_padded_asDays   = None,
                     **kwargs):
        """
        Initialize gmdata as as GeoDataFrame subclass
        
        :param all params are part of the date system that will
                be setted up here
        """
        super().__init__(*args, **kwargs)     
        
           
        object.__setattr__(self, 'dt_dats', dt_dats)
        object.__setattr__(self, 'nodats', nodats) 
        object.__setattr__(self, 'dt_dats_asDays', dt_dats_asDays)
        object.__setattr__(self, 'dt_dats_padded', dt_dats_padded)
        object.__setattr__(self, 'dt_dats_diffs', dt_dats_diffs)
        object.__setattr__(self, 'dt_dats_padded_asDays', dt_dats_padded_asDays)
        
        self.__get_day_system()
        
    def __getattr__(self, name):
        return object.__getattribute__(self, name)
    
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
     
    #TODO: Full date system only when at least one date-field
    # otherwise: none
    # override loc and iloc    
    def __get_day_system(self):
        """
        Get the acquisition scheme from the GeoDataFrame's date-columns
        and express this time line as
            - number of days from start of acquisition
            - revisiting time between acquisitions
        furthermore
            - seperate the acquisition days and the non-date values in
              two lists
        
        All values are stored as attributes of the GeoDataFrame.
        """
        
        dats            = []
        nodats          = []
        
        # SEPRERATE ACQUISITION DAYS FROM NON-DATE VALUES
        for i in self.columns:
            if isinstance(i, pd.Timestamp):
                dats.append(i)
            else:
                nodats.append(i)
        
        object.__setattr__(self, "dt_dats", dats)
        object.__setattr__(self, "nodats", nodats)  
        
        # REFORMULATE TIME LINE IN NUMBER OF DAYS FROM BEGINNING
        dats_asDays = [int((i - dats[0]).days) for i in dats[1:]]
        dats_asDays = [0] + dats_asDays
        object.__setattr__(self, "dt_dats_asDays", dats_asDays)
        
        # CALCULATE ALL REVISITING PERIODS 
        dats_diffs = [
            int((j-i).days) for i,j in 
            zip(dats[0:-1], dats[1:])
            ]
        object.__setattr__(self, "dt_dats_diffs", dats_diffs) 
        
             
        
    @property
    def _constructor(self):
        # ENSURE THAT A GMDATA OBJECT WILL BE RETURNED INSTEAD OF A
        # GEOPANDAS DATAFRAME
        return TSGeoDataFrame

   
    
    def __getitem__(self, key):
        # ENSURE THAT A GMDATA OBJECT IS RETURNED AFTER SLICING OPERATIONS
        # SLICING OTHERWISE RETURNS A PANDAS-DATAFRAME
        result = super().__getitem__(key)
        if isinstance(result, pd.DataFrame) or \
                isinstance(result, gpd.GeoDataFrame): 
            return self._constructor(result).__finalize__(self)
        return result  
    
    def __finalize__(self, other, method=None, **kwargs):
        # ENSURE THAT ALL CUSTOM ATTRIBUTES WILL BE COPIED WHEN
        # COPYING A GMDATA OBJECT
        if isinstance(other, TSGeoDataFrame):
            for attr in ["dt_dats", "nodats", "dt_dats_asDays", "dt_dats_diffs",\
                         "dt_dats_padded", "dt_dats_padded_asDays"]:  # Copy any custom attributes
                object.__setattr__(self, attr, getattr(other, attr, None))
        return self
    
    def copy(self, deep=True):
        # ENSURE THAT copy TOO RETURNS A GMDATA OBJECT
        copied = super().copy(deep=deep)
        return self._constructor(copied).__finalize__(self)
    

    ###########################################################################
    # TESTING FOR STATIONARITY ################################################   

    def adf_kpss(self, res_adf, res_kpss):
        res_adf_kpss = []
        for i,j in zip(res_adf, res_kpss):
            if i == 1:      #fail to reject ADF-h_0 -> non-stationary           
                if j == 1:  #reject KPSS-h_0 -> non-trend-stationary
                    res_adf_kpss.append(1)            #fully non-stationary ts
                else:                   #fail to reject KPSS-h_0 -> trend-stationay
                    res_adf_kpss.append(2)            #trend-statonary - trend-removement by regression
            else:                       #reject DF-h_0 -> stationary
                if j == 0: #fail to reject KPSS-h_0 -> trend-stationary
                    res_adf_kpss.append(0)            #fully stationary
                else:                   #reject KPSS-h_0 -> non trend-stationary
                    res_adf_kpss.append(3)            #difference stationary: trend remove by differencing
        
        return res_adf_kpss
            #TODO: Check again the conditions for rejection H_0 hypotethis
            #and check the combination-results again, again and again

    def __kpss_flag(self, in_ts, p_crit=0.05, **kwargs):
        warnings.simplefilter('ignore', InterpolationWarning)
        res_kpss = kpss(in_ts, **kwargs)
        
        if res_kpss[1] >= p_crit:   #fail to reject KPSS-h_0 -> trend-stationary           
            return 1                #trend-statonary - trend-removement by regression
        else:                       #reject KPSS-h_0 -> stationary
            return 0                #reject KPSS-h_0 -> non-trend-stationary
    
    def __adf_flag(self, in_ts, p_crit=0.05, **kwargs):
        warnings.simplefilter('ignore', InterpolationWarning)
        res_adf = adfuller(in_ts, **kwargs)
        
        if res_adf[1] >= p_crit:    #fail to reject ADF-h_0 -> non-stationary           
            return 1                #non-stationary
        else:                       #reject ADF-h_0 -> stationary
            return 0                #reject ADF_h_= -> stationary
                
    def ts_adf(self, **kwargs):
        return self.apply(lambda row: self.__adf_flag(row[self.dt_dats], 
                                                **kwargs), axis=1)
    def ts_kpss(self, **kwargs):
        return self.apply(lambda row: self.__kpss_flag(row[self.dt_dats], 
                                                     **kwargs), axis=1)
            
    def ts_stationarity(self):
        return self.apply(lambda row: 
                          self.adf_kpss(row[self.dt_dats]), axis=1)
    # END TESTING FOR STATIONARITY ############################################    
    ###########################################################################
    

    # FIT POLYNOMIAL MODEL ####################################################
    ###########################################################################
    def __ts_polyfit(self, in_ts, **kwargs):
        return np.polyfit(self.dt_dats_asDays, in_ts, **kwargs)
    
    def ts_polyfit(self, **kwargs):
        return self.apply(lambda row: 
            self.__ts_polyfit(row[self.dt_dats].to_list(), **kwargs), axis=1)
        
        
        
    def __ts_polyval(self, in_ts, **kwargs):
        poly_coefs = np.polyfit(self.dt_dats_asDays, in_ts, **kwargs)
        return np.polyval(poly_coefs, self.dt_dats_asDays)
    
    def ts_polyval(self, **kwargs):
        return self.apply(lambda row: 
            self.__ts_polyval(row[self.dt_dats].to_list(), **kwargs), axis=1)
    # END FIT POLYNOMIAL MODEL ################################################
    ###########################################################################
    
    def df_hist(self, column=None): #dummy dummy dummy dummy
        if column==None or not(is_numeric_dtype(self[column])): 
            return -2
        else:
            return 'float64'
        
    def df_sigma(self, column=None, indicator='mean', 
                 cmap=["#d7191c", "#fcec07", "#22f91b", "#abdda4", "#0814f0"], 
                 **kwargs):
        n_lines = 7
        colors = cmap(np.linspace(0, 1, n_lines))
        if indicator in ['mean', 'med']:
            indicator = indicator
            if indicator == 'mean':
                edges = [-2, -1, -0.5, 0.5, 1, 2]
            else:
                edges = [2.3, 5, 32, 68, 95, 99.7]
        else:
            return -1
        
        if indicator == 'med':
            edge_values = [np.percentile(self[column],i) for i in edges]
        else:
            mn = np.mean(self[column])
            sd = np.std(self[column])
            edge_values = [mn + i * sd for i in edges]
        
        
       
        return edge_values
    

    
    def ts_mean(self):
        return self.apply(lambda row: np.mean(row[self.dt_dats]), axis=1)
    
    def ts_std(self):
        return self.apply(lambda row: np.std(row[self.dt_dats]), axis=1)
    
    
    def get_ts(self):
        return self.apply(lambda row: row[self.dt_dats], axis=1)

    # TESTFUNCTIONS
    def get_mean(self, row, featurename):
        return row[featurename]*100
    
    def apply_mean(self, featurename):
        return self.apply(lambda row: row[featurename]*50, axis=1)




# STATICS #####################################################################
def read_bbd_tl5_gmfile(geofile, layer = None, engine='fiona'):
    
    datepattern = r'date_\d{8}'

    def get_pdTimestamp_dates(in_list):
        
        out_dt_dats = []
        out_dats    = []
        out_nodats  = []
        out_dt_dats_asDays = [0]       
    
        
        for i in in_list:
            if re.match(datepattern, i):
                #out_dt_dats.append(datetime.strptime(i[-8:], "%Y%m%d"))
                out_dt_dats.append(pd.Timestamp(i[-8:]))
                out_dats.append(i) 
            else:
                out_nodats.append(i)
         
        for i in out_dt_dats[1:]:
            out_dt_dats_asDays.append((i - out_dt_dats[0]).days)
    
        out_dt_dats_asDays = [int(i) for i in out_dt_dats_asDays]
        
        #out_dt_dats = [pd.Timestamp(j) for j in out_dt_dats]
        return out_dt_dats, out_dats, out_nodats, out_dt_dats_asDays
    
    
    
    cached_engine = gpd.options.io_engine
    gpd.options.io_engine = engine
    
    if layer == None:
        data = gpd.read_file(filename=geofile, engine=engine)
    else:
        data = gpd.read_file(filename=geofile, layer=layer, engine=engine)
    
    gpd.options.io_engine = cached_engine

    # should I stay or should i go
    data.index = data['PS_ID']
    data = data.drop('PS_ID', axis=1)
    dt_dats,\
    dats,\
    nodats, \
    dt_dats_asDays = get_pdTimestamp_dates(data)
    
    dt_dats_asDays = np.array(dt_dats_asDays)
    
    # rename all ts columns from strings to pd.Timestamps
    dt_dats = [pd.Timestamp(j) for j in dt_dats]
    
    for i,j in zip(dats, dt_dats):
        data.rename(columns={i : j}, errors="raise", inplace = True)

    # casting all ts values from perhaps string to float
    data[dt_dats] = data[dt_dats].astype('float')
    
    return TSGeoDataFrame(data)
