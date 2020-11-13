import os
import csv
#import traceback
import numpy as np
import pickle
from time import sleep,strftime,time
import multiprocessing as mp
#import geopandas as gpd
import logging
import traceback
import pandas as pd
from geogtools import GeogTool as gt # turned off after data bult
from mylogger import myLogger
from pi_db_tool import DBTool


class PiscesPredictTool(PiscesDataTool,myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='pisces_data_huc12.log')
        self.logger.info('starting pisces_data_huc12 logger')
        self.gt=gt()
        
    def generateXPredictData(self,spec_list=None):
        self.buildspecieshuc8list()
        specieshuc8list=self.specieshuc8list
        huc12comiddict=self.gethuc12comiddict()
        huc8_huc12dict=self.gt.build_huchuc()['huc8_huc12dict']
        species_comid_dict={spec:[] for spec in specieshuc8list.keys()}
        for species in specieshuc8list:
            for huc8 in specieshuc8list[species]:
                for huc12 in huc8_huc12dict[huc8]:
                    species_comid_dict[species].extend(huc12comiddict[huc12])
                    
            
                
            
        
        
        