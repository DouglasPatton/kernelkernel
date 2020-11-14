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
        specieshuc8list=self.specieshuc8list #from the separate list
        specieshuclist_newhucs=self.specieshuclist_newhucs
        
        self.buildspecieslist()
        specieshuclist=self.specieshuclist #from the survey
        specieslist=self.specieslist
        species_huc8_dict={species:[] for species in self.specieslist}
        for idx in range(len(specieslist)):
            species_huc8_dict[specieslist[idx]].append(
                [*specieshuclist[idx],*specieshuc8list[idx]])
        for species in species_huc8_dict.keys():#remove duplicates
            species_huc8_dict[species]=list(set(species_huc8_dict[species]))
            
            
        
        huc12comiddict=self.gethuc12comiddict()
        huc8_huc12dict=self.gt.build_huchuc()['huc8_huc12dict']
        species_comid_dict={spec:[] for spec in species_huc8_dict.keys()}
        for species in species_huc8_dict:
            for huc8 in species_huc8_dict[species]:
                for huc12 in huc8_huc12dict[huc8]:
                    species_comid_dict[species].extend(huc12comiddict[huc12])
                    
            
                
            
        
        
        