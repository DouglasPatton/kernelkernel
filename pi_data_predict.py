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
from collections import OrderedDict
from geogtools import GeogTool as gt # turned off after data bult
from mylogger import myLogger
from pi_db_tool import DBTool


class PiscesPredictTool(PiscesDataTool,myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='pisces_data_huc12.log')
        self.logger.info('starting pisces_data_huc12 logger')
        self.gt=gt()
        
    def generateXPredictData(self,spec_list=None,rebuild=True):
        name='XPredict'
        self.buildspecieshuc8list()
        specieshuc8list=self.specieshuc8list #from the separate list
        specieshuclist_newhucs=self.specieshuclist_newhucs
        
        self.buildspecieslist()
        specieshuclist=self.specieshuclist #from the survey
        if spec_list is None:
            specieslist=self.specieslist
        else:
            specieslist=spec_list
            name+=f'_{joblib.hash(specieslist)}' # only if list is constrained
        
        if not rebuild:
            Xdb=self.anyNameDB(name,'X')
            if len(Xdb.keys())==len(spec_list):
                return Xdb
            else:
                self.logger.critical(f'Xdb.keys() has len:{len(Xdb.keys())} but spec_list has len:{len(spec_list)}}. rebuilding Xdf at name:{name}')
        
        species_huc8_dict={species:[] for species in specieslist}
        for idx in range(len(specieslist)): #merge the two sources of huc8s
            species_huc8_dict[specieslist[idx]].append(
                [*specieshuclist[idx],*specieshuc8list[idx]])
        for species in species_huc8_dict.keys():#remove duplicates
            species_huc8_dict[species]=list(dict.fromkeys(species_huc8_dict[species]))
            
            
        
        huc12comiddict=self.gethuc12comiddict()
        huc8_huc12dict=self.gt.build_huchuc()['huc8_huc12dict']
        species_comid_dict={spec:[] for spec in species_huc8_dict.keys()}
        species_huc12_dict={spec:[] for spec in species_huc8_dict.keys()}
        comidlist=[]
        for species in species_huc8_dict:
            for huc8 in species_huc8_dict[species]:
                for huc12 in huc8_huc12dict[huc8]:
                    comids=huc12comiddict[huc12]
                    species_comid_dict[species].extend(comids)
                    species_huc12_dict[species].extend([huc12 for _ in range(len(comids))])
                    comidlist.extend(comids)
        comidlist=list(OrderedDict.fromkeys(comidlist)) # remove repeats,preserve order
        
        comid_species_dict={comid:[] for comid in comidlist} #this is comids where they *could* be
        for species,comidlist_sp in species_comid_dict:
            for comid in comidlist_sp
                comid_species_dict[comid].append(species)
                
        sitedatacomid_dict=self.buildCOMIDsiteinfo(
            comidlist=list(comid_species_dict.keys()),predict=True,rebuild=False) 
        
        with self.anyNameDB(name,'raw') as db: # raw b/c 
            #    initialize the dict
            for spec in species_comid_dict.keys():
                db[spec]={}
            db.commit()
            for comid,specs in comid_species_dict.items():
                comid_data=sitedatacomid_dict[comid]
                for spec in specs:
                    db[spec][comid]=comid_data
                db.commit()
        
        with self.anyNameDB(name,'X') as Xdb:
            for spec,comid_data_dict in self.anyNameDB(name,'raw').items():
                comidlist_sp=list(comid_data_dict.keys())
                self.logger.info(f'Xpredict building DF for {spec} ')
                species_df=self.buildSpeciesDF(comidlist_sp,comid_data_dict)
                self.logger.info(f'{spec} has df with shape:{species_df.shape}')
                Xdb[spec]=species_df
                self.logger.info(f'xpredict writing {spec} to drive')
                Xdb.commit()
        
        
        
        
        
        
        
                
        
        