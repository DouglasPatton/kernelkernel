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
from pisces_data_huc12 import PiscesDataTool


class PiscesPredictDataTool(PiscesDataTool,myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='pisces_data_predict.log')
        self.logger.info('starting pisces_data_predict logger')
        PiscesDataTool.__init__(self,)
        self.gt=gt()
        
    def getXdb(self,):
        return self.anyNameDB('XPredict','X',folder='data_tool')       
    
    def generateXPredictData(self,spec_list=None,rebuild=False):
        name='XPredict'
        self.buildspecieshuc8list()
        specieshuc8list=self.specieshuc8list #from the separate list
        #specieshuclist_newhucs=self.specieshuclist_newhucs
        
        self.buildspecieslist()
        specieshuclist=self.specieshuclist #from the survey
        if spec_list is None:
            specieslist=self.specieslist
            species_idx_list=list(range(len(specieslist)))
        else:
            specieslist=spec_list
            #name+=f'_{joblib.hash(specieslist)}' # only if list is constrained
            species_idx_list=[self.specieslist.index(spec) for spec in enumerate(specieslist)]
            
        with self.getXdb() as xdb:
            specieslist,species_idx_list=zip(*[
                (specieslist[s_idx],l_idx) for s_idx,l_idx in enumerate(species_idx_list) if not specieslist[s_idx] in xdb])
        if not rebuild:
            if len(specieslist)==0:
                return self.getXdb
            else:
                self.logger.info('rebuild is FAlse, but not all species are built, ')
        else:
            if len(specieslist)==0:
                self.logger.info(f'rebuild is True, but all species in Xdb, returnin Xdb')
                return self.getXdb
        self.logger.info(f'building {specieslist}')
        species_huc8_dict={}
        for short_idx,long_idx in enumerate(species_idx_list): #merge the two sources of huc8s and remove duplicates
            list1=specieshuclist[long_idx]
            list2=specieshuc8list[long_idx]
            species_huc8_dict[specieslist[short_idx]]=list(dict.fromkeys([*list1,*list2]))
        
        huc12comiddict={**self.gt.gethuc12comiddict()}
        huc8_huc12dict={**self.gt.build_huchuc()['huc8_huc12dict']} #pull into memory
        species_comid_dict={spec:[] for spec in species_huc8_dict.keys()}
        species_huc12_dict={spec:[] for spec in species_huc8_dict.keys()}
        comidlist=[]
        missing_huc8_list=[]
        spec_count=len(species_huc8_dict)
        self.logger.info(f'starting build of main comidlist')
        for s_i,species in enumerate(species_huc8_dict):
            self.logger.info(f'getting comids for {s_i+1} of {spec_count}')
            for huc8 in species_huc8_dict[species]:
                if len(huc8)<8:huc8='0'+huc8
                if huc8 in huc8_huc12dict:
                    for huc12 in huc8_huc12dict[huc8]:
                        comids=huc12comiddict[huc12]
                        species_comid_dict[species].extend(comids)
                        species_huc12_dict[species].extend([huc12 for _ in range(len(comids))])
                        comidlist.extend(comids)
                else:
                    missing_huc8_list.append(huc8)
        missing_huc8_list=list(dict.fromkeys(missing_huc8_list))#remove duplicates
        self.logger.warning(f'NHDplus datset is missing the following {len(missing_huc8_list)} huc8s: {missing_huc8_list}')
        self.logger.info(f'condensing comidlist with len:{len(comidlist)}')
        comidlist=list(dict.fromkeys(comidlist)) # remove repeats,
        #    and preserve order for efficient huc8 lookup by gt streamcat
        self.logger.info(f'after condensing len comidlist: {len(comidlist)}')
        comid_species_dict={comid:[] for comid in comidlist} #this is comids where they *could* be
        self.logger.info(f'building species_comid_dict')
        for species,comidlist_sp in species_comid_dict.items():
            for comid in comidlist_sp:
                comid_species_dict[comid].append(species)
        self.logger.info(f'sending to buildcomidsiteinfo')
        sitedatacomid_dict=self.buildCOMIDsiteinfo(
            comidlist=list(comid_species_dict.keys()),predict=True,rebuild=True) 
        
        self.logger.info(f'assigning comids to species')
        with sitedatacomid_dict() as sitedatacomid_db:
            with self.anyNameDB(name+'_raw','raw',folder='data_tool') as db: # raw b/c 
                #    initialize the dict
                for spec in species_comid_dict.keys():
                    if not spec in db:
                        db[spec]={}
                db.commit()
                for comid,specs in comid_species_dict.items():
                    comid_data=sitedatacomid_db[comid]
                    for spec in specs:
                        db[spec][comid]=comid_data
                    db.commit()
        self.logger.info(f'building Xdfs')
        with self.anyNameDB(name+'_raw','raw',folder='data_tool',flag='r') as rawdb:
            with self.anyNameDB(name,'X',folder='data_tool') as Xdb:

                for spec,comid_data_dict in rawdb.items():
                    comidlist_sp=list(comid_data_dict.keys())
                    self.logger.info(f'Xpredict building DF for {spec} ')
                    species_df=self.buildSpeciesDF(comidlist_sp,comid_data_dict)
                    self.logger.info(f'{spec} has df with shape:{species_df.shape}')
                    Xdb[spec]=species_df
                    self.logger.info(f'xpredict writing {spec} to drive')
                    Xdb.commit()

        
        
        
        
        
        
                
        
        