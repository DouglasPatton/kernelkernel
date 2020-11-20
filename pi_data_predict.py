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
        
    def getXdb(self,key=None,df=None):
        path='data_tool/Xpredict.h5'
        if not key is None:
            key=key.replace(' ','_')
        if not df is None:
            df.to_hdf(path,key,complevel=1)
            return
        elif not os.path.exists(path):
            return []
        elif key is None:
            with pd.HDFStore(path) as XDB:
                specs=list(XDB.keys())
                specs=[w[1:] for w in specs]
                specs=[w.replace('_',' ') for w in specs]
            return specs
        else:
            return pd.read_hdf(path,key)
    #self.anyNameDB('Xpredict','X',folder='data_tool')       
    
    def generateXPredictData(self,spec_list=None,rebuild=False):
        name='Xpredict'
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
            
        xdb_keys=self.getXdb()
        self.logger.info(f'at start, xdb_keys:{xdb_keys}')
        specieslist,species_idx_list=zip(*[
            (specieslist[s_idx],l_idx) for s_idx,l_idx in enumerate(species_idx_list) if not specieslist[s_idx] in xdb_keys])
        if not rebuild:
            if len(specieslist)==0:
                return 
            else:
                self.logger.info('rebuild is FAlse, but not all species are built, ')
        else:
            if len(specieslist)==0:
                self.logger.info(f'rebuild is True, but all species in Xdb. returning')
                return 
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
        comids_with_data=dict.fromkeys(sitedatacomid_dict().keys()) #dict for quick search
        self.logger.info(f'assigning comids to species')
        c_count=len(comid_species_dict)
        s_count=len(species_comid_dict)
        with sitedatacomid_dict() as sitedatacomid_db:
            for s,(spec,comids) in enumerate(species_comid_dict.items()):
                if not spec in xdb_keys:
                    sdict={}

                    s_comids=[comid for comid in comids if comid in comids_with_data]
                    c_count=len(s_comids)
                    self.logger.info(f'building species:{spec} df. {s+1} of {s_count}. comids_with_data/comids:{c_count}/{len(comids)}')
                    logint=5*np.log10(c_count)+10
                    for c,comid in enumerate(s_comids):
                        if c%(c_count//logint)==0:self.logger.info(f'adding comid {c+1}/{c_count}')
                        sdict[comid]=sitedatacomid_db[comid]
                    species_df=self.buildSpeciesDF(s_comids,sdict,pecies_name=spec)
                    self.logger.info(f'{spec} has df with shape:{species_df.shape}')
                    self.getXdb(key=spec,df=species_df) #save it
                    self.logger.info(f'spec:{spec} added to Xdb')
                else:
                    self.logger.info(f'spec:{spec} already in Xdb, so skipping')
        return
                        
                        
            
            
            
            
            
        """
            
            
            with self.anyNameDB(name+'_raw','raw',folder='data_tool') as db: 
                #    initialize the dict
                for spec in species_comid_dict.keys():
                    if not spec in db:
                        db[spec]={}
                db.commit()
            with self.anyNameDB(name+'_raw','raw',folder='data_tool') as db:     
                for c,(comid,specs) in enumerate(comid_species_dict.items()):
                    if c%100==0:self.logger.info(f'adding comid data to specs. {c+1}/{c_count}')
                    if comid in comids_with_data:
                        try:
                            comid_data=sitedatacomid_db[comid]
                            for spec in specs:
                                db[spec][comid]=comid_data
                            db.commit()
                        except KeyError:
                            self.logger.exception(f'error for comid:{comid}')
                        except:
                            self.logger.exception(f'unexpected kind of error. comid:{comid}')
        self.logger.info(f'building Xdfs')
        with self.anyNameDB(name+'_raw','raw',folder='data_tool',flag='r') as rawdb:
            with self.anyNameDB(name,'X',folder='data_tool') as Xdb:
                scount=len(rawdb)
                for s,(spec,comid_data_dict) in enumerate(rawdb.items()):
                    if s%10==0:self.logger.info(f'building df {s+1}/{scount}')
                    comidlist_sp=list(comid_data_dict.keys())
                    self.logger.info(f'Xpredict building DF for {spec} ')
                    species_df=self.buildSpeciesDF(comidlist_sp,comid_data_dict)
                    self.logger.info(f'{spec} has df with shape:{species_df.shape}')
                    Xdb[spec]=species_df
                    self.logger.info(f'xpredict writing {spec} to drive')
                    Xdb.commit()"""

        
        
        
        
        
        
                
        
        