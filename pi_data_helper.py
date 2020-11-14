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
#from geogtools import GeogTool as gt # turned off after data bult
from mylogger import myLogger
from pi_db_tool import DBTool

        
class MpBuildSpeciesData01(mp.Process,myLogger,DBTool):       
    def __init__(self,q,i,speciesidx_list,savedir,specieslist,sitedatacomid_dict,specieshuclist_survey_idx,specieshuclist_survey_idx_newhucs,huccomidlist_survey,speciescomidlist,predictXonly):
        self.mypid=os.getpid()
        myLogger.__init__(self,name=f'build_{self.mypid}.log')
        DBTool.__init__(self)
        super().__init__(name=f'buildspeciesdata01_{self.mypid}.log')
        self.logger.info(f'build_{self.mypid} starting  logger')
        self.q=q
        self.i=i
        self.speciesidx_list=speciesidx_list
        self.savedir=savedir
        self.specieslist=specieslist
        self.sitedatacomid_dict=sitedatacomid_dict
        self.specieshuclist_survey_idx=specieshuclist_survey_idx
        self.specieshuclist_survey_idx_newhucs=specieshuclist_survey_idx_newhucs
        self.huccomidlist_survey=huccomidlist_survey
        self.speciescomidlist=speciescomidlist
        self.predictXonly=predictXonly
    
    def run(self):
        fail_record=[];recordfailcount=0
        self.pi_db=self.pidataDBdict()
            
        dbkeys=list(self.pi_db.keys())
        df_dict_add_list=[]
        for i,idx in enumerate(self.speciesidx_list):
            spec_i=self.specieslist[idx]
            try:
                if not spec_i in dbkeys:
                    try: 
                        self.specieshuc_allcomid
                        specieshuc_allcomid=self.specieshuc_allcomid[idx]
                        species01list=self.species01list[idx]
                    except AttributeError:
                        specieshuc_allcomid_list,species01list_list=self.buildspecieshuccomidlist(species_idx_list=[idx])
                        specieshuc_allcomid=specieshuc_allcomid_list[0]
                        species01list=species01list_list[0]
                    
                    comidlist_i=[comid for comid in specieshuc_allcomid if comid in self.sitedatacomid_dict]
                    comid_idx=[idx for idx,comid in enumerate(specieshuc_allcomid) if comid in self.sitedatacomid_dict]
                    c_count=len(comid_idx)
                    assert len(comidlist_i)==len(comid_idx),f'expecting equal lengths but len(comidlist_i):{len(comidlist_i)}!=len(comid_idx):{len(comid_idx)}'
                    species01=[species01list[comid] for comid in comid_idx]
                    
                    keylist=[]
                    for comid in comidlist_i:
                        keylist.extend(list(self.sitedatacomid_dict[comid].keys()))
                    keylist=list(set(keylist))
                    #self.logger.info(f'full keylist:{keylist}')
                    badvars=['COMID','COMID_2','HUC8','TOHUC','BMMI','IWI',
                             'WsPctFullRp100','REACHCODE', 'WsAreaSqKmRp100',
                             'CatAreaSqKmRp100','CatPctFullRp100',]#'WA']
                    keylist=sorted([key for key in keylist if not key in badvars])
                    keylist=self.drop_multi_version_vars(keylist)
                    if not self.predictXonly:
                        
                        vardatadict={'presence':species01}
                    else:
                        vardatadict={}
                    for j,comidj in enumerate(comidlist_i):
                        #sitevars=[val for _,val in self.sitedatacomid_dict[comidj].items()]
                        comid_data=self.sitedatacomid_dict[comidj]
                        for key in keylist:
                            try:
                                val=comid_data[key]
                            except KeyError:
                                val=np.nan
                            except:
                                self.logger.exception('')
                                assert False,'unexpected error'
                                val=np.nan
                            if not key in vardatadict:
                                vardatadict[key]=[val]
                            else:
                                vardatadict[key].append(val)
                    self.logger.info(f'starting verification of length of data for {spec_i}')  
                    v=0
                    for var,obs_list in vardatadict.items():
                        if len(obs_list)!=c_count:
                            v+=1
                            self.logger.critical(f'for {spec_i}, var:{var} len is {len(obs_list)} but expecting {c_count}')
                    if v:
                        self.logger.info(f'{v} problems came up for {spec_i}') 
                    else:
                        self.logger.info(f'data built for {spec_i} with no length errors')
                    species_df=pd.DataFrame(data=vardatadict,index=comidlist_i)
                    #self.logger.warning(f'created df for species:{spec_i}')
                    #self.logger.info(f'for species:{spec_i} df.head(): {species_df.head()}')
                    #speciesdata=pd.concat(dflist,axis=1)
                    df_dict_add_list.append({spec_i:species_df})
                    self.logger.info(f'i:{i},idx:{idx},species:{spec_i}. species_df.shape:{species_df.shape}')
                else:
                    self.logger.info(f'{spec_i} already in pisces database')
                fail_record.append(0)
            
            except:
                self.logger.exception(f'problem came up for species number {i},{spec_i}')
                try: 
                    fail_record.append((spec_i,missingkeys))
                except: 
                    fail_record.append((spec_i,'none'))
                recordfailcount+=1
        self.addToDBDict(df_dict_add_list,pi_data='species01')
        if fail_record:
            self.logger.warning(f'succesful completion. len(self.speciesidx_list): {len(self.speciesidx_list)}, recordfailcount: {recordfailcount}')
            self.q.put([self.i,fail_record])
            #self.logger.warning(f'i:{self.i} added to builder fail record q')
        else:
            self.logger.warning(f'fail record empty,recordfailcount: {recordfailcount}. i:{self.i} exiting')
        
    def drop_multi_version_vars(self,keylist):
        key_v_dict={}
        vlist=['_v1','_v2','_v2_1']
        for k,key in enumerate(keylist):
            for v in vlist:
                vl=len(v)
                if key[-vl:]==v:
                    raw_key=key[:-vl]
                    if raw_key in key_v_dict:
                        key_v_dict[raw_key].append((v,k))
                    else: key_v_dict[raw_key]=[(v,k)]
        drop_idx_list=[];kept_v_list=[]
        for key,vtuplist in key_v_dict.items():
            vtuplist.sort()
            for v,k_idx in vtuplist[:-1]:
                drop_idx_list.append(k_idx)
            kept_v_list.append(vtuplist[-1][1])
                                                   
        keylist2=[key for k,key in enumerate(keylist) if not k in drop_idx_list]
        dropkeys=[key for k,key in enumerate(keylist) if k in drop_idx_list]
        keptkeys=[key for k,key in enumerate(keylist) if k in kept_v_list]
                                                   
        self.logger.info(f'keptkeys:{keptkeys}')
        self.logger.info(f'dropkeys:{dropkeys}')                                           
        return keylist2
                                                   
                                                   
            
            
            
            
            


        
    def buildspecieshuccomidlist(self,species_idx_list=None):
        if species_idx_list is None:
            
            full_list=1
            species_searchcount=len(self.specieslist)
            species_idx_list=[i for i in range(species_searchcount)]
            filepath=os.path.join(self.savedir,'specieshuccomid')
            if os.path.exists(filepath):
                try:
                    with open(filepath,'rb') as f:
                        specieshuccomidtup=pickle.load(f)
                    self.specieshuc_allcomid=specieshuccomidtup[0]
                    self.species01list=specieshuccomidtup[1]
                    print(f'opening {filepath} with length:{len(specieshuccomidtup)} and has first item length: {len(self.specieshuc_allcomid)} and type:{type(self.specieshuc_allcomid)}')
                    return
                except: print(f'error when opening {filepath}, rerunning buildspecieshuccomidlist')
        else: 
            full_list=0
            species_searchcount=len(species_idx_list)
        species01list=[[] for _ in range(species_searchcount)]
        specieshuc_allcomid=[[] for _ in range(species_searchcount)]
        
        for i,idx in enumerate(species_idx_list):
            try:
                if i%((species_searchcount+1)//2)==0:
                    print('')
                    print(f'{round(100*i/species_searchcount,1)}%',end=',')
            except:
                
                pass
            
            print(f'idx:{idx}',end='. ')
            foundincomidlist=self.speciescomidlist[idx]
            hucidxlist=self.specieshuclist_survey_idx[idx]
            try:
                hucidxlist.extend(self.specieshuclist_survey_idx_newhucs[idx])
            except:
                self.logger.exception('error appending new hucs.')
            species_huc_count=len(hucidxlist)
            #p#rint('huc_count:',species_huc_count,sep=',')
            
            for j,hucidx in enumerate(hucidxlist):
                try:
                    if j%((species_huc_count+1)//10)==0:
                        print(f'{round(100*j/species_huc_count,1)}%',end=',')
                except:pass
                allhuccomids=self.huccomidlist_survey[hucidx]
                specieshuc_allcomid[i].extend(allhuccomids)
                #p#rint('len(specieshuc_allcomid[i])',len(specieshuc_allcomid[i]),'specieshuc_allcomid[i][-1]',specieshuc_allcomid[i][-1],end=',')
                for comid in allhuccomids:
                    if comid in foundincomidlist:
                        species01list[i].append(1)
                    else:
                        species01list[i].append(0)
        #print('len(species01list)',len(species01list))
        #print(f'specieshuc_allcomid length: {len(specieshuc_allcomid)} and type:{type(specieshuc_allcomid)}')
        #print(f'species01list length: {len(species01list)} and type:{type(species01list)}')
        if full_list==1:    
            self.specieshuc_allcomid=specieshuc_allcomid
            self.species01list=species01list    
            with open(filepath,'wb') as f:
                pickle.dump((specieshuc_allcomid,species01list),f) 
            return
        if full_list==0:
            return specieshuc_allcomid,species01list
        
        
        

"""no long needed since getting X vars from streamcat
class MpSearchComidHuc12(mp.Process,myLogger):
    def __init__(self,q,i,comidlist,gt):
        self.mypid=os.getpid()
        super().__init__()
        myLogger.__init__(self,name=f'search_{self.mypid}.log')
        self.logger.info(f'search_{self.mypid} starting  logger')
        self.q=q
        self.gt=gt
        self.comidlist=comidlist
        self.i=i
    
    def run(self,):
        comidlist=self.comidlist # a list of comids as strings
        comidcount=len(comidlist)
        self.logger.info(f'retrieving streamcat')
        sc_comid_dict=self.gt.getstreamcat(comidlist)
        self.logger.info(f'type(sc_comid_dict):{type(sc_comid_dict)}')
        self.logger.info(f'pid:{self.mypid} adding to q')
        self.q.put([self.i,sc_comid_dict])
        self.logger.info(f'pid:{self.mypid} completed add to q')
        
    """