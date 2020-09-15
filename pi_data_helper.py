import os
import csv
#import traceback
import numpy as np
import pickle
from time import sleep,strftime,time
import multiprocessing as mp
import geopandas as gpd
import logging
import traceback
import pandas as pd
#from geogtools import GeogTool as gt
from mylogger import myLogger

        
class MpBuildSpeciesData01(mp.Process,myLogger):       
    def __init__(self,q,i,speciesidx_list,savedir,specieslist,sitedatacomid_dict,specieshuclist_survey_idx,specieshuclist_survey_idx_newhucs,huccomidlist_survey,speciescomidlist):
        self.mypid=os.getpid()
        myLogger.__init__(self,name=f'build_{self.mypid}.log')
        super().__init__()
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
        
    
    def run(self):
        datadir=os.path.join(self.savedir,'speciesdata01')
        fail_record=[];recordfailcount=0
        for i,idx in enumerate(self.speciesidx_list):
            spec_i=self.specieslist[idx]
            try:
                species_filename=os.path.join(datadir,spec_i+'.data')
                if not os.path.exists(species_filename):
                    try: 
                        self.specieshuc_allcomid
                        specieshuc_allcomid=self.specieshuc_allcomid[idx]
                        species01list=self.species01list[idx]
                    except AttributeError:
                        specieshuc_allcomid_list,species01list_list=self.buildspecieshuccomidlist(species_idx_list=[idx])
                        specieshuc_allcomid=specieshuc_allcomid_list[0]
                        species01list=species01list_list[0]
                    species_n=len(specieshuc_allcomid)
                    comidlist_i=[comid for comid in specieshuc_allcomid if comid in self.sitedatacomid_dict]
                    varcountlist=[len(self.sitedatacomid_dict[comidk].items()) for comidk in comidlist_i]
                    varcount=max(varcountlist)
                    maxvarcountcomid=specieshuc_allcomid[varcountlist.index(varcount)]
                    keylist=[key for key,_ in self.sitedatacomid_dict[maxvarcountcomid].items()]
                    #p#rint('varcount',varcount)
                    speciesdata=np.empty((species_n,varcount+1),dtype=object)#+1 for dep var
                    speciesdata[:,0]=np.array(species01list)
                    #self.missingvals=[]
                    for j,comidj in enumerate(specieshuc_allcomid):
                        if comidj in self.sitedatacomid_dict:
                            sitevars=[val for _,val in self.sitedatacomid_dict[comidj].items()]
                        try: speciesdata[j,1:]=np.array(sitevars)
                        except: 
                            self.logger.exception(f'i:{i},idx:{idx},species:{spec_i}, comid:{comidj}')
                            keylistj=[key for key,_ in self.sitedatacomid_dict[comidj].items()]
                            missingkeys=[]
                            
                            for k,key in enumerate(keylist):
                                try:#added try: to handle missing bmmi values even if key exists
                                    data_point=self.sitedatacomid_dict[comidj][key]
                                    speciesdata[j,1+k]=data_point
                                except:
                                    missingkeys.append(key)
                                    speciesdata[j,1+k]='999999'
                            self.logger.warning(f'missing keys from exception are: {missingkeys}')
                    with open(species_filename,'wb') as f:
                        pickle.dump(speciesdata,f)  
                    self.logger.info(f'i:{i},idx:{idx},species:{spec_i}. speciesdata.shape:{speciesdata.shape}')
                else:
                    self.logger.info(f'{species_filename} already exists')
                fail_record.append(0)
            except:
                self.logger.exception(f'problem came up for species number {i},{spec_i}')
                try: 
                    fail_record.append((spec_i,missingkeys))
                except: 
                    fail_record.append((spec_i,'none'))
                recordfailcount+=1
        self.logger.warning(f'succesful completion. len(self.speciesidx_list): {len(self.speciesidx_list)}, recordfailcount: {recordfailcount}')
        self.q.put([self.i,fail_record])
        self.logger.warning(f'i:{self.i} added to builder fail record q')
        


        
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
        
        
        

class MpSearchComidHuc12(mp.Process,myLogger):
    def __init__(self,q,i,comidlist,NHDplus,NHDpluscomidlist,NHDvarlist,gt,sitedata_comid_digits,sitedata):
        self.mypid=os.getpid()
        super().__init__()
        myLogger.__init__(self,name=f'search_{self.mypid}.log')
        self.logger.info(f'search_{self.mypid} starting  logger')
        self.q=q
        self.NHDplus=NHDplus
        self.NHDpluscomidlist=NHDpluscomidlist
        self.NHDvarlist=NHDvarlist
        self.gt=gt
        self.sitedata=sitedata
        self.sitedata_comid_digits=sitedata_comid_digits
        self.comidlist=comidlist
        self.i=i
    
    def run(self,):
        comidlist=self.comidlist # a list of comids as strings
        
        '''logdir=os.path.join(self.savedir,'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=f'MpSearchComidHuc12{os.getpid()}.log'
        handler=logging.FileHandler(os.path.join(logdir,handlername))
        self.logger = logging.getLogger(__name__+str(os.getpid()))
        self.logger.addHandler(handler)'''
        
        comidcount=len(comidlist)
        self.logger.info(f'retrieving streamcat')
        sc_comid_dict=self.gt.getstreamcat(comidlist)
        
        comidsitedataidx=[]
        sitedatacomid_dict={}
        huc12findfaillist=[0 for _ in range(comidcount)]
        huc12failcount=0
        comidsiteinfofindfaillist=[0 for _ in range(comidcount)]
        
        printselection=[int(idx) for idx in np.linspace(0,comidcount,11)]
        for i,comid_i in enumerate(comidlist):
            if i in printselection and i>0:
                progress=np.round(100.0*i/comidcount,1)
                failrate=np.round(100.0*huc12failcount/i,5)
                self.logger.info(f"{self.mypid}'s progress:{progress}%, failrate:{failrate}")
            hucdatadict=self.findcomidhuc12reach(comid_i)
            if not type(hucdatadict) is dict or not comid_i in sc_comid_dict : 
                huc12findfaillist[i]=1
                huc12failcount+=1
                found=0
            else:
                found=1
            if found:
                try:
                    j=self.sitedata_comid_digits.index(comid_i)
                    sitedict=self.sitedata[j]
                    comidsitedataidx.append(j)
                except:                
                    #sitedatacomid_dict[comid_i]='Not Found'
                    #comidsitedataidx.append('Not Found')
                    found=0
                    comidsiteinfofindfaillist[i]=1
            if found:
                try:
                    sc_dict=sc_comid_dict[comid_i]
                except:
                    found=0
            if found:
                if type(hucdatadict) is dict:
                    sitedict=self.mergelistofdicts([sitedict,sc_dict,hucdatadict])#,overwrite=1)
                    #elf.logger.info(f'sitedict:{sitedict}')
                sitedatacomid_dict[comid_i]=sitedict
                            
            #if i in printselection:print('i==',i,comid_i)
        self.logger.info(f'pid:{self.mypid} adding to q')
        self.q.put([self.i,(comidsitedataidx,sitedatacomid_dict,comidsiteinfofindfaillist,huc12findfaillist)])
        self.logger.info(f'pid:{self.mypid} completed add to q')
        
    def mergelistofdicts(self,listofdicts,overwrite=0):
        mergedict={}
        for i,dict_i in enumerate(listofdicts):
            for key,val in dict_i.items():
                
                if not key in mergedict:
                    mergedict[key]=val
                elif overwrite:
                    oldval=mergedict[key]
                    self.logger.info(f'merge is overwriting oldval:{oldval} for key:{key} dwith val:{val}')
                    mergedict[key]=val
                else:
                    newkey=f'{key}_{i}'
                    #p#rint(f'for dict_{i} oldkey:{key},newkey:{newkey}')
                    mergedict[newkey]=val
        return mergedict
        
    def findcomidhuc12reach(self,comid):
        #self.NHDpluscomidlist is a list of strings!
        #p#rint('self.NHDpluscomidlist[0]',self.NHDpluscomidlist[0],type(self.NHDpluscomidlist[0]))
        #comid_digits=comid#''.join(filter(str.isdigit,comid))
        datadict={}
        try:
            i=self.NHDpluscomidlist.index(comid)
            
        except:
            print(f'{comid} huc12find failed.',end=',')
            return None
        for key in self.NHDvarlist:
            datadict[key]=self.NHDplus.loc[i,key]
        return datadict