import os,re
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
from mylogger import myLogger
from pi_db_tool import DBTool


class Helper():
    def __init__(self,):
        pass
    
    def mergelistofdicts(self,listofdicts,overwrite=0):
        try:
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
                        newkey=key+f'_{i}'
                        mergedict[newkey]=val
            return mergedict
        except: self.logger.exception(f'')
    
    
    def runAsMultiProc(self,the_proc,args_list,kwargs={},no_mp=False,add_to_db=None):
        try:
            starttime=time()
            if no_mp:q=None
            else:q=mp.Queue()
            I=len(args_list)
            q_args_list=[[q,i,*args_list[i]] for i in range(I)]
            proc_count=I
            procs=[the_proc(*q_args_list[i],**kwargs) for i in range(proc_count)]
            if no_mp:
                self.procs=procs
                if type(no_mp) is int:procs=procs[-no_mp:]
                self.procs=procs
                for proc in procs:
                    proc.run()
                self.logger.info(f'procs have run with no_mp:{no_mp}')
                results=[proc.result for proc in procs]
                return results
            else:
                [proc.start() for proc in procs]
            outlist=[None for _ in range(I)]
            countdown=proc_count
        except:
            self.logger.exception('error in runasmultiproc')
            assert False,'unexpected error'
        if no_mp:
            self.logger.warning(f'something went wrong with no_mp')
            return
        while countdown and not no_mp:
            try:
                self.logger.info(f'multiproc checking q. countdown:{countdown}')
                i,result=q.get(True,20)
                if not add_to_db is None and type(i) is str:
                    if i=='partial':
                        self.logger.info(f'multiproc adding to db')
                        with add_to_db() as db:
                            for key,val in result.items():
                                db[key]=val
                            db.commit()
                    else:
                        assert False, f'i:{i} but no other string understood'
                else:
                    self.logger.info(f'multiproc has something from the q!')
                    outlist[i]=result
                    countdown-=1
                    self.logger.info(f'proc completed. countdown:{countdown}')
            except:
                #self.logger.exception('error')
                if not q.empty(): self.logger.exception(f'error while checking q, but not empty')
                else: sleep(5)
        [proc.join() for proc in procs]
        q.close()
        self.logger.info(f'all procs joined sucessfully')
        endtime=time()
        self.logger.info(f'pool complete at {endtime}, time elapsed: {(endtime-starttime)/60} minutes')
        return outlist
    
    
    
    
    
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
    
    
    def buildSpeciesDF(
        self,comidlist,sitedatacomid_dict,
        presence_dict={}, keylist=None, species_name='none'):
        """
        """
        if keylist is None:
            keylist=[key for klist in [list(sitedatacomid_dict[comid].keys()) for comid in comidlist] for key in klist] 
            keylist=list(dict.fromkeys(keylist))
            badvars=['WsPctFullRp100', 'WsAreaSqKmRp100',
                     'CatAreaSqKmRp100','CatPctFullRp100',
                     'CatAreaSqKm','WsAreaSqKm','CatPctFull','WsPctFull','NARS_Region','NRSA_Frame']
            keylist=[key for key in keylist if not key in badvars]
            keylist=[key for key in keylist if not re.search(r'0[8-9](cat|ws)$',key)] #drop 2008,09 prism measures
            keylist=[key for key in keylist if not re.search(r'^Dam',key)] #drop redun
            keylist=self.drop_multi_version_vars(keylist)
        vardatadict=presence_dict#may be empty dict
        k_count=len(keylist)
        c_count=len(comidlist)
        for key in keylist:
                vardatadict[key]=[np.nan]*c_count
        self.logger.info(f'vardatadict initialized with {k_count} keys and {c_count} comids')
        drop_comids=[]
        for j,comidj in enumerate(comidlist):
            #sitevars=[val for _,val in self.sitedatacomid_dict[comidj].items()]
            try:
                comid_data=sitedatacomid_dict[comidj]
            except KeyError:
                comid_data={}
                drop_comids.append(comidj)
                self.logger.info(f'{comidj} not in sitedatacomid_dict')
            except:assert False,'unexpected error'
            for key in keylist:
                val=None
                try:
                    val=comid_data[key]
                except KeyError:
                    pass
                except:
                    self.logger.exception('')
                    assert False,'unexpected error'
                if not val is None:
                    vardatadict[key][j]=val
        v=0
        for var,obs_list in vardatadict.items():
            if len(obs_list)!=c_count:
                v+=1
                self.logger.critical(f'for {species_name}, var:{var} len is {len(obs_list)} but expecting {c_count}')
        if v:
            self.logger.info(f'{v} problems came up for {species_name}') 
        else:
            self.logger.info(f'data built for {species_name} with no length errors')
        species_df=pd.DataFrame(data=vardatadict,index=comidlist)
        species_df.index.name='COMID'
        species_df.drop(list(set(drop_comids)),axis=0,inplace=True)
        self.logger.info(f'dropped drop_comids:{drop_comids}')
        return species_df
        
class MpBuildSpeciesData01(mp.Process,myLogger,DBTool,Helper):       
    def __init__(self,q,i,speciesidx_list,savedir,specieslist,sitedatacomid_dict,specieshuclist_survey_idx,specieshuclist_survey_idx_newhucs,huccomidlist_survey,speciescomidlist):
        self.mypid=os.getpid()
        myLogger.__init__(self,name=f'build_{self.mypid}.log')
        DBTool.__init__(self)
        super().__init__(name=f'buildspeciesdata01_{self.mypid}.log')
        Helper.__init__(self)
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
        fail_record=[];recordfailcount=0
        self.pi_db=self.pidataDBdict()
        sitedatacomid_dict=self.sitedatacomid_dict   
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
                    with self.sitedatacomid_dict() as db:
                        comidlist_i,comid_idx=zip(*[(comid,idx) for idx,comid in enumerate(specieshuc_allcomid) if comid in db])
                    
                    
                    c_count=len(comid_idx)
                    assert len(comidlist_i)==len(comid_idx),f'expecting equal lengths but len(comidlist_i):{len(comidlist_i)}!=len(comid_idx):{len(comid_idx)}'
                    species01=[species01list[comid] for comid in comid_idx]
                    presence_dict={'presence':species01}
                    with self.sitedatacomid_dict() as db:
                        species_df=self.buildSpeciesDF(
                            comidlist_i,db,
                            presence_dict=presence_dict,species_name=spec_i)
                    #self.logger.warning(f'created df for species:{spec_i}')
                    #self.logger.info(f'for species:{spec_i} df.head(): {species_df.head()}')
                    #speciesdata=pd.concat(dflist,axis=1)
                    self.addToDBDict([{spec_i:species_df}],pi_data='species01')
                    self.logger.info(f'added to db i:{i},idx:{idx},species:{spec_i}. species_df.shape:{species_df.shape}')
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
        
        if fail_record:
            self.logger.warning(f'succesful completion. len(self.speciesidx_list): {len(self.speciesidx_list)}, recordfailcount: {recordfailcount}')
            self.q.put([self.i,fail_record])
            #self.logger.warning(f'i:{self.i} added to builder fail record q')
        else:
            self.logger.warning(f'fail record empty,recordfailcount: {recordfailcount}. i:{self.i} exiting')
        

        
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
            assert False, 'needs updating for sqlitedict'
            self.specieshuc_allcomid=specieshuc_allcomid
            self.species01list=species01list    
            with open(filepath,'wb') as f:
                pickle.dump((specieshuc_allcomid,species01list),f) 
            return
        if full_list==0:
            return specieshuc_allcomid,species01list
        
        
        

class MpBuildStreamcatFromComids(mp.Process,myLogger):
    def __init__(self,q,i,comidlist,gt):
        
        super().__init__()
        myLogger.__init__(self,)
        self.logger.info(f'MpBuildStreamcatFromComids starting  logger')
        self.q=q
        self.gt=gt
        self.comidlist=comidlist
        self.i=i
    
    def run(self,):
        self.mypid=os.getpid()
        comidlist=self.comidlist # a list of comids as strings
        comidcount=len(comidlist)
        blocksize=200
        blockcount=-(-comidcount//blocksize)
        com_idx_blocks=np.array_split(np.arange(comidcount),blockcount)
        comid_blocks=[[comidlist[c_i] for c_i in c_i_block] for c_i_block in com_idx_blocks]
        for b in range(blockcount):
            self.logger.info(f'pid:{self.mypid} starting block{b+1} of {blockcount}')
            sc_comid_dict=self.gt.getstreamcat(comid_blocks[b],add_huc12=1)
            if len(sc_comid_dict)==0:
                self.logger.warning(f'sc_comid_dict has len 0 from getstreamcat for comid_blocks[{b}]:{comid_blocks[b]}')
            else:
                self.logger.info(f'adding block{b} to queue as partial')
                self.q.put(['partial',sc_comid_dict])
        
        
        self.q.put([self.i,'complete'])
        self.logger.info(f'pid:{self.mypid} completed add to q')
        
