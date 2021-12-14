import os
import csv
#import traceback
import numpy as np
import pickle
import joblib
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
#from sklearn.model_selection import HalvingRandomSearchCV
#from sklearn.linear_model import SGDOneClassSVM
#from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline


class NHDPlusDownloader(myLogger):
    def __init__(self):
        super().__init__()
        
    


class PiscesPredictDataTool(PiscesDataTool,myLogger):
    def __init__(self,cv_run=None):
        myLogger.__init__(self,name='pisces_data_predict.log')
        self.logger.info('starting pisces_data_predict logger')
        PiscesDataTool.__init__(self,cv_run=cv_run)
        self.cv_run=cv_run
        assert not cv_run is None,'cv_run must be set to True or False!'
        #self.gt=gt()
        
    def build_species_hash_id_dict(self,output_estimator_tuple=False):
        #copied from pi_results.py and simplified
        species_hash_id_dict={}
        with DBTool(self.cv_run).genDBdict() as db:
            for hash_id,run_record in db.items():
                species=run_record['data_gen']['species']
                if output_estimator_tuple:
                    est_name=run_record['model_gen']['name']
                    hash_id_=(est_name,hash_id)
                else:
                    hash_id_=hash_id
                try: species_hash_id_dict[species].append(hash_id_)
                except KeyError: species_hash_id_dict[species]=[hash_id_]
                except: assert False, 'halt'
        return species_hash_id_dict
    
    def buildComidXHashCHashDict():
        # ex. 12345:[(xhash,chash),(xhash,chash),...]
        try: self.hash_ids_in_Xresults
        except: 
            self.hash_ids_in_Xresults=dict.fromkeys(self.XpredictHashIDComidHashResultsDB(hash_id=None))
        for hash_id in self.hash_ids_in_Xresults:
            for c_hash in self.XpredictHashIDComidHashResultsDB(hash_id=None):
                pass
    
    def buildNoveltyFilter(self,species,method=None,try_load=True):
        assert False, 'not developed'
        n_hash=joblib.hash([f'{species},{method}'])
        path=os.path.join(os.getcwd(),'data_tool',f'novelty_filter_{species}_{n_hash}.pkl')
        if try_load and os.path.exists(path):
            try:
                with open(path,'rb') as f:
                    return pickle.load(f)
            except:
                self.logger.excpetion(f'load error, rebuilding novelty filter')
            
        
    
        
    
    def buildXPredict(self,):
        species_hash_id_dict=self.build_species_hash_id_dict()
        self.hash_ids_in_Xresults=dict.fromkeys(
            self.XpredictHashIDComidHashResultsDB(hash_id=None))#dict for fast search

        for spec,spec_hash_ids in species_hash_id_dict.items():
            self.buildSpeciesXPredict(spec,spec_hash_ids,)

    def buildSpeciesXPredict(self,spec,spec_hash_ids,):
        comid_result_dict_list=[]
        hash_count=len(spec_hash_ids)
        for i,hash_id in enumerate(spec_hash_ids):
            logstr=f'for {spec},on {i+1} of {hash_count}'
            self.logger.info(logstr)
            print(logstr)
            if hash_id in self.hash_ids_in_Xresults:
                self.addToDBDict(
                    [self.XpredictHashIDComidHashResultsDB(hash_id=hash_id)],
                    db=lambda:self.XpredictSpeciesResults(spec,hash_id),fast_add=True)
            else:
                self.logger.warning(f'for {spec}, hash_id:{hash_id} not in results')
        return
    
    
    
    def BuildBigSpeciesXPredictDF(self,species,estimator_name=None,hucdigitcount=2,cv_agg_kwargs={}):
        #pulls together all of the data required to map a species, broken up into a dictionary: {huc:y_series,...}
        try: self.species_hash_id_est_dict
        except:self.species_hash_id_est_dict=self.build_species_hash_id_dict(output_estimator_tuple=True)
        tuplist=self.species_hash_id_est_dict[species]
        df_list=[]
        for est_name,hash_id in tuplist:
            if not estimator_name is None:
                if not est_name==estimator_name:continue
            for c_hash,df in self.XpredictSpeciesResults(species,hash_id).items():
                #if not type(estimator_name) is str: df.index=pd.MultiIndex.from_tuples(zip(df.index.to_list(),[est_name]*len(df.index)),names=['COMID','estimator'])
                if type(df.index) is pd.MultiIndex:
                    df=df.mean(axis=0,level=['COMID','HUC12']) #takes mean across estimators
                df_list.append(df)
                if self.cv_run:
                    assert type(df.columns) is pd.MultiIndex,f'expecting type(df.columns) is multiindex, but it is: {type:(df.columns)}'
                    
        #df=pd.concat(df_list,axis=0)
        #self.df_list=df_list
        #self.df=df
        if hucdigitcount>0:
            huc_ydf_dict=self.splitDFByHucDigits(df_list,hucdigitcount=hucdigitcount)#ydf means y in a pd.Series
        else:
            huc_ydf_dict={'all_hucs':pd.concat(df_list,axis=0)}
        
        if self.cv_run:
            huc_ydf_dict=self.cvAggregate(huc_ydf_dict,**cv_agg_kwargs)
        
        self.huc_ydf_dict=huc_ydf_dict
        return huc_ydf_dict
    
    def cvAggregate(self,huc_ydf_dict,agg_type='single_probability'):# 'PI_CV_Plus'
        if agg_type=='single_probability':
            out_huc_ydf_dict={huc:ydf.mean(axis=1,level='y').to_frame() for huc,ydf in huc_ydf_dict.items()}
            
            
        else: assert False, f'agg_type:{agg_type} not developed'
        
        return out_huc_ydf_dict
        
    def splitDFByHucDigits(self,df_list,hucdigitcount=2):
        assert hucdigitcount%2==0,f'expecting an even number for hucdigitcount: {hucdigitcount}'
        assert hucdigitcount<=12,f'expecting a number <=12 for hucdigitcount: {hucdigitcount}'
        cols=df_list[0].columns
        added_hucs=[]
        huc_ylist_dict={} #for each huc 4, a dictionary containing comid:y
        for list_idx,df in enumerate(df_list): #np.flatten will make it 1d either way
            #y_arr=df.to_numpy().flatten()
            for df_i,(comid,h12) in enumerate(df.index.to_list()):
                Huc=h12[:hucdigitcount]
                if not Huc in added_hucs:
                    added_hucs.append(Huc)
                    huc_ylist_dict[Huc]=[]
                huc_ylist_dict[Huc].append(df.iloc[df_i])
                
        huc_df_dict={Huc:pd.concat(df_list) for Huc,df_list in huc_ylist_dict.items()}
        
        return huc_df_dict
            
        
        
            
        
            
            
    def setSpeciesFoundComidDict(self):
        '''savepath=os.path.join(self.savedir,'species_found_comid_dict.pkl')
        if os.path.exists(savepath):
            try:
                with open(savepath,'rb') as f:
                    self.species_found_comid_dict=pickle.load(f)
                return
            except:
                self.logger.exception(f'buildSpeciesFoundComidDict found {savepath} but there was an error loading variables')'''
        self.logger.info('building Species_found_comid_dict')
        try: self.specieslist,self.speciescomidlist
        except: self.buildspecieslist()
        species_found_comid_dict={}    
        for s_idx,spec in enumerate(self.specieslist):
            species_found_comid_dict[spec]=self.speciescomidlist[s_idx]
        self.species_found_comid_dict=species_found_comid_dict        
     
    def setSpeciesFullComidDict(self,):
        self.fullcomidlist,self.species_full_comid_dict=self.generateXPredictSpeciesComidBlockDicts(
            return_comidlist=True)
     
    
    def generateXPredictBlockDF(self,spec,comidlist=None,keylist=None):
        try:
            
            self.logger.info(f'sending to buildcomidsiteinfo')
            sitedatacomid_dict=self.buildCOMIDsiteinfo(comidlist=comidlist,predict=True,rebuild=False) 
            with sitedatacomid_dict() as db:
                species_df=self.buildSpeciesDF( #function can be found in pi_data_helper.py
                    comidlist,db,keylist=keylist,species_name=spec)
            self.logger.info(f'df built for spec:{spec}, species_df.shape:{species_df.shape}')
            return species_df
        except:self.logger.exception('unexpected error')    
    
    
    
    def generateXPredictSpeciesComidBlockDicts(self,return_comidlist=False):
        try: self.gt
        except:self.gt=gt()
        self.buildspecieshuc8list()
        specieshuc8list=self.specieshuc8list
        
        self.buildspecieslist()
        specieshuclist=self.specieshuclist #from the survey
        specieslist=self.specieslist
        species_idx_list=list(range(len(specieslist)))
        #########
        species_huc8_dict={}
        for short_idx,long_idx in enumerate(species_idx_list): #merge the two sources of huc8s and remove duplicates #short_idx and long_idx are identical
            list1=specieshuclist[long_idx]
            list2=specieshuc8list[long_idx]
            species_huc8_dict[specieslist[short_idx]]=list(dict.fromkeys([*list1,*list2]))
        
        huc12comiddict={**self.gt.gethuc12comiddict()}
        huc8_huc12dict={**self.gt.build_huchuc()['huc8_huc12dict']} #pull into memory
        species_comid_dict={spec:[] for spec in species_huc8_dict.keys()}
        #species_huc12_dict={spec:[] for spec in species_huc8_dict.keys()}
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
                        #species_huc12_dict[species].extend([huc12 for _ in range(len(comids))])
                        comidlist.extend(comids)
                else:
                    missing_huc8_list.append(huc8)
        missing_huc8_list=list(dict.fromkeys(missing_huc8_list))#remove duplicates
        self.logger.warning(f'NHDplus datset is missing the following {len(missing_huc8_list)} huc8s: {missing_huc8_list}')
        if return_comidlist:
            return comidlist,species_comid_dict
        
        proc_count=10
        spec_per_chunk=int(-(-spec_count//proc_count))
        spec_chunks=(specieslist[spec_per_chunk*i:spec_per_chunk*(i+1)] for i in range(proc_count))
        args_list=[[{spec:species_comid_dict[spec] for spec in specs}] for specs in spec_chunks]
        outlist=self.runAsMultiProc(
            ComidBlockBuilder,args_list,kwargs={},
            no_mp=False,add_to_db=self.getXpredictSpeciesComidBlockDict())
        self.logger.info(f'back from MP. outlist:{outlist}')
        
    def getXpredictSpeciesComidBlockDict(self,):
        name='XpredictSpeciesComidBlockDicts'
        return self.anyNameDB(name,'data',folder='data_tool')    
        
class ComidBlockBuilder(mp.Process,myLogger):
    def __init__(self,q,i,species_comid_dict):
        myLogger.__init__(self,name='ComidBlockBuilder.log')
        self.logger.info('starting ComidBlockBuilder logger')
        super().__init__()
        self.species_comid_dict=species_comid_dict
        self.q=q
        self.i=i
        
        
    def run(self):

        spec_count=len(self.species_comid_dict)
        for s,(spec,comids) in enumerate(self.species_comid_dict.items()):
            self.logger.info(f'building comidblocks for {spec} #{s+1} of {spec_count}')
            b_size=int(1e5)
            c_count=len(comids)
            b_count=int(-(-c_count//b_size))
            self.logger.info(f'building {spec} comidblocks')
            comidblocks=(comids[b_size*b:b_size*(b+1)] for b in range(b_count))
            self.logger.info(f'buiding spec_c_dict for {spec}')
            spec_c_dict={joblib.hash(''.join(b_comids)):b_comids for b_comids in comidblocks}
            self.logger.info(f'adding to queue for {spec} ')
            self.q.put(('partial',{spec:spec_c_dict}))
        self.q.put((self.i,'complete'))
        
        
        
"""

    
    
    
    
    
    
    def generateXPredictData(self,spec_list=None,rebuild=False):
        assert False,'no longer using this approach 11/23/2020'
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
            
        xdb_keys=dict.fromkeys(self.getXdb())#dict for fast search
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
                    species_df=self.buildSpeciesDF(s_comids,sdict,species_name=spec)
                    self.logger.info(f'{spec} has df with shape:{species_df.shape}')
                    self.getXdb(key=spec,df=species_df) #save it
                    self.logger.info(f'spec:{spec} added to Xdb')
                else:
                    self.logger.info(f'spec:{spec} already in Xdb, so skipping')
        return
                        
                        
            
            
    def buildXdbBlock(self,comid_hash_key):
        pass
    
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

if __name__=="__main__":
    PiscesPredictDataTool(cv_run=True).buildXPredict()
    #PiscesPredictDataTool(cv_run=False).buildXPredict()
        
        
        
        
        
                
        
        
