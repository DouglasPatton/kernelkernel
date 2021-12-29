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
    
    
    
    def BuildBigSpeciesXPredictDF(self,species,estimator_name=None,hucdigitcount=2,cv_agg_kwargs={},return_predicted_huc8_list=True):
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
                 #takes mean across estimators
                #if not df.index.is_monotonic_increasing:
                #    df.sort_index(inplace=True)
                df_list.append(df)
                if self.cv_run:
                    assert type(df.columns) is pd.MultiIndex,f'expecting type(df.columns) is multiindex, but it is: {type:(df.columns)}'
                    
        #df=pd.concat(df_list,axis=0)
        #self.df_list=df_list
        #self.df=df
        
        if return_predicted_huc8_list:
            huc8_list=[]
            for df in df_list:
                huc8_list.extend([h[:8] for h in df.index.get_level_values('HUC12').tolist()])
            huc8_list=list(dict.fromkeys(huc8_list).keys()) #unique vals
        self.df_list=df_list
        if hucdigitcount>0:
            huc_ydf_dict=self.splitDFByHucDigits(df_list,hucdigitcount=hucdigitcount)#ydf means y in a pd.Series
        else:
            huc_ydf_dict={'all_hucs':pd.concat(df_list,axis=0)}
        
        #if estimator_name is None:
        for huc,df in huc_ydf_dict.items():
            #assert type(df.index) is pd.MultiIndex
            huc_ydf_dict[huc]=df.groupby(level=['COMID','HUC12']).mean()
        
        if True:#self.cv_run:
            huc_ydf_dict=self.cvAggregate(huc_ydf_dict,**cv_agg_kwargs)
        self.huc_ydf_dict=huc_ydf_dict
        if return_predicted_huc8_list:
            return huc_ydf_dict,huc8_list
        return huc_ydf_dict
    
    def cvAggregate(self,huc_ydf_dict,agg_type='single_probability'):# 'PI_CV_Plus'
        if agg_type=='single_probability':
            
            out_huc_ydf_dict={huc:ydf.mean(axis=1).to_frame(name='y') for huc,ydf in huc_ydf_dict.items()}
            
            
        else: assert False, f'agg_type:{agg_type} not developed'
        
        return out_huc_ydf_dict
        
    def splitDFByHucDigits(self,df_list,hucdigitcount=2):
        assert hucdigitcount%2==0,f'expecting an even number for hucdigitcount: {hucdigitcount}'
        assert hucdigitcount<=12,f'expecting a number <=12 for hucdigitcount: {hucdigitcount}'
        """
        huc_comid_dict_y_dict={} #for each huc 4, a dictionary containing comid:y
        for df in df_list:
            y_arr=df.to_numpy().flatten()
            for df_i,(comid,h12) in enumerate(df.index.to_list()):
                Huc=h12[:hucdigitcount]
                try:
                    Hucdict=huc_comid_dict_y_dict[Huc]
                except KeyError:
                    huc_comid_dict_y_dict[Huc]={}
                huc_comid_dict_y_dict[Huc][comid]=y_arr[df_i]
                
        Huc_yser_dict={}
        for Huc,comid_y_dict in huc_comid_dict_y_dict.items():
            Hucser=pd.Series(comid_y_dict,name='y')
            Hucser.index.name='COMID'
            Huc_yser_dict[Huc]=Hucser
        return Huc_yser_dict
        """
        
        #cols=df_list[0].columns
        added_hucs=[]
        huc_ylist_dict={} #for each huc 4, a dictionary containing comid:y
        huc_idx_tup_dict={}
        if type(df_list[0].columns) is pd.MultiIndex:
            col_names=df_list[0].columns.names
            col_huc_dict={}
        else:
            col_names=df_list[0].columns
            col_huc_dict=None
        '''
        if type(cols) is pd.MultiIndex:
            if 'var' in cols.names:
                var_name=cols.get_level_values('var')[0]
                assert all([var_name==val for val in cols.get_level_values('var')])
                cols=[var_name]
            else:
                assert False,f'unexpected column multiindex: {cols} in df:{df_list[0]}'
        '''        
        for list_idx,df in enumerate(df_list): #np.flatten will make it 1d either way
            #y_arr=df.to_numpy().flatten()
            
            h12_pos=df.index.names.index('HUC12')
            comid_pos=df.index.names.index('COMID')
            df_cols=df.columns.to_list()
            for df_i,(idx_tup,row) in enumerate(df.iterrows()):
                comid=idx_tup[comid_pos]
                h12=idx_tup[h12_pos]
                Huc=h12[:hucdigitcount]
                if not Huc in added_hucs:
                    added_hucs.append(Huc)
                    huc_ylist_dict[Huc]=[]
                    huc_idx_tup_dict[Huc]=[]
                    if not col_huc_dict is None:
                        col_huc_dict[Huc]=[]
                huc_ylist_dict[Huc].append(row.values)
                huc_idx_tup_dict[Huc].append(idx_tup)
                #if not col_huc_dict is None:
                #    col_huc_dict[Huc].append(df_cols[df_i])
        self.huc_ylist_dict=huc_ylist_dict
        if True:#not col_huc_dict is None:
            huc_df_dict={
                Huc:pd.DataFrame(
                    np.array(huc_ylist_dict[Huc]),
                    columns=df_list[0].columns,#pd.MultiIndex.from_tuples(col_huc_dict[Huc],names=col_names),
                    index=pd.MultiIndex.from_tuples(huc_idx_tup_dict[Huc],names=df.index.names)
                    ) for Huc in huc_ylist_dict.keys()}
        else:
            huc_df_dict={
                Huc:pd.DataFrame(
                    np.array(huc_ylist_dict[Huc]),
                    columns=col_names,
                    index=pd.MultiIndex.from_tuples(huc_idx_tup_dict[Huc],names=df.index.names)
                    ) for Huc in huc_ylist_dict.keys()} 
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
        
    

if __name__=="__main__":
    PiscesPredictDataTool(cv_run=True).buildXPredict()
    #PiscesPredictDataTool(cv_run=False).buildXPredict()
        
        
        
        
        
                
        
        
