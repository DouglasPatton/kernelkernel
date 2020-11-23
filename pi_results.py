import joblib
import pickle
import os,sys,psutil
import re
import gc
from time import strftime,sleep
import datetime
import traceback
import shutil
from random import randint,seed,shuffle
import logging
from multiprocessing  import Queue
import numpy as np
import pandas as pd
from  sk_tool import SKToolInitializer
from sk_estimators import sk_estimator
from datagen import dataGenerator 
#from pisces_params import PiSetup,MonteSetup
from mylogger import myLogger
from pi_db_tool import DBTool
import matplotlib.pyplot as plt
from pi_data_viz import DataPlotter
from helpers import Helper
import pickle
from geogtools import GeogTool as GT
from pisces_data_huc12 import PiscesDataTool as PDT
from pi_runners import PredictRunner
from pi_mp_helper import MpHelper,MulXB,MatchCollapseHuc12
#from sklearn.inspection import permutation_importance
 
class PiResults(DBTool,myLogger): 
#class PiResults(DBTool,DataPlotter,myLogger):
    '''
    from qcluster.runnode: 
        model_dict={'model':SKToolInitializer(model_gen),'data_gen':data_gen,'model_gen':model_gen}
    '''
    def __init__(self,):
        func_name=f'PiResults'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        DBTool.__init__(self)
        #DataPlotter.__init__(self)
        #self.results_dict={**self.resultsDBdict()} # assuming plenty of memory to hold it
        self.printdir=os.path.join(os.getcwd(),'print')
        if not os.path.exists(self.printdir):os.mkdir(self.printdir)
        #self.scor_est_spec_dict={}
        self.ske=sk_estimator()
        self.sk_est_dict=self.ske.get_est_dict() 
        self.scorer_list=list(SKToolInitializer(None).get_scorer_dict().keys())
        self.helper=Helper()
        self.fit_scorer='f1_micro'
        self.dp=DataPlotter()
        
    
        
        
        
    def scale_coef_by_X(self,coef_df=None,wt_type='fitscor_diffscor',rebuild=0,zzzno_fish=False,spec_wt=False,cv_collapse=False,fit_scorer=None,row_norm=False):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        if type(wt_type) is str: 
            if wt_type =='none':
                wt=1
            else:
                wt=dict(fit_scorer=fit_scorer,
                    zzzno_fish=zzzno_fish,return_weights=True,row_norm=row_norm,
                    wt_type=wt_type,spec_wt=spec_wt,scale_by_X=False,cv_collapse=cv_collapse)
            
        name=os.path.join(os.getcwd(),'results','bigXB.h5')
        key='data'
        if spec_wt:
            key+=f'_{spec_wt}'
        if zzzno_fish:
            key+='_zzzno-fish'
        if type(wt) is str:
            key+=f'_{wt}'
        if cv_collapse:
            key+=f'_cv-collapse-{cv_collapse}'
        if row_norm:
            key+='_row-norm'
        key+=f'_{fit_scorer}'
        if not rebuild:
            try:
                XB_df=pd.read_hdf(name,key)
                return XB_df#sqlitedict needs a key to pickle and save an object in sqlite
            except:
                self.logger.exception(f'rebuilding {name} but rebuild:{rebuild}')
        else:
            if type(rebuild) is int:
                rebuild-=1
        if coef_df is None:
            coef_df,_=self.get_coef_stack(rebuild=rebuild,drop_zzz=not zzzno_fish,row_norm=row_norm)
        spec_list=coef_df.index.unique(level='species')
        Big_X_train_df=self.build_X_train_df(std=True)
        self.logger.info(f'Big_X_train_df.shape:{Big_X_train_df.shape}')
        if type(wt) is dict:
            self.logger.info('getting weights')
            wt_df=self.build_wt_comid_feature_importance(rebuild=rebuild,**wt) #wts with no coefs.
        elif type(wt) is pd.DataFrame:
            wt_df=wt
        else:
            wt_df=None
        if type(wt_df) is int:
            wt_df=None
        proc_count=10
        cycles=2
        chunks=proc_count*cycles
        while chunks>len(spec_list):
            self.logger.info(f'chunks bigger than spec_list, modifying')
            if cycles>1:
                cycles-=1
            else:
                proc_count-=1
                if proc_count==1:
                    chunks=1
                    
            chunks=proc_count*cycles
            self.logger.warning(f'chunks>len(spec_list, so now chunks:{chunks},proc_count:{proc_count} and len(spec_list):{len(spec_list)}')
        chunk_n=-(-len(spec_list)//chunks)
        #args_list_list=[]
        args_list=[]
        self.logger.info(f'building MulXB args_list. chunk_n:{chunk_n}, chunks:{chunks}')
        for c in range(chunks):
            
            left=c*chunk_n
            right=left+chunk_n
            spec_list_ch=spec_list[left:right]
            self.logger.info(f'spec_list_ch:{spec_list_ch}')
            if len(spec_list_ch)==0:
                self.logger.warning(f'spec_list_ch is empty, break. len(args_list):{len(args_list)}')
                break
            if wt_df is None:
                spec_wt=None
            else:
                spec_wt=wt_df.loc[spec_list_ch]
                spec_wt.index=spec_wt.index.remove_unused_levels()
                spec_wt=spec_wt
            spec_x=Big_X_train_df.loc[spec_list_ch]
            spec_x.index=spec_x.index.remove_unused_levels()
            spec_b=coef_df.loc[spec_list_ch]
            spec_b.index=spec_b.index.remove_unused_levels()  
            args=[spec_x,spec_b,spec_wt]
            for i in range(len(args)):
                if type(args[i]) is pd.DataFrame:
                    args[i]=args[i].astype(np.float32)
            args_list.append(args)
            #if right>=len(spec_list):break
        #del(Big_X_train_df)
        #del(coef_df)
        #del(wt_df)
        #gc.collect()
        mph=MpHelper()
        #self.mph=mph
        dflist=[]
        XB_df=None
        for cy in range(cycles):
            self.logger.info(f'MulXB cycle {cy+1} of {cycles}')
            mp_result=mph.runAsMultiProc(MulXB,args_list[cy*proc_count:(cy+1)*proc_count],no_mp=False,concat=0)
            if XB_df is None:
                XB_df=mp_result
            else:XB_df=pd.concat([XB_df,mp_result],axis=0)
        """self.dflist=dflist
        if type(dflist[0]) is list:
            dflist=[df for dfl in dflist for df in dfl]
        self.logger.info(f'concatenating dflist after MulXB')
        XB_df=pd.concat(dflist,axis=0) """
        
        XB_df.to_hdf(name,key,complevel=5)
        return XB_df
    
    
    def build_wt_comid_feature_importance(
        self,rebuild=0,fit_scorer=None,zzzno_fish=False,return_weights=False,row_norm=False,
        spec_wt=None,scale_by_X=False,presence_filter=False,wt_type='fitscor_diffscor',cv_collapse=False,
        spec_list=None):
        #get data
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        if return_weights:
            name=f'comid_weights_{wt_type}'
        else:
            name=f'wtd_comid_feature_importance_{wt_type}'
            if type(spec_wt) is str:#not relevant for return_weights
                name+='_'+spec_wt
        if zzzno_fish:
            name+='_zzzno-fish'
        if scale_by_X:
            name+='_scaled_by_X'
        if presence_filter:
            name+='_presence_filter'
        if cv_collapse:
            name+=f'_cv-collapse-{cv_collapse}'
        if not spec_list is None:
            sp_hash=joblib.hash(spec_list)
            self.getsave_postfit_db_dict('spec_list',{sp_hash:spec_list})
            name+=f'sp_count:{len(spec_list)}-hash:{sp_hash}'
        name+='_'+fit_scorer
        
            
        if False:#not rebuild: #just turning off rebuild here
             
            try:
                saved_data=self.getsave_postfit_db_dict(name)
                return saved_data['data']#sqlitedict needs a key to pickle and save an object in sqlite
            except:
                self.logger.info(f'rebuilding {name} but rebuild:{rebuild}')
        else:
            if type(rebuild) is int:
                rebuild-=1
        
        
        coef_df,scor_df,y,yhat=self.get_coef_stack(
            rebuild=rebuild,drop_zzz=not zzzno_fish,return_y_yhat=True,
            drop_nocoef_scors=True,row_norm=row_norm)
        if not spec_list is None:
            df_list=[coef_df,scor_df,y,yhat]
            coef_df,scor_df,y,yhat= self.select_by_index_level_vals(
                df_list,spec_list,level_name='species')      
        #y=datadict['y']#.astype('Int8')
        #yhat=datadict['yhat']#.astype('Int8')
        #coef_scor_df=datadict['coef_scor_df']#.astype('float32')
        
        #coef_df,scor_df=self.split_coef_scor_df(coef_scor_df,drop_nocoef_scors=True)
        
        if scale_by_X:
            self.logger.info(f'building coef_df scaled_by_X')
            coef_df=self.scale_coef_by_X(
                coef_df=coef_df,wt_type=wt_type,rebuild=rebuild,zzzno_fish=zzzno_fish,cv_collapse=cv_collapse,)
            self.logger.info(f'sucessfully built coef_df scaled_by_X')
        #get the scorer
        scor_select=scor_df.loc[:,('scorer:'+fit_scorer,slice(None),slice(None))]

        #drop non-linear ests
        ests_without_coefs=['gradient-boosting-classifier',
                    'hist-gradient-boosting-classifier','rbf-svc']
        for est in ests_without_coefs:
            try:yhat.drop(est,level='estimator',inplace=True)
            except:self.logger.exception(f'error dropping est:{est}, moving on')

        #make diffs
        #self.y=y;self.yhat=yhat
        yhat_a,y_a=yhat.align(y,axis=0)
        adiff_scor=np.exp(yhat_a.subtract(y_a.values,axis=0).abs()).mul(-1) # a for abs
        #self.adiff_scor=adiff_scor
        #wt coefs w/ score
        #self.scor_select=scor_select
        #self.coef_df=coef_df
        mch_kwargs={'spec_wt':spec_wt,'scale_by_X':scale_by_X,'return_weights':return_weights,
                    'presence_filter':presence_filter,'wt_type':wt_type,'cv_collapse':cv_collapse}
        
        
        #coef_a,scor_a=coef_df.align(scor_select,axis=1)
        if zzzno_fish: # no need for mp
            proc_count=1
        else: # to split list of huc's for parallel processing, 
            ##   with internal splits as well for ram savings
            proc_count=10
        
        wtd_coef_df=self.mul_wt_norm_coefs(
                adiff_scor,scor_select,coef_df,y,proc_count=proc_count,mch_kwargs=mch_kwargs)
        
        save_data={'data':wtd_coef_df} 
        self.getsave_postfit_db_dict(name,save_data)
        return wtd_coef_df
    
    
    def select_by_index_level_vals(self,df_list,level_vals,level_name='species'):
        new_list=[]
        if not type(df_list) is list:
            returndf=True
            df=[df]
        else:
            returndf=False
        for df in df_list:
            pos=df.index.names.index(level_name)
            selector=[slice(None) for _ in range(len(df.index.names))]
            selector[pos]=level_vals
            selector=tuple(selector)
            new_list.append(df.loc[selector,:])
        if returndf:
            return new_list[0]
        else:
            return new_list 
        
        
    """below replaced by general version above
    def select_specs(self,df_list,spec_list): 
        new_list=[]
        for df in df_list:
            spec_pos=df.index.names.index('species')
            selector=(slice(None) for _ in range(len(df.index.names)))
            selector[spec_pos]=spec_list
            new_list.append(df.loc[selector])
        return new_list"""
        
    
    def mul_wt_norm_coefs(self,adiff_scor,scor_select,coef_df,y,proc_count=4 ,mch_kwargs={}):
        
        
        huc12_list=adiff_scor.index.levels[1].to_list()
        
        chunk_size=-(-len(huc12_list)//proc_count) # ceiling divide
        huc12chunk=[huc12_list[chunk_size*i:chunk_size*(i+1)] for i in range(proc_count)]
        
        if mch_kwargs['return_weights']:
            coef_df=None
            
        '''adiff_scor_chunks=[adiff_scor.loc[(slice(None),huc12chunk[i],slice(None),slice(None)),:]
                           for i in range(proc_count)]'''
        adiff_scor_chunks=self.chunker(adiff_scor,'HUC12',huc12_list,proc_count)
        if mch_kwargs['presence_filter']:
            y_chunks=self.chunker(y,'HUC12',huc12_list,proc_count)
        else:
            y_chunks=[None for i in range(proc_count)]
        if mch_kwargs['scale_by_X']: #cut p coef_df b/c it has already been aligned to huc12,comid
            coef_df_chunks=self.chunker(coef_df,'HUC12',huc12_list,proc_count)
            '''h12_pos=coef_df.index.names.index('HUC12')
            selectors=[[slice(None) for _ in range(len(coef_df.index.names))] for __ in range(proc_count)]
            for i in range(proc_count):
                selectors[i][h12_pos]=huc12chunk[i]
            
            coef_df_chunks=[coef_df.loc[tuple(selectors[i])] for i in range(proc_count)]'''
            args_list=[[huc12chunk[i],coef_df_chunks[i],scor_select,adiff_scor_chunks[i],y_chunks[i]]
                       for i in range(proc_count)]
        else: #not scale_by_X, so can't cut up coef_df b/c just coef_df.index.names are spec,est
            args_list=[[huc12chunk[i],coef_df,scor_select,adiff_scor_chunks[i],y_chunks[i]]
                       for i in range(proc_count)]
        
        print(f'starting {proc_count} procs')
        dflistlist=MpHelper().runAsMultiProc(MatchCollapseHuc12,args_list,kwargs=mch_kwargs,no_mp=False)
        print('multiprocessing complete')
        #self.dflistlist=dflistlist
        #self.logger.info(f'self.dflistlist:{self.dflistlist}')
        #assert False, 'no_mp for debugging'
        dflist=[]
        for dfl in dflistlist:
            dflist.extend(dfl) 
        print('concatenating dflist')
        if mch_kwargs['presence_filter']:
            dflist0,dflist1=zip(*dflist)
            wtd_coef_df=[pd.concat(dflist0,axis=0),pd.concat(dflist1,axis=0)]
        else:
            wtd_coef_df=pd.concat(dflist,axis=0) 
        return wtd_coef_df
    
    def chunker(self,df,name,partlist,split_count):
        df_chunk_list=[]
        part_count=len(partlist)
        parts_per_split=-(-part_count//split_count)
        part_chunks=[partlist[ch*parts_per_split:(ch+1)*parts_per_split] for ch in range(split_count)]
        part_pos=df.index.names.index(name)
        selectors=[[slice(None) for _ in range(len(df.index.names))] for __ in range(split_count)]
        for i in range(split_count):
            selectors[i][part_pos]=part_chunks[i]
        df_chunk_list=[df.loc[tuple(selectors[i])] for i in range(split_count)]
        return df_chunk_list
    
    
    def build_X_train_df(self,rebuild=0,std=True):
        name=os.path.join(os.getcwd(),'results','bigX.h5')
        if std:
            key='std_data'
        else:
            key='data'
        if not rebuild:
            try:
                Big_X_train_df=pd.read_hdf(name,key)
                return Big_X_train_df#sqlitedict needs a key to pickle and save an object in sqlite
            except:
                self.logger.exception(f'rebuilding {name} but rebuild:{rebuild}')
        else:
            if type(rebuild) is int:
                rebuild-=1
        try:
            self.gen_dict
        except:
            self.gen_dict=self.genDBdict()
        species_hash_id_dict=self.build_species_hash_id_dict(rebuild=rebuild) 
        spec_list=list(species_hash_id_dict.keys())
        if std:
            metadict=self.metadataDBdict()
            
        spec_df_list=[]
        for spec in spec_list:
            spec_meta_dict=metadict[spec]
            a_hash_id=species_hash_id_dict[spec][0]
            run_record=self.gen_dict[a_hash_id]
            datagen_dict=run_record['data_gen']
            species=datagen_dict['species']
            data=dataGenerator(datagen_dict,fit_gen=False)
            n=data.y_train.shape[0]
            X_train=data.X_train
            if std:
                X_std=spec_meta_dict['X_train_std']
                X_mean=spec_meta_dict['X_train_mean']
                X_train=(X_train-X_mean)/X_std
            huc12s=data.df.loc[:,'HUC12']
            huc12strs=huc12s.apply(self.huc12float_to_str)
            comids=data.y_train.index
            names=['species','HUC12','COMID']
            index=pd.MultiIndex.from_tuples([(species,huc12strs[i],comids[i])  for i in range(n)],names=names)
            X_train.index=index
            spec_df_list.append(X_train)
        Big_X_train_df=pd.concat(spec_df_list,axis=0) 
        self.logger.info(f'writing Big_X_train_df to hdf. shape:{Big_X_train_df.shape}, name:{name},key:{key}')
        Big_X_train_df.to_hdf(name,key,complevel=1)
        return Big_X_train_df
        
                
    def huc12float_to_str(self,huc12):
        huc12str=str(int(huc12))
        if len(huc12str)==11:huc12str='0'+huc12str
        assert len(huc12str)==12,'expecting len 12 from huc12str:{huc12str}'
        return huc12str
    
    
    def get_coef_stack(
        self,rebuild=0,drop_zzz=True,
        return_y_yhat=False,drop_nocoef_scors=True,row_norm=False
        ):
        """"""
        pdict=self.stack_predictions(rebuild=rebuild)
        if drop_zzz:
            pdict=self.drop_zzz(pdict)
        coef_scor_df=pdict['coef_scor_df']
        coef_df,scor_df=self.split_coef_scor_df(coef_scor_df,drop_nocoef_scors=drop_nocoef_scors)
        
        if row_norm:
            sum_by_levs=[name for name in coef_df.columns.names if not name=='var']
            denom=coef_df.sum(axis=1,level=sum_by_levs)
            _,denom=coef_df.align(denom,axis=1,join='left')
            coef_df=coef_df.divide(denom,axis=0)
            #print('coef_df',coef_df)
        
        
        if return_y_yhat: 
            return coef_df,scor_df,pdict['y'],pdict['yhat']
        else: 
            return coef_df,scor_df
        
    def drop_zzz(self,data):
        if type(data) is dict:
            for key in data.keys():
                try:
                    data[key].drop('zzzno fish',level='species',inplace=True)
                    self.logger.info(f'zzzno_fish dropped from key:{key}')
                except:
                    self.logger.exception(f'error dropping zzzno fish from key:{key}')
        elif type(data) is pd.DataFrame:
            data.drop(species='zzzno fish',level='species',inplace=True)
        return data
    
    def split_coef_scor_df(self,coef_scor_df,drop_nocoef_scors=False):
        scor_indices=[]
        coef_indices=[]
        coef_scor_cols=coef_scor_df.columns
        for i,tup in enumerate(coef_scor_cols):
            if tup[0][:7]=='scorer:':
                scor_indices.append(i)
            else:
                coef_indices.append(i)
        scor_df=coef_scor_df.iloc[:,scor_indices]  
        coef_df=coef_scor_df.iloc[:,coef_indices]  
        coef_df=coef_df.dropna(axis=0,how='all')
        self.logger.info(f'BEFORE drop coef_df.shape:{coef_df.shape},scor_df.shape:{scor_df.shape}')
        if drop_nocoef_scors:
            coef_df,scor_df=coef_df.align(scor_df,axis=0,join='inner')#dropping scors w/o coefs
            self.logger.info(f'AFTER drop coef_df.shape:{coef_df.shape},scor_df.shape:{scor_df.shape}')
        else:
            self.logger.info(f'splitting coef_scor_df and not dropping nocoe_scors, so no after... coef_df.shape:{coef_df.shape},scor_df.shape:{scor_df.shape}.')
        #remove unused levels???
        return coef_df,scor_df
        
        
    def stack_predictions(self,rebuild=0):
        #combines different models (species+data+fit) over DF index and repetitions over columns (rep_idx level) 
        keys=['y','yhat','coef_scor_df']
        name=os.path.join(os.getcwd(),'results','prediction_stack.h5')
        
        if not rebuild:
            try:
                stacked_predict_dict={key:pd.read_hdf(name,key) for key in keys}
                return stacked_predict_dict#sqlitedict needs a key to pickle and save an object in sqlite
            except:
                self.logger.exception(f'rebuilding {name} but rebuild:{rebuild}')
        else:
            if type(rebuild) is int:
                rebuild-=1
        try: self.predict_dict
        except: self.predict_dict=self.predictDBdict()
        try: self.gen_dict
        except: self.gen_dict=self.genDBdict()
        species_hash_id_dict=self.build_species_hash_id_dict(rebuild=rebuild) 
        predictresults_tostack=[];species_y_dict={}
        for species,hash_id_list in species_hash_id_dict.items():
            est_dict={};needs_col_stacking=[]
            
            for hash_id in hash_id_list:
                
                try:
                    predict_dict=self.predict_dict[hash_id]
                    est_name=self.gen_dict[hash_id]['model_gen']['name']
                    
                    success=1
                except:
                    self.logger.exception(f'error for hash_id:{hash_id}, so skipping')
                    success=0
                if success:
                    if not species in species_y_dict:
                        species_y_dict[species]=predict_dict['y']
                    if not est_name in est_dict:
                        est_dict[est_name]=[predict_dict]
                    else:
                        if not est_name in needs_col_stacking: #so just once per est
                            needs_col_stacking.append(est_name)
                        est_dict[est_name].append(predict_dict)
            for est_name in est_dict:
                if est_name in needs_col_stacking: #if same spec and est, stack columns
                    predict_list=est_dict[est_name]

                    #predict_list=[tup[1] for tup in model_predict_tup_list]
                    stacked_predict_result=self.do_col_stack(predict_list)
                    predictresults_tostack.append(stacked_predict_result)
                else:
                    predictresults_tostack.append(est_dict[est_name][0]) #should be list with len=1
        stacked_predict_dict={'y':None,'yhat':None,'coef_scor_df':None}
        for key in ['yhat','coef_scor_df']:
            dflist=[pr[key] for pr in predictresults_tostack]
            stacked_df=pd.concat(dflist,axis=0).sort_index(axis=0) #sort_index groups levels together
            self.logger.info(f'key:{key} and stacked_df:{stacked_df}')
            stacked_predict_dict[key]=stacked_df
        y_df_list=list(species_y_dict.values())
        stacked_predict_dict['y']=pd.concat(y_df_list,axis=0).sort_index(axis=0)
        for key in keys:
            stacked_predict_dict[key].to_hdf(name,key,complevel=5)
        return stacked_predict_dict
                                                   
    def do_col_stack(self,predict_list):
        keys_with_reps=['yhat','coef_scor_df']
        hstack_dict={key:[predict_list[0][key]] for key in keys_with_reps}
        #col_midx_stack_dict={'yhat':[],'coef_scor_df':[]}
        cols=predict_list[0]['yhat'].columns # using this as the starting
        colnames=cols.names
        ## point for adjusting reps in both df's (yhat has fewer columns!)
        _,reps,_=zip(*cols)
        maxr=max(reps)
        
        for p,pr in enumerate(predict_list[1:]):
            for key in keys_with_reps:
                df=pr[key]
                oldtuplist=df.columns #each tup is ['var','rep_idx','split_idx']
                newtuplist=[(tup[0],tup[1]+maxr+1,tup[2]) for tup in oldtuplist]# add old max+1 to each rep
                df.columns=pd.MultiIndex.from_tuples(newtuplist,names=colnames) #p off by 1.
                hstack_dict[key].append(df)
            cols=pr['yhat'].columns 
            _,reps,_=zip(*cols)
            maxr=max(reps)
        self.hstack_dict=hstack_dict
        yhat_list=hstack_dict['yhat']
        new_yhat=yhat_list[0].join(yhat_list[1:])
        #new_yhat=pd.concat(hstack_dict['yhat'],axis=1).sort_index(axis=1)
        coef_scor_list=hstack_dict['coef_scor_df']
        new_coef_scor_df=coef_scor_list[0].join(coef_scor_list[1:])
        #new_coef_scor_df=pd.concat(hstack_dict['coef_scor_df'],axis=1).sort_index(axis=1)
        y=predict_list[0]['y'] # just need one of them
        new_predictresult={'y':y,'yhat':new_yhat,
                           'coef_scor_df':new_coef_scor_df}
        return new_predictresult
    
    def build_prediction_rundicts(self,rebuild=1,test=False,XpredictDB=None): # used by pisce_params PiSetup to build runners for in-sample prediction on cv test sets
        try:
            self.results_dict
        except:
            self.results_dict=self.resultsDBdict()
        
        
        if  XpredictDB is None:
            try: predictDB=self.predictDB
            except:predictDB=self.predictDBdict()
        else:
            predictDB=XpredictDB
        dbkeydict=dict.fromkeys(predictDB.keys()) #dict for fast search
        try:
            datagenhash_hash_id_dict=self.build_dghash_hash_id_dict(rebuild=0)
            rundict_list=[]
            keep_hash_id_list=[]
            self.logger.info(f'building rundict_list')
            for d,(dg_hash,hash_id_list) in enumerate(datagenhash_hash_id_dict.items()):
                if test and d>test: break
                rundict_list.append({})
                for hash_id in (hash_id_list):
                    if not hash_id in dbkeydict:
                        rundict_list[d][hash_id]=None# will be added by jobqfiller. #self.results_dict[hash_id]['model'] #
                        if not 'data_gen' in rundict_list[d]: #just add once per run_dict
                            rundict_list[d]['data_gen']=self.results_dict[hash_id]['data_gen']
                        keep_hash_id_list.append(hash_id)
            drop_idx_list=[]
            for r,rundict in enumerate(rundict_list):
                if len(rundict)==0:
                    drop_idx_list.append(r)
            for r in drop_idx_list[::-1]:
                del rundict_list[r] # delete from rhs to avoid change of order on remaining idx's to delete
            self.logger.info('building rundict_list is complete')
            return rundict_list,keep_hash_id_list
        except:
            self.logger.exception('outer catch, halting')
            assert False,'halt'
        
            
                
    def build_y_like_agg_pred_spec(self,rebuild=0,species_hash_id_dict=None):
        #just runs once per species 
        name='y_like_agg_pred_spec'
        if not rebuild:
            try:
                datadict=self.getsave_postfit_db_dict(name)
                return datadict['data']#sqlitedict needs a key to pickle and save an object in sqlite
            except:
                self.logger.info(f'rebuilding {name} but rebuild:{rebuild}')
        else:
            if type(rebuild) is int:
                rebuild-=1
        if species_hash_id_dict is None: 
            species_hash_id_dict=self.build_species_hash_id_dict(rebuild=rebuild) 
        hash_id_list1=[hash_id_list[0] for species,hash_id_list in species_hash_id_dict.items()] # just get 1 hash_id per species
        runners=[]
        
        for hash_id in hash_id_list1:
            rundict={}
            result=self.results_dict[hash_id]
            data_gen=result['data_gen'] # same as datagen_dict
            data_gen['make_y']=1
            rundict['data_gen']=data_gen
            rundict[hash_id]=None #this is where the model would go
            runners.append(PredictRunner(rundict))
        
        dflist=[runner.run() for runner in runners] # no build b/c of 'make_y'
        all_y_df=pd.concat(dflist,axis=0)
        datadict={'data':all_y_df} 
        self.getsave_postfit_db_dict(name,datadict)
        return all_y_df # just returning the df
        
            
        
        
    
    
    def getsave_postfit_db_dict(self,name,data=None,):
        if data is None:
            return self.postFitDBdict(name)
        else:
            if not type(data) is dict:
                data={'data':data}
            self.addToDBDict(data,db=lambda: self.postFitDBdict(name))
    
    def build_species_hash_id_dict(self,rebuild=0):
        try: self.gen_dict
        except:self.gen_dict=self.genDBdict
        name='species_hash_id_dict'
        if not rebuild:
            try:
                species_hash_id_dict=self.getsave_postfit_db_dict(name)
                return species_hash_id_dict
            except:
                self.logger.info(f'rebuilding species_hash_id_dict but  rebuild:{rebuild}')
        species_hash_id_dict={}
        for hash_id,run_record in self.gen_dict.items():
            species=run_record['data_gen']['species']
            try: species_hash_id_dict[species].append(hash_id)
            except KeyError: species_hash_id_dict[species]=[hash_id]
            except: assert False, 'halt'
        self.getsave_postfit_db_dict(name,species_hash_id_dict)
        return species_hash_id_dict
    
    def add_check_dghash_hash_id_dict(self,dghash_hash_id_dict):
        try: self.results_dict
        except:self.results_dict=self.resultsDBdict()
        hash_id_list=[]
        for _,h_i_list in dghash_hash_id_dict.items():
            hash_id_list.extend(h_i_list)
        results_hash_id_list=list(self.results_dict.keys())
        for hash_id in results_hash_id_list:
            if not hash_id in hash_id_list:
                result=self.results_dict[hash_id]
                data_gen=result['data_gen']
                dghash=joblib.hash(data_gen)
                try:
                    dghash_hash_id_dict[dghash].append(hash_id)
                except KeyError:
                    dghash_hash_id_dict[dghash]=[hash_id]
        return dghash_hash_id_dict
        
        
    def build_dghash_hash_id_dict(self,rebuild=0,add=0):
        name='dghash_hash_id_dict'
        db=lambda: self.anyNameDB(name,tablename='data')
        if not rebuild:
            try:
                dghash_hash_id_dict=db()
                if not add:
                    if len(dghash_hash_id_dict)==0:
                        rebuild=1
                    else:
                        return dghash_hash_id_dict
                else:
                    dghash_hash_id_dict=self.add_check_dghash_hash_id_dict(dghash_hash_id_dict)
            except:
                self.logger.info(f'rebuilding {name} but rebuild:{rebuild}')
                rebuild=1
        if rebuild:
            try: self.results_dict
            except:self.results_dict=self.resultsDBdict()
            try: self.gen_dict
            except: self.gen_dict=self.genDBdict()
            datagenhash_hash_id_dict={}
            self.logger.info(f'building datagen hash hash_id dict ')
            hash_id_list=list(self.results_dict.keys())
            """hash_id_count=len(hash_id_list)
            blocksize=100
            block_count=-(-hash_id_count//blocksize) #ceil divide
            hash_id_chunks=[hash_id_list[ch*blocksize:(ch+1)*blocksize] for ch in range(block_count)]
            for h,hash_id_chunk in enumerate(hash_id_chunks):"""
            try:
                #self.results_dict=self.resultsDBdict()
                #self.logger.info(f'starting block {h+1} of {block_count}')
                hash_id_chunk=hash_id_list
                for hash_id in hash_id_chunk:
                    run_record=self.gen_dict[hash_id]
                    data_gen=run_record["data_gen"]
                    datagenhash=joblib.hash(data_gen)
                    try:
                        datagenhash_hash_id_dict[datagenhash].append(hash_id) # in case diff species have diff 
                            #     datagen_dicts. if wrong random_state passed to cv, split is wrong
                        #datagenhash_hash_id_dict.commit()
                    except KeyError:
                        datagenhash_hash_id_dict[datagenhash]=[hash_id]
                        #datagenhash_hash_id_dict.commit()
                    except:
                        self.logger.exception(f'not a keyerror, unexpected error')
                        assert False,'halt'
                with db() as dghash_db:
                    self.logger.info(f'adding to datagenhash_hash_id_dict')
                    for key,val in datagenhash_hash_id_dict.items():
                        dghash_db[key]=val
                    dghash_db.commit()
                    self.logger.info(f'datagenhash_hash_id_dict built')
            except:
                self.logger.exception('unexpected error building dghash...')
            self.logger.info('datagen hash hash_id dict complete')
        return db()
    
    
    def build_species_estimator_prediction_dict(self,rebuild=0):
        try:
            spec_est_prediction_dict={}
            datagenhash_hash_id_dict=self.build_datagen_hash_id_dict()

            for _,model_list in datagenhash_hash_id_dict:
                assert False, 'broken'



                species=data_gen["species"]
                est_name=model_dict["model_gen"]["name"]

                try:
                    spec_est_prediction_dict[species]
                except KeyError:
                    spec_est_prediction_dict[species]={est_name:[]}
                except:
                    assert False,'unexpected'
                try:
                    spec_est_prediction_dict[species][est_name]
                except KeyError:
                    spec_est_prediction_dict[species][est_name]=[]
                if type(model_dict['model']) is dict:
                    assert False, 'broken'
                   
                else: assert False, 'only developed for CV!'
            self.spec_est_prediction_dict=spec_est_prediction_dict
            self.addToDBDict(spec_est_prediction_dict,predict=1)
        except:
            self.logger.exception(f'outer catch in building spec_est predictions')
    def spec_est_scor_df_from_dict(self,rebuild=0,scorer=None):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        try: self.scor_est_spec_dict
        except:self.build_scor_est_spec_dict(rebuild=rebuild)
        scor_est_spec_dict=self.scor_est_spec_dict
        #self.logger.info(f'scor_est_spec_dict:{scor_est_spec_dict}')
        est_spec_dict=scor_est_spec_dict[scorer]
        tup_list=[]
        data_list=[]
        for est,spec_dict in est_spec_dict.items():
            for spec,arr in spec_dict.items():
                if arr.size>0:
                    tup_list.append((spec,est))
                    data_list.append(arr)
                
        score_stack=pd.DataFrame(data_list)  
        colcount=max()
        columns=[f'cv_{i}' for i in range(score_stack.shape[1])]     #was scorer, no cv   
        score_stack.columns=columns
        m_idx=pd.MultiIndex.from_tuples(tup_list,names=['species','estimator'])
        scor_df=score_stack.set_index(m_idx)
        scor_df=pd.DataFrame(data_list,columns=columns)
        #scor_df=score_stack#pd.DataFrame(data=score_stack,index=m_idx,columns=columns)
        return scor_df
        
        
    def build_scor_est_spec_dict(self,rebuild=0):
        savename=os.path.join('results','scor_est_spec_dict.pkl')
        
        if rebuild:
            try:self.results_dict
            except:self.results_dict=self.resultsDBdict()
            scor_est_spec_dict={scorer:{est:{} for est in self.sk_est_dict.keys()} for scorer in self.scorer_list}
            for hash_id,result_dict in self.results_dict.items():
                data_gen=result_dict['data_gen']
                species=data_gen['species']
                model_gen=result_dict['model_gen']
                est_name=model_gen['name']
                model=result_dict['model']
                if 'cv' in data_gen['data_split']:
                    model_keys=list(model.keys())
                    #test_result_keys=[key for key in model_keys if key[:5]=='test_']
                    test_result_keys=[f'test_{scorer}' for scorer in self.scorer_list]
                    for key in test_result_keys:
                        val=model[key]
                        try:
                            oldval=scor_est_spec_dict[key[5:]][est_name][species]
                            val=np.concatenate([oldval,val],axis=0)
                            self.logger.info(f'scor_est_spec_dict adding to existing array of scores oldval.shape:{oldval.shape}, new val.shape:{val.shape}')
                        except:
                            pass
                        scor_est_spec_dict[key[5:]][est_name][species]=val
            self.logger.info(f'scor_est_spec_dict before dropping NAN species: {scor_est_spec_dict}')
            scor_est_spec_dict=self.drop_nan_species(scor_est_spec_dict)
            self.scor_est_spec_dict=scor_est_spec_dict
            self.save_dict(scor_est_spec_dict,filename=savename,bump=1,load=0)
        else:
            try:
                self.scor_est_spec_dict=self.save_dict(None,filename=savename,load=1)
            except:
                self.logger.exception(f'cannot locate scor_est_spec_dict, rebuilding')
                self.build_scor_est_spec_dict(rebuild=1)
                
    def drop_nan_species(self,sepd):
        drop_specs=[]
        for s,epd in sepd.items():
            for e,pd in epd.items():
                for p,scor in pd.items():
                    if any([np.isnan(scor_i) for scor_i in scor]):
                        drop_specs.append(p)
        for s,epd in list(sepd.items()):
            for e,pd in list(epd.items()):
                for p in list(pd.keys()):
                    if p in drop_specs:
                        sepd[s][e].pop(p)
        return sepd
        
        
    
    
    def save_dict(self,a_dict,filename='test.pkl',bump=1,load=0):
        if load:
            with open(filename,'rb') as f:
                dict_=pickle.load(f)
            return dict_
        if os.path.exists(filename) and bump:
            bumped_name=self.helper.getname(filename)
            with open(filename,'rb') as f:
                dict_=pickle.load(f)
            with open(bumped_name,'wb') as f:
                pickle.dump(dict_,f)
        with open(filename,'wb') as f:
            pickle.dump(a_dict,f)
            
    '''def plot_scor_df(self,alpha=0.05):
        """
        this is the incomplete, updated version that uses df framework rather than nested dicts
        """
        coef_df,scor_df,y,yhat=self.get_coef_stack(
            rebuild=rebuild,drop_zzz=True,return_y_yhat=True,
            drop_nocoef_scors=False)
        
        scor_df.columns=scor_df.columns.map(lambda x: (x[0],(*x[1:])))
        scor_df.columns.names=['var','rep_split']
        self.scor_df=scor_df
        quan=np.array(alpha/2,1-alpha/2)
        scorers=scor_df.columns.unique(level='var')
        dflist=[]
        for scorer in scorers:
            scorer_scor_df=scor_df.loc[:,scorer].apply(lambda x:x.quantile(q=quan),axis=1)
            scor_df_low_q=scorer_scor_df
        scor_df_mean=scor_df.mean(axis=1,level='var')
        
    def cv_split_merge(self,mindex):
        """
        assuming levels are var,rep,split
        """
        
        new_tups=[]
        for tup in mindex:
            new_tups.append()'''
            
        
        
    def plot_species_estimator_scores(self,ci_off=False):
        scorer_list=self.scorer_list
        scorer_count=len(scorer_list)
        est_list=list(self.sk_est_dict.keys())
        est_count=len(est_list)
        metadata={**self.metadataDBdict()} #pull into memory for re-use 
        try: self.fit_sorted_species_dict,self.scor_est_spec_MLU
        except:self.build_mean_score_sort_spec_and_MLU()
        fig=plt.figure(dpi=300,figsize=[10,scorer_count*4])
        
        for s,scorer in list(enumerate(scorer_list)):
            sorted_species_list=self.fit_sorted_species_dict[scorer]
            n_list,ymean_list=zip(*[(metadata[spec]['n'],metadata[spec]['ymean']) for spec in sorted_species_list])
            est_spec_MLU=self.scor_est_spec_MLU[scorer]
            just_numbers=np.arange(len(sorted_species_list))
            ax=fig.add_subplot(scorer_count,1,s+1)
            ax.set_title(f'results for scorer:{scorer}')
            for e,est_name in list(enumerate(est_list))[:]:
                spec_MLU=est_spec_MLU[est_name]
                mean,lower,upper=zip(*[spec_MLU[spec] for spec in sorted_species_list]) # best across estimators, 
                mean_arr=np.array(mean)
                lower_arr=np.array(lower)
                upper_arr=np.array(upper)
                ax.margins(0)
                self.dp.makePlotWithCI(
                        just_numbers,mean_arr,None,
                        ax,plottitle=est_name,color=e,
                        hatch=e,ls=e,lower=lower_arr,upper=upper_arr,ci_off=ci_off)
            ax.plot(just_numbers,ymean_list,ls='dashdot',linewidth=0.5,label="share of 1's")
            ax.plot(just_numbers,np.log10(np.array(n_list))/10,ls='--',linewidth=0.5,label="relative sample size (log scale)")
            ax.legend(loc='upper center',ncol=3,bbox_to_anchor=(0.5,-0.05))
            ax.set_xticks([])
        plt.tight_layout()
        fig.show()
        #self.fig1=fig
        figpath=self.helper.getname(os.path.join(self.printdir,f'test_scores_by_species_scorer_est.png'))
        fig.savefig(figpath)
        
    def build_mean_score_sort_spec_and_MLU(self):
        try: scor_est_spec_dict=self.scor_est_spec_dict
        except: 
            self.build_scor_est_spec_dict()
            scor_est_spec_dict=self.scor_est_spec_dict
        scorer_list=self.scorer_list
        scorer_count=len(scorer_list)
        est_list=list(self.sk_est_dict.keys())
        est_count=len(est_list)
        species_list=list(set([spec for s,est in self.scor_est_spec_dict.items() for est,spec_dict in est.items() for spec in spec_dict.keys()]))
        
        self.logger.info(f'species_list: {species_list}')
        
        scor_spec_est_mean_dict={scorer:{spec:{est:None for est in est_list} for spec in species_list} for scorer in scorer_list}
        scor_est_spec_MLU={scorer:{est:{spec:None for spec in species_list} for est in est_list} for scorer in scorer_list}
        # create a separate graph for each scorer. with each estimator on each graph
        
        
        for s,scorer in enumerate(scorer_list):
            est_spec_dict=self.scor_est_spec_dict[scorer]

            
            for e,est_name in enumerate(est_list):
                try:
                    spec_dict=est_spec_dict[est_name] # this may trigger the exception
                    score_arr_list=[] # ordered by species_list
                    for spec in species_list:
                        if spec in spec_dict:
                            score_arr_list.append(np.array(spec_dict[spec]))
                        else:
                            score_arr_list.append(np.array([np.nan]))
                    mean_scores=[np.nanmean(scores) for scores in score_arr_list] #ordered by species_list
                    
                    for i in range(len(species_list)):
                        scor_spec_est_mean_dict[scorer][species_list[i]][est_name]=mean_scores[i]
                    sorted_score_arr_list=[np.sort(arr) for arr in score_arr_list] # ordered by species_list
                    len_list=[arr.shape[0]-1 for arr in sorted_score_arr_list] 
                    l_idx,u_idx=zip(*[(int(l*0.025),int(-(-(l*0.975)//1))) for l in len_list]) 
                    #self.logger.warning(f'len_list:{len_list},l_idx:{l_idx}, u_idx:{u_idx}')
                    lower,upper=zip(*[(arr[l_idx[i]],arr[u_idx[i]]) for i,arr in enumerate(sorted_score_arr_list)])
                    for i in range(len(species_list)):
                        scor_est_spec_MLU[scorer][est_name][species_list[i]]=(mean_scores[i],lower[i],upper[i])
                    #    MLU mean lower upper
                    
                except:
                    self.logger.exception('')
        scor_spec_bestesttup_dict=self.build_best_est_dict(scor_spec_est_mean_dict) 
        #    scorer:species:(val,est) where val is lowest across ests for scorer and species
        fit_sorted_species_dict={}
        for s,scorer in list(enumerate(scorer_list)):
            est_spec_MLU=scor_est_spec_MLU[scorer]
            spec_best_est_dict_s=scor_spec_bestesttup_dict[scorer]
            best_scor_est_tups=[spec_best_est_dict_s[spec] for spec in species_list]#ordering by species_list
            
            bestscors=[tup[0] for tup in best_scor_est_tups]
            species_scorsort_idx=list(np.argsort(np.array(bestscors))) #argsort to handle np.nan in the list
            best_cvmean_scors=[best_scor_est_tups[i][1] for i in species_scorsort_idx]
            
            sorted_species_list=[species_list[i] for i in species_scorsort_idx]
            #plt.scatter(sorted_species_list,best_cvmean_scors)
            #plt.show()
            fit_sorted_species_dict[scorer]=sorted_species_list
        self.fit_sorted_species_dict=fit_sorted_species_dict
        self.scor_est_spec_MLU=scor_est_spec_MLU
            
    def build_best_est_dict(self,scor_spec_est_mean_dict,):
        scor_spec_bestesttup_dict={}
        for scorer,spec_dict in scor_spec_est_mean_dict.items():
            spec_best_est_dict={}
            for spec,est_dict in spec_dict.items():
                ests,scors=zip(*[(est,scor) for est,scor in est_dict.items()])
                maxscor=np.nanmax(np.array([scors]))
                max_idx=scors.index(maxscor)
                spec_best_est_dict[spec]=(scors[max_idx],ests[max_idx]) # (val,name) for sorting later
            scor_spec_bestesttup_dict[scorer]=spec_best_est_dict
        return scor_spec_bestesttup_dict
    
    def get_basic_meta(self,):
        m_db=self.metadataDBdict()
        spec_list=[]
        n_str='species N' # labels for x axis on histograms
        ymean_str='species Y mean'
        viz_dict={n_str:[],ymean_str:[]}
        for key,metadict in m_db.items():
            spec_list.append(key)
            viz_dict[n_str].append(metadict['n'])
            viz_dict[ymean_str].append(metadict['y_train_mean'])
        return viz_dict
    
    
    def plot_basic_hists(self,log_bins=True,bin_count=50):
        viz_dict=self.get_basic_meta()
        fig=plt.figure(dpi=200,figsize=[6,5])
        #fig.suptitle('Histograms',size='medium')
        sub_tups=[(2,1,1),(2,1,2)]
        for i,(name,data) in enumerate(viz_dict.items()):
            
            ax=fig.add_subplot(*sub_tups[i])
            self.dp.my2dHist(data,name,ax=ax,log_bins=log_bins,bin_count=bin_count)
        plt.tight_layout()
        fig.savefig(
            self.helper.getname(
                os.path.join(self.printdir,f'{name}_histogram.png')))
     
    def plot_basic_scatter(self,log_scale=True):
        viz_dict=self.get_basic_meta()
        fig=plt.figure(dpi=200,figsize=[6,5])
        ax=fig.add_subplot(111)
        args=[]
        for key,data in viz_dict.items():
            args.extend([data,key])
        self.dp.my2dscatter(*args,ax=ax,log_scale=log_scale)
        fig.savefig(
            self.helper.getname(
                os.path.join(self.printdir,f'xy_scatter.png')))
    