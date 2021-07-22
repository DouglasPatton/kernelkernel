import os
import pickle,json
import pandas as pd
import numpy as np
from sqlitedict import SqliteDict
from mylogger import myLogger
from  sk_tool import SKToolInitializer
from sk_estimators import sk_estimator
from datagen import dataGenerator,XdataGenerator
from pi_db_tool import DBTool


## Runners are oriented around the data_gen in 
###the rundict they recieve. For predictrunners, the data is not pulled 
###until the node starts it up to reduce time/memory for creating runlist 



        
    
class PredictRunner(myLogger):
    # runners are initialized by qmulticluster_master and built by pisces_params.py
    # qmulticluster_node runs passQ, build, and then run(), which adds to the saveq itself.
    #     the build step adds to the runner any needed data found only on the master, such as the result for prediction.
    def __init__(self,rundict):
        myLogger.__init__(self,name='PredictRunner.log')
        self.logger.info('starting PredictRunner logger')
        self.rundict=rundict
        self.saveq=None
    def passQ(self,saveq):
        self.saveq=saveq
    def build(self):
        #called on master machine by jobqfiller before sending to jobq
        try:
            none_hash_id_list=[key for key,val in self.rundict.items() if key!='data_gen' and val is None]
            if len(none_hash_id_list)>0: #fill in None with 'model' from resultsDBdict
                with DBTool().resultsDBdict() as rdb:
                    for hash_id in none_hash_id_list:
                        result=rdb[hash_id]
                        if type(result) is str:
                            result=self.getResult(result)
                        self.rundict[hash_id]=result['model']
                        self.logger.info(f'sucessful rundict build for hash_id:{hash_id}')
        except: 
            self.logger.exception(f'build error with rundict:{rundict}')
            
    def getResult(self,result):
        if type(result) is str:
            if os.path.exists(result):
                try:
                    with open(result,'rb') as f:
                        return pickle.load(f)
                    self.logger.info(f'result loaded from file at path:{result}')
                except:
                    self.logger.exception(f'error loading file from {result}')
            else:
                self.logger.info(f'no file at result:{result}')
        else:
            self.logger.info(f'getResult has result that is not a str: {result}')
        return None
    def run(self,):
        self.ske=sk_estimator()
        data,hash_id_model_dict=self.build_from_rundict(self.rundict)
        hash_id_list=list(hash_id_model_dict.keys())
        for hash_id in hash_id_list:
            model=hash_id_model_dict[hash_id]
            
            try:
                success=0
                predictresult=self.predict(data,model)
                success=1
            except:
                self.logger.exception('error for model_dict:{model_dict}')
            if self.saveq is None and success:
                    self.logger.info(f'no saveq, returning predictresult')
                    return predictresult
            else:
                savedict={hash_id:predictresult}
                qtry=0
                while success:
                    self.logger.debug(f'adding savedict to saveq')
                    try:
                        qtry+=1
                        self.saveq.put(savedict)
                        self.logger.debug(f'savedict successfully added to saveq')
                        break
                    except:
                        if not self.saveq.full() and qtry>3:
                            self.logger.exception('error adding to saveq')
                        else:
                            sleep(1)

                            
    def make_coef_scor_df(self,model,r,s,m,species):
        try:
            est_name=model['estimator'][0].name  

            model_m=model['estimator'][m]
            if est_name in ['logistic-reg','linear-svc','linear-probability-model']: 
                coefs=self.ske.get_coef_from_fit_est(species,est_name,model_m.model_,std_rescale=True)
                x_vars=model_m.x_vars
            else:
                coefs=[]
                x_vars=[]
            scor_names,scors=zip(*[(f'scorer:{key[5:]}',model[key][m]) for key in model.keys() if key[:5]=='test_'])
            coefs=list(coefs)
            arr_list=[*scors,*coefs]
            #arr_list=[arr[:,None] for arr in arr_list]
            #data=np.concatenate(arr_list,axis=1)
            data=np.array(arr_list)[None,:]
            columns=[*scor_names,*x_vars]
            column_tups=[(col,r,s) for col in columns]
            col_midx=pd.MultiIndex.from_tuples(column_tups,names=['var','rep_idx','split_idx'])
            row_tup=[(species,est_name)]
            row_midx=pd.MultiIndex.from_tuples(row_tup,names=['species','estimator'])
            coef_scor_df=pd.DataFrame(data=data,columns=col_midx,index=row_midx)
            return coef_scor_df
        except:
            self.logger.exception(f'outer catch')
            assert False,'unexpected'
                            
    def predict(self,data,model):
        #FitRunner doesnt have equivalent to this
        #  becuase SKToolInitializer and sktool make this stuff happen
            
        n=data.y_train.shape[0]
        species=data.datagen_dict['species']
        
        #yhat_stack=[]#[None for _ in range(n_splits)] for __ in range(n_repeats)]
        
        if type(model) is dict:
            cv_run=True
            self.logger.info(f'predicting for cv result')
            est_name=model['estimator'][0].name    
            _,cv_test_idx=zip(*list(data.get_split_iterator())) # not using cv_train_idx # can maybe remove  *list?
            cv_count=len(cv_test_idx)
            cv_dict=data.datagen_dict['data_split']['cv']
            n_repeats=cv_dict['n_repeats']
            n_splits=cv_dict['n_splits']
            self.logger.info(f'n_repeats:{n_repeats}, n_splits:{n_splits}')
        else:
            cv_run=False
            self.logger.info(f'predicting for non-CV result')
            n_repeats=1
            n_splits=1
            cv_test_idx=[np.arange(n)]
            est_name=model.name
        yhat=np.empty([n,])
        yhat[:]=np.nan
        m=0
        coef_scor_df_list=[];col_tup_list=[]
        yhat_list=[]
        for rep in range(n_repeats):
            #mstack=[]
            
            for s in range(n_splits):
                #self.logger.info(f'for {species} & {est_name}, {m}/{cv_count}')
                if cv_run:
                    model_m=model['estimator'][m]
                else:
                    model_m=model
                coef_scor_df_m=self.make_coef_scor_df(model,rep,s,m,species)
                coef_scor_df_list.append(coef_scor_df_m)
                
                m_idx=cv_test_idx[m]
                
                X=data.X_train.iloc[m_idx]
                #mstack.extend(m_idx)
                try:
                    y_=model_m.predict(X)
                    yhat_=yhat.copy() #all np.nan's
                    yhat_[m_idx]=y_
                    yhat_list.append(yhat_)
                    #keep together in case failed predictions
                except:
                    yhat_list.append(yhat.copy())
                    self.logger.exception(f'error with species:{species}, est_name:{est_name}, m:{m}')
                col_tup_list.append(('y',rep,s))
                m+=1
            #assert len(mstack)==n, 'uneven length error!'
            #yhat[mstack]=np.concatenate(yhat_list,axis=0)#undo shuffle
            #yhat_stack.append(yhat.copy()[:,None])#add axis for concatenating 
        yhat_list=[y[:,None] for y in yhat_list] # make columns for concatenation
        yhat_stack_arr=np.concatenate(yhat_list,axis=1)
        
        #col_tups=[('yhat',r,s) for r in range(n_repeats)]
        
        columns=pd.MultiIndex.from_tuples(col_tup_list,names=['var','rep_idx','split_idx'])
        huc12s=data.df.loc[:,'HUC12']
        huc12strs=huc12s.apply(self.huc12float_to_str)
        comids=data.y_train.index
        
        names=['species','estimator','HUC12','COMID']
        index=pd.MultiIndex.from_tuples([(species,est_name,huc12strs[i],comids[i])  for i in range(n)],names=names) # reps stacked across columns
        self.logger.info(f'yhat_stack_arr.shape:{yhat_stack_arr.shape}, yhat_stack_arr:{yhat_stack_arr}')
        self.logger.info(f'columns.shape:{columns.shape}, columns:{columns}')
        self.logger.info(f'index:{index}')
        yhat_df=pd.DataFrame(yhat_stack_arr,columns=columns,index=index)
        self.logger.info(f'yhat_df:{yhat_df}')
        
        ########create y_df
        y=data.y_train
        y_stack_arr=y.to_numpy()
        columns=['y']
        names=['species','HUC12','COMID']
        index=pd.MultiIndex.from_tuples([(species,huc12strs[i],comids[i])  for i in range(n)],names=names) # reps stacked across columns
        y_df=pd.DataFrame(y_stack_arr,columns=columns,index=index)
        self.logger.info(f'y_df:{y_df}')
        #### create coef_scor_df
        full_coef_scor_df=pd.concat(coef_scor_df_list,axis=1)
        
        return {'yhat':yhat_df,'y':y_df,'coef_scor_df':full_coef_scor_df}
      
    def huc12float_to_str(self,huc12):
        huc12str=str(int(huc12))
        if len(huc12str)==11:huc12str='0'+huc12str
        assert len(huc12str)==12,'expecting len 12 from huc12str:{huc12str}'
        return huc12str
        
    def build_from_rundict(self,rundict):
        
        data_gen=rundict.pop('data_gen') #how to generate the data
        data=dataGenerator(data_gen)
        return data,rundict 

    
    
class XPredictRunner:#(PredictRunner):
    def __init__(self,rundict):
        myLogger.__init__(self,name='XPredictRunner.log')
        self.logger.info('starting XPredictRunner logger')
        self.rundict=rundict
        self.saveq=None
        #PredictRunner.__init__(self,rundict)
        self.hash_id_c_hash_dict=None
        
    def passQ(self,saveq):
        self.saveq=saveq
                             
    def getResult(self,result):
        if type(result) is str:
            if os.path.exists(result):
                try:
                    with open(result,'rb') as f:
                        return pickle.load(f)
                except:
                    self.logger.exception(f'error loading file from {result}')
            else:
                self.logger.info(f'no file at result:{result}')
        else:
            self.logger.info(f'getResult has result that is not a str: {result}')
        return None
                             
    def build(self):
        #called on master machine by jobqfiller before sending to jobq
        try:
            none_hash_id_list=[key for key,val in self.rundict.items() if key!='data_gen' and val is None]
            if len(none_hash_id_list)>0: #fill in None with 'model' from resultsDBdict
                resultsDBdict=DBTool().resultsDBdict()
                for hash_id in none_hash_id_list:
                    result=resultsDBdict[hash_id]
                    if type(result) is str:
                        result=self.getResult(result)
                    self.rundict[hash_id]=result['model']
                    self.logger.info(f'sucessful rundict build for hash_id:{hash_id}')
                    
            data,rundict=self.build_from_rundict(self.rundict.copy())
            comidblockhashdict=data.getXpredictSpeciesComidBlockDict()[data.spec]
            hash_id_list=rundict.keys()
            self.hash_id_c_hash_dict=self.checkXPredictHashIDComidHashResults(hash_id_list,comidblockhashdict)
            self.logger.info(f'{data.spec} predictrunner built')
                    
            
        except: 
            self.logger.exception(f'build error with rundict:{rundict}')
    
    
    def huc12float_to_str(self,huc12):
        huc12str=str(int(huc12))
        if len(huc12str)==11:huc12str='0'+huc12str
        assert len(huc12str)==12,'expecting len 12 from huc12str:{huc12str}'
        return huc12str
    
    
    def checkXPredictHashIDComidHashResults(self,hash_ids,comidblockdict):
        dbt=DBTool()
        hash_id_c_hash_dict={}
        hash_ids_in_results=dbt.XPredictHashIDComidHashResultsDB(hash_id=None)#tablenames
        for hash_id in hash_ids:
            if hash_id in hash_ids_in_results:
                with dbt.XPredictHashIDComidHashResultsDB(hash_id=hash_id) as hash_resultsdb:
                    hash_c_hash_with_results=dict.fromkeys(hash_resultsdb.keys())
                hash_id_c_hash_dict[hash_id]=[]

                for c_hash in comidblockdict.keys():
                    if not c_hash in hash_c_hash_with_results:
                        hash_id_c_hash_dict[hash_id].append(c_hash)
                    else:
                        self.logger.info(f'hash_id:{hash_id},c_hash:{c_hash} already complete')
                if len(hash_id_c_hash_dict[hash_id])==0:
                    with dbt.XpredictDBdict() as done_db:
                        done_db[hash_id]='complete'
                        done_db.commit()
                    self.logger.info(f'hash_id:{hash_id} is complete and added to XpredictDBdict as "complete"')
            else:
                hash_id_c_hash_dict[hash_id]=list(comidblockdict.keys())
        return hash_id_c_hash_dict
                    
        
    def build_from_rundict(self,rundict):
        data_gen=rundict.pop('data_gen') #how to generate the data
        data=XdataGenerator(data_gen)
        return data,rundict   
    
    
        
    
    def run(self,):
        #self.hash_id_c_hash_dict
        c_hash_hash_id_dict={}#just reversing the dict
        for hash_id,c_hash_list in self.hash_id_c_hash_dict.items():
            for c_hash in c_hash_list:
                if not c_hash in c_hash_hash_id_dict:
                    c_hash_hash_id_dict[c_hash]=[hash_id]
                else:
                    c_hash_hash_id_dict[c_hash].append(hash_id)
                    
        self.ske=sk_estimator()
        data,hash_id_model_dict=self.build_from_rundict(self.rundict)
        """for hash_id,model in hash_id_model_dict.items():
            for skt in model['estimator']:
                if 'HUC12' in skt.x_vars:
                    self.logger.error(f'HUC12 found in hash_id:{hash_id}')
                    assert False, f'huc12 error for hash_id:{hash_id}, {data.spec}'"""
        keylist=data.datagen_dict['x_vars']
        keylist.append('HUC12')
        comidblockdict=data.getXpredictSpeciesComidBlockDict()[data.spec]
        for c_hash,hash_id_list in c_hash_hash_id_dict.items():
            
            comidlist=comidblockdict[c_hash]
            
            datadf=data.generateXPredictBlockDF(
                data.spec,comidlist=comidlist,keylist=keylist)
            self.logger.info(f'')
            
            
        
            for hash_id in hash_id_list:
                model=hash_id_model_dict[hash_id]
                predictresult=None
                try:
                    success=0
                    predictresult={hash_id:{c_hash:self.Xpredict(datadf,data,model,hash_id)}}
                    self.logger.info(f'predictresult:{predictresult}')
                    success=1
                except:
                    self.logger.exception('error for model_dict:{model_dict}')
                if self.saveq is None and success:
                        self.logger.info(f'no saveq, returning predictresult')
                        return predictresult
                elif not success:
                    self.logger.error(f'failure for c_hash:{c_hash}, hash_id:{hash_id}, predictresult:{predictresult}')
                else:
                    qtry=0
                    while True:
                        self.logger.debug(f'adding predictresult to saveq')
                        try:
                            qtry+=1
                            self.saveq.put(predictresult)
                            self.logger.debug(f'savedict successfully added to saveq')
                            break
                        except:
                            if not self.saveq.full() and qtry>3:
                                self.logger.exception('error adding to saveq')
                            else:
                                sleep(1)

    
    def Xpredict(self,datadf,data,model,hash_id):
        if type(model) is dict:
            is_cv_model=True
            est_name=model['estimator'][0].name  
            train_vars=list(model['estimator'][0].x_vars)
            cv_dict=data.datagen_dict['data_split']['cv']
            n_repeats=cv_dict['n_repeats']
            n_splits=cv_dict['n_splits']
        else:
            is_cv_model=False
            est_name=model.name
            train_vars=list(model.x_vars)
            n_repeats=1
            n_splits=1
            
        n=datadf.shape[0]
        species=data.spec
        huc12s=datadf.loc[:,'HUC12']
        huc12strs=huc12s.apply(self.huc12float_to_str)
        Xdf=datadf.drop('HUC12',axis=1)
        Xdf=Xdf.loc[:,train_vars] # make sure order matches and all variables present
        predict_vars=list(Xdf.columns)
        
        #check to make sure data matches model
        self.logger.info(f'starting an Xpredict for {species} - {est_name} ')
        if len(predict_vars)!=len(train_vars):
            self.logger.error(f'{species} predict_vars:{predict_vars}')
            self.logger.error(f'{species} train_vars:{train_vars}')
        assert all([predict_vars[i]==train_vars[i] for i in range(len(train_vars))]),f'predict and train vars should match, but predict_vars:{predict_vars} and train_vars:{train_vars}'
            
        
        yhat_list=[]#[None for _ in range(n_splits)] for __ in range(n_repeats)]
         
        cv_count=n_repeats*n_splits
        self.logger.info(f'n_repeats:{n_repeats}, n_splits:{n_splits}')
        m=0;col_tup_list=[]
        for rep in range(n_repeats):
            for s in range(n_splits):
                if is_cv_model:
                    model_m=model['estimator'][m]  
                else:
                    model_m=model
                try:
                    self.logger.info(f'about to predict {m+1} of {cv_count}')
                    yhat=model_m.predict(Xdf)
                    yhat_list.append(yhat)
                    col_tup_list.append(('y',rep,s))
                except:
                    self.logger.exception(f'error with species:{species}, est_name:{est_name},m:{m}')

                m+=1
        yhat_list=[y[:,None] for y in yhat_list] # make into 2d as columns for concatenation
        yhat_stack_arr=np.concatenate(yhat_list,axis=1)


        columns=pd.MultiIndex.from_tuples(
            col_tup_list,names=['var','rep_idx','split_idx'])
        """else:
            yhat=model.predict(Xdf)
            self.logger.info(f'no data_split, no cv for {species}, {est_name}')
            yhat_stack_arr=yhat[:,None]#make a column
            columns=['y']"""
            
        comids=datadf.index
        
        names=['species','estimator','HUC12','COMID']
        index=pd.MultiIndex.from_tuples([(species,est_name,huc12strs[i],comids[i])  for i in range(n)],names=names) # reps stacked across columns
        self.logger.info(f'yhat_stack_arr.shape:{yhat_stack_arr.shape}, yhat_stack_arr:{yhat_stack_arr}')
        #self.logger.info(f'columns.shape:{columns.shape}, columns:{columns}')
        self.logger.info(f'index:{index}')
        yhat_df=pd.DataFrame(yhat_stack_arr,columns=columns,index=index)
        self.logger.info(f'yhat_df.shape:{yhat_df.shape}')
        return yhat_df
        
        
        
        
        
        
    

class FitRunner(myLogger):
    def __init__(self,rundict):
        myLogger.__init__(self,name='FitRunner.log')
        self.logger.info('starting FitRunner logger')
        self.rundict=rundict
    def passQ(self,saveq):
        self.saveq=saveq
    def build(self):
        pass
    def run(self,cv_n_jobs=None):
        
        data,hash_id_model_dict=self.build_from_rundict(self.rundict)
        hash_id_list=list(hash_id_model_dict.keys())
        for hash_id in hash_id_list:
            model_dict=hash_id_model_dict[hash_id]
            try:
                success=0
                data.X_train 
                self.logger.info(f'fitrunner running hash_id:{hash_id}')
                if cv_n_jobs is None:
                    model_dict['model']=model_dict['model'].run(data)
                else:
                    model_dict['model']=model_dict['model'].run(data,cv_n_jobs=cv_n_jobs)
                success=1
            except:
                self.logger.exception(f'error for model_dict:{model_dict}')
            savedict={hash_id:model_dict}
            if not success:
                self.logger.warning(f'adding fitfail key to savedict.')
                savedict['fitfail']=True
            qtry=0
            while True: #saving regardless of success
                self.logger.debug(f'adding savedict to saveq')
                try:
                    qtry+=1
                    self.saveq.put(savedict)
                    self.logger.debug(f'savedict sucesfully added to saveq')
                    break
                except:
                    if not self.saveq.full() and qtry>3:
                        self.logger.exception('error adding to saveq, continuing to try')
                    else:
                        sleep(5*qtry)


                                    
    def build_from_rundict(self,rundict):
        data_gen=rundict['data_gen'] #how to generate the data
        data=dataGenerator(data_gen)
        model_gen_dict=rundict['model_gen_dict'] # {hash_id:data_gen...}
        hash_id_model_dict={}
        for hash_id,model_gen in model_gen_dict.items():
            model_dict={'model':SKToolInitializer(model_gen),'data_gen':data_gen,'model_gen':model_gen}
            hash_id_model_dict[hash_id]=model_dict# hashid based on model_gen and data_gen
        return data,hash_id_model_dict
