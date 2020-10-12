import pandas as pd
import numpy as np
from mylogger import myLogger
from  sk_tool import SKToolInitializer
from datagen import dataGenerator
from pi_db_tool import DBTool


## Runners are oriented around the data_gen in 
###the rundict they recieve. the data is not pulled 
###until the node starts it up to reduce q load 
###but requires data be sent around.


class PredictRunner(myLogger):
    # runners are initialized by qmulticluster_master and built by pisces_params.py
    # qmulticluster_node runs passQ and then run(), which adds to teh saveq itself.
    def __init__(self,rundict):
        myLogger.__init__(self,name='PredictRunner.log')
        self.logger.info('starting PredictRunner logger')
        self.rundict=rundict
        self.saveq=None
    def passQ(self,saveq):
        self.saveq=saveq
    def build(self):
        #called on master machine before sending to jobq
        try:
            none_hash_id_list=[key for key,val in self.rundict.items() if key!='data_gen' and val is None]
            if len(none_hash_id_list)>0:
                resultsDBdict=DBTool().resultsDBdict()
                for hash_id in none_hash_id_list:
                    self.rundict[hash_id]=self.resultsDBdict[hash_id]['model']
    def run(self,):
        
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
            savedict={hash_id:predictresult}
            qtry=0
            if self.saveq is None:
                    self.logger.info(f'yielding savedict')
                    return predictresult
            else:
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

    def predict(self,data,model):
        #FitRunner doesnt have equivalent to this
        #  becuase SKToolInitializer and sktool make this stuff happen
            
        if 'make_y' in data.datagen_dict:
            make_y=1
        else:
            make_y=0
        
        n=data.df.shape[0]
        species=data.datagen_dict['species']
        
        if not make_y:
            yhat_stack=[]#[None for _ in range(n_splits)] for __ in range(n_repeats)]
            est_name=model['estimator'][0].name    
            _,cv_test_idx=zip(*list(data.get_split_iterator())) # not using cv_train_idx # can maybe remove  *list?
            cv_count=len(cv_test_idx)
            cv_dict=data.datagen_dict['data_split']['cv']
            n_repeats=cv_dict['n_repeats']
            n_splits=cv_dict['n_splits']
            yhat=np.empty([n,])
            m=0
            for rep in range(n_repeats):
                mstack=[];yhat_list=[]
                for s in range(n_splits):
                    #self.logger.info(f'for {species} & {est_name}, {m}/{cv_count}')
                    model_m=model['estimator'][m]
                    m_idx=cv_test_idx[m]
                    X=data.X_train.iloc[m_idx]
                    #y=data.y_train.iloc[m_idx]
                    mstack.extend(m_idx)
                    try:
                        yhat_list.append((model_m.predict(X)))
                        #keep together in case failed predictions
                    except:
                        yhat_list.append(np.array([np.nan for _ in range(len(m_idx))]))
                        self.logger.exception(f'error with species:{species}, est_name:{est_name}, m:{m}')
                    m+=1
                assert len(mstack)==n, 'uneven length error!'
                yhat[mstack]=np.concatenate(yhat_list,axis=0)#undo shuffle
                yhat_stack.append(yhat.copy()[:,None])#add axis for concatenating  
                    #yhat_stack[r,mstack]=yhat_list# mstack reverses the shuffling done by the cv indices
                    # ,**prediction_kwargs))
                    #self.logger.info(f'y_yhat_tup_list:{y_yhat_tup_list}')
                #y_arr=data.y_train#.iloc[mstack]
            yhat_stack_arr=np.concatenate(yhat_stack,axis=1)
            columns=[f'yhat_{r}' for r in range(n_repeats)]
        else:
            est_name='y_train'
            y=data.y_train
            yhat_stack_arr=y.to_numpy()[:,None] #not really yhat...
            columns=['y']#
        
        huc12s=data.df.loc[:,'HUC12']
        huc12strs=huc12s.apply(self.huc12float_to_str)
        comids=data.y_train.index
        
        names=['species','estimator','HUC12','COMID']
        index=pd.MultiIndex.from_tuples([(species,est_name,huc12strs[i],comids[i])  for i in range(n)],names=names) # reps stacked across columns
        yhat_df=pd.DataFrame(yhat_stack_arr,columns=columns,index=index)
        self.logger.info(f'yhat_df:{yhat_df}')
        return yhat_df
      
    def huc12float_to_str(self,huc12):
        huc12str=str(int(huc12))
        if len(huc12str)==11:huc12str='0'+huc12str
        assert len(huc12str)==12,'expecting len 12 from huc12str:{huc12str}'
        return huc12str
        
    def build_from_rundict(self,rundict):
        
        data_gen=rundict.pop('data_gen') #how to generate the data
        data=dataGenerator(data_gen)
        return data,rundict 
        

class FitRunner(myLogger):
    def __init__(self,rundict):
        myLogger.__init__(self,name='FitRunner.log')
        self.logger.info('starting FitRunner logger')
        self.rundict=rundict
    def passQ(self,saveq):
        self.saveq=saveq
    def build(self):
        pass
    def run(self,):
        
        data,hash_id_model_dict=self.build_from_rundict(self.rundict)
        hash_id_list=list(hash_id_model_dict.keys())
        for hash_id in hash_id_list:
            model_dict=hash_id_model_dict[hash_id]
            try:
                success=0
                model_dict['model']=model_dict['model'].run(data)
                success=1
            except:
                self.logger.exception(f'error for model_dict:{model_dict}')
            savedict={hash_id:model_dict}
            qtry=0
            while success:
                self.logger.debug(f'adding savedict to saveq')
                try:
                    qtry+=1
                    self.saveq.put(savedict)
                    self.logger.debug(f'savedict sucesfully added to saveq')
                    break
                except:
                    if not self.saveq.full() and qtry>3:
                        self.logger.exception('error adding to saveq')
                    else:
                        sleep(1)


                                    
    def build_from_rundict(self,rundict):
        data_gen=rundict['data_gen'] #how to generate the data
        data=dataGenerator(data_gen)
        model_gen_dict=rundict['model_gen_dict'] # {hash_id:data_gen...}
        hash_id_model_dict={}
        for hash_id,model_gen in model_gen_dict.items():
            model_dict={'model':SKToolInitializer(model_gen),'data_gen':data_gen,'model_gen':model_gen}
            hash_id_model_dict[hash_id]=model_dict# hashid based on model_gen and data_gen
        return data,hash_id_model_dict