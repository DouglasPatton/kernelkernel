#from sk_tool import SkTool
from sk_estimators import sk_estimator
from pisces_data_huc12 import PiscesDataTool
from mylogger import myLogger
import logging
from copy import copy
import joblib
from random import randint
from pi_db_tool import DBTool
from pi_runners import FitRunner

class PiSetup(myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='PiSetup.log')
        self.logger.info('starting PiSetup logger')
        self.fit_run=True
        rs=1
        splits=5
        
        self.permutation_kwargs=dict(
            n_repeats=5,
            random_state=rs,
            #scoring:None
            
            )
        self.model_setup_dict=dict(
            gridpoints=5,
            inner_cv_splits=splits,
            inner_cv_reps=1,
            random_state=rs # for inner cv and some estimators
            )
        
        self.datagen_dict_template=dict(
            min_sample=32,
            min_1count=8, # at least 4 ones per split
            shuffle=True,
            source='Pisces',
            species='all', # or a range, i.e., (0,100) # set in data_setup
            data_split=dict(
                test_share=0,
                cv=dict(n_splits=splits,
                        n_repeats=20,
                        strategy=None, # e.g., 'balanced-HUC8'
                        random_state=rs),
            ),
            drop_vars=[],
            loc_vars=['HUC12','COMID'],
            )
    
    def model_setup(self,):
        #sk_tool uses model_gen to create the estimator
        model_gen_list=[]
        for est_name in sk_estimator().get_est_dict().keys():
            kwargs=self.model_setup_dict
            model_gen={'kwargs':kwargs,'name':est_name}
            model_gen_list.append(model_gen)
        return model_gen_list
            
    
    def data_setup(self,):
        species_list=PiscesDataTool.returnspecieslist()
        data_params=self.datagen_dict_template
        if not data_params['species']=='all':
            species_selector=data_params['species']
            species_list=[species_list[i] for i in range(*species_selector)]
        data_gen_list=[]
        for species in species_list:
            params=copy(data_params)
            params['species']=species
            data_gen_list.append(params)
        return data_gen_list
     
        
    
    def setupRunners(self,):
        #a run dict has a 
        #model_gen_dict of models per instance of data_gen
        #each of those model-data combos has a hash_id built from
        #the model_dict and data_gen
        dbt=DBTool()
        if self.fit_run:
            rundict_list=[]
            run_record_dict={}
            data_gen_list=self.data_setup()
            hash_id_list=[]
            for data_gen in data_gen_list:
                model_gen_dict={}
                model_gen_list=self.model_setup()
                for model_gen in model_gen_list:
                    run_record={'model_gen':model_gen,'data_gen':data_gen}
                    hash_id=joblib.hash(run_record)
                    hash_id_list.append(hash_id)
                    run_record_dict[hash_id]=run_record # store the _gen dicts for reference
                    model_gen_dict[hash_id]=model_gen # 
                rundict={'data_gen':data_gen, 'model_gen_dict':model_gen_dict}
                rundict_list.append(rundict)

            dbt.addToDBDict(run_record_dict,gen=1) # create a record of the rundicts to check when it's all complete.
            self.logger.debug(f'len(rundict_list):{len(rundict_list)}')
            runlist=[]
            rundict_list=self.checkComplete(rundict_list=rundict_list) # remove any that are already in resultsDB
            for rundict in list_of_rundicts:
                runlist.append(FitRunner(rundict))

        elif self.predict_run:
            pi_results.build_prediction_rundict()
            for rundict in rundict_list:
                runlist.append(PredictRunner(rundict))
            
        return runlist,hash_id_list
    
       
            

        
    def checkComplete(self,db=None,rundict_list=None,hash_id_list=None):
        #rundict_list is provided at startup, and if not, useful for checking if all have been saved
        if db is None:
            db=self.resultsDBdict()
        complete_hash_id_list=list(db.keys())
        if rundict_list:
            for r,run_dict in enumerate(rundict_list):
                model_gen_dict=run_dict['model_gen_dict']
                for hash_id in list(model_gen_dict.keys()): #so model_gen_dict can be changed
                    if hash_id in complete_hash_id_list:
                        del model_gen_dict[hash_id]
                        self.logger.info(f'checkComplete already completed hash_id:{hash_id}')
            return rundict_list
        else:
            for hash_id in hash_id_list:
                if not hash_id in complete_hash_id_list:
                    return False
        return True # must be done
        
    
class MonteSetup(myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='MonteSetup.log')
        self.logger.info('starting MonteSetup logger')
    
    def model_setup(self,):
        est_dict=sk_estimator().get_est_dict()
        model_gen_list=[]
        for est_name in est_dict.keys():
            kwargs={}
            model_gen={'kwargs':kwargs,'name':est_name}
            model_gen_list.append(model_gen)
        return model_gen_list
    
    def data_setup(self,):
        self.data_params={
            
        }
        
