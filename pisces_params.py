#from sk_tool import SkTool
from sk_estimators import sk_estimator
from pisces_data_huc12 import PiscesDataTool
from mylogger import myLogger
import logging
from copy import copy
import joblib

class PiSetup(myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='PiSetup.log')
        self.logger.info('starting PiSetup logger')
        self.pdt=PiscesDataTool()
        self.est_dict=sk_estimator().get_est_dict()
        self.model_setup_dict=dict(
            gridpoints=3,
            inner_cv_splits=5,
            inner_cv_reps=1,
            )
        self.initial_datagen_dict=dict(
            source='Pisces',
            species=None, # set in data_setup
            data_split=dict(
                test_share=0.1,
                cv=dict(n_splits=5,n_repeats=1,strategy=None),# e.g., 'balanced-HUC8'
            ),
            drop_vars=[],
            loc_vars=['HUC12'],
            )
    
    def model_setup(self,):
        #sk_tool uses model_gen to create the estimator
        model_gen_list=[]
        for est_name in self.est_dict.keys():
            kwargs=self.model_setup_dict
            model_gen={'kwargs':kwargs,'name':est_name}
            model_gen_list.append(model_gen)
        return model_gen_list
            
    
    def data_setup(self,):
        species_list=self.pdt.returnspecieslist()
        data_params=self.initial_datagen_dict
        
        data_gen_list=[]
        for species in species_list:
            params=copy(data_params)
            params['species']=species
            data_gen_list.append(params)
        return data_gen_list
            
    
    def setupRundictList(self,):
        #a run dict has a 
        #model_gen_dict of models per instance of data_gen
        #each of those model-data combos has a hash_id built from
        #the model_dict and data_gen
        run_dict_list=[]
        run_record_dict={}
        data_gen_list=self.data_setup()
        
        for data_gen in data_gen_list:
            model_gen_dict={}
            model_gen_list=self.model_setup()
            for model_gen in model_gen_list:
                run_record={'model_gen':model_gen,'data_gen':data_gen}
                hash_id=joblib.hash(run_record)
                run_record_dict[hash_id]=run_record
                model_gen_dict[hash_id]=model_gen
            run_dict={'data_gen':data_gen, 'model_gen_dict':model_gen_dict}
            run_dict_list.append(run_dict)
        
        return run_dict_list,run_record_dict
    
    
    
    
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
        