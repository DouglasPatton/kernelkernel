#from sk_tool import SkTool
from sk_estimators import sk_estimator
from pisces_data_huc12 import PiscesDataTool
from mylogger import myLogger
import logging
from copy import copy
import joblib
from random import randint

class PiSetup(myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='PiSetup.log')
        self.logger.info('starting PiSetup logger')
        self.pdt=PiscesDataTool()
        self.est_dict=sk_estimator().get_est_dict()
        rs=randint(0,1e6) # to easily change random_states
        splits=5
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
                cv=dict(n_splits=splits,n_repeats=1,strategy=None,random_state=rs),# e.g., 'balanced-HUC8'
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
                run_record_dict[hash_id]=run_record # store the _gen dicts for reference
                model_gen_dict[hash_id]=model_gen # 
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
        