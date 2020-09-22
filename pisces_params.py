from sk_tool import SkTool
from pisces_data_huc12 import PiscesDataTool
from mylogger import myLogger
import logging

class PiSetup:
    def __init__(self,)
        pdt=PiscesDataTool()
        self.modelsetupdict=dict(
            gridpoints=5,
            inner_cv_splits=10,
            inner_cv_reps=2,
            )
        self.initial_datagen_dict=dict(
            species=None, # set in data_setup
            data_split=dict(
                test_share=0.1,
                cv=dict(n_folds=10,n_reps=2,strategy=None),# e.g., 'balanced-HUC8'
                drop_vars=[],
                loc_vars=['HUC12'],
                
                )
            )
    
    def model_setup(self,):
        #sk_tool uses model_gen to create the estimator
        est_dict=SkTool().get_est_dict()
        model_gen_list=[]
        for est_name in est_dict.keys():
            kwargs=self.model_setup_dict
            model_gen={'kwargs':kwargs,'name':est_name}
            model_gen_list.append(model_gen)
        return model_gen_list
            
    
    def data_setup(self,):
        species_list=pdt.returnspecieslist()
        cvdict=self.cvdict
        data_params=self.datagen_dict
        
        data_gen_list=[]
        for species in species_list:
            params=copy(data_params)
            params['species']=species
            data_gen_list.append(params)
        return data_gen_list
            
        
    
    
    
    
    
    
class MonteSetup(self,myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='MonteSetup.log')
        self.logger.info('starting MonteSetup logger')
    
    def model_setup(self,):
        est_dict=SkTool().get_est_dict()
        model_gen_list=[]
        for est_name in est_dict.keys():
            kwargs={}
            model_gen={'kwargs':kwargs,'name':est_name}
            model_gen_list.append(model_gen)
        return model_gen_list
    
    def data_setup(self,):
        self.data_params={
            
        }
        