from sk_tool import SkTool
from pisces_data_huc12 import PiscesDataTool
from mylogger import myLogger
import logging

class PiSetup:
    def __init__(self,myLogger):
        myLogger.__init__(self,name='PiSetup.log')
        self.logger.info('starting PiSetup logger')
        floatselecttup=(3,5,6)
        spatialselecttup=(8,)
        pdt=PiscesDataTool()
        
    
    
    def model_setup(self,):
        #sk_tool uses model_gen to create the estimator
        est_dict=SkTool().get_est_dict()
        model_gen_list=[]
        for est_name in est_dict.keys():
            kwargs={'gridpoints':5}
            model_gen={'kwargs':kwargs,'name':est_name}
            model_gen_list.append(model_gen)
        return model_gen_list
            
    
    def data_setup(self,):
        species_list=pdt.returnspecieslist()
        data_params={
            'source':'pisces'
            'species':None
            'test_share':0.1,
            'cross_validate':False,
            'missing':'drop_row', #drop the row(observation) if any data is missing
            'floatselecttup':floatselecttup,
            'spatialselecttup':spatialselecttup,
            'param_count':param_count,
            'sample':'y-balanced'
        }
        
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
        