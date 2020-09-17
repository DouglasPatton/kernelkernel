import logging
import random
import numpy as np
from pisces_data_huc12 import PiscesDataTool
import os
import sklearn
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
from mylogger import myLogger
class datagen(PiscesDataTool,myLogger):
    '''
    
    '''
    #def __init__(self, data_shape=(200,5), ftype='linear', xval_size='same', sparsity=0, xvar=1, xmean=0, evar=1, betamax=10):
    def __init__(self,datagen_dict,):
        myLogger.__init__(self,name='datagen.log')
        self.logger.info('starting new datagen log')
        source=datagen_dict['source']
        self.datagen_dict=datagen_dict
        
        
        if source=='pisces':
            PiscesDataTool.__init__(self,)
            species=datagen_dict['species']
            test_share=0.1
            cv_folds=5
            cv_reps=2
            stratify='balanced-HUC8'
            inner_cv_folds=5
            inner_cv_reps=5
            
           
            cv_count=cv_folds*cv_reps
        
