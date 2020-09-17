import logging
import random
import numpy as np
from pisces_data_huc12 import PiscesDataTool
import os
import sklearn
from sklearn.model_selection import StratifiedKFold,RepeatedKFold,train_test_split
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
        datagen_dict=self.datagen_dict
        PiscesDataTool.__init__(self,)
        species=datagen_dict['species']
        if type(species) is str:
            self.fulldata=retrievespeciesdata(species_name=species)
        elif type(species) is int:
            self.fulldata=retrievespeciesdata(species_idx=species)
        
        self.cv=RepeatedKFold(n_splits=cv_folds, n_repeats=cv_reps, random_state=rs)    
        y_name='presence'
        self.loc_vars=['HUC12']
        drop_vars=datagen_dict['drop_vars']
        drop_vars.extend(self.loc_vars)
        drop_vars.append(y_name)
        self.x_vars=[var for var in all_vars if var not in drop_vars]
        X_df=df.loc[:,self.x_vars]
        y_df=df.loc[:,y_name]
        if test_share:
            X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=test_share, random_state=rs)
        else:
            X_train, X_test, y_train, y_test = (X_df, None, y_df, None)
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        