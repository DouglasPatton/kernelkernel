import logging
import random
import numpy as np
from pisces_data_huc12 import PiscesDataTool
import os
import sklearn
from sklearn.model_selection import StratifiedKFold,RepeatedKFold,train_test_split,RepeatedStratifiedKFold,cross_validate
from sklearn.preprocessing import StandardScaler
from mylogger import myLogger



class dataGenerator(PiscesDataTool,myLogger):
    '''
    
    '''
    #def __init__(self, data_shape=(200,5), ftype='linear', xval_size='same', sparsity=0, xvar=1, xmean=0, evar=1, betamax=10):
    def __init__(self,datagen_dict,):
        myLogger.__init__(self,name='datagen.log')
        self.logger.info('starting new datagen log')
        self.datagen_dict=datagen_dict
        source=datagen_dict['source']
        PiscesDataTool.__init__(self,)
        species=datagen_dict['species']
        if type(species) is str:
            self.df=self.retrievespeciesdata(species_name=species)
        elif type(species) is int:
            self.df=self.retrievespeciesdata(species_idx=species)
        all_vars=list(self.df.columns)
        
        col_convert={}
        for col in all_vars:
            if self.df[col].dtype=='object':
                self.logger.warning(f'converting {col} to float')
                col_convert[col]='float64'
        self.df=self.df.astype(col_convert)
        data_split_dict=self.datagen_dict['data_split']
        test_share=data_split_dict['test_share']
        
        y_name='presence'
        loc_vars=datagen_dict['loc_vars']
        drop_vars=datagen_dict['drop_vars']
        drop_vars.extend(loc_vars)
        drop_vars.append(y_name)
        self.x_vars=[var for var in all_vars if var not in drop_vars]
        X_df=self.df.loc[:,self.x_vars]
        y_df=self.df.loc[:,y_name]
        if test_share:
            X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=test_share, random_state=0)
        else:
            X_train, X_test, y_train, y_test = (X_df, None, y_df, None)
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        if data_split_dict['cv']:
            cv_kwargs={key:val for key,val in data_split_dict['cv'].items() if not val is None}
            self.cv=RepeatedKFold(**cv_kwargs)    
        else:
            self.cv=None
            
    def get_split_iterator(self,):
        if self.cv:
            return self.cv.split(X_train,y_train)
        else:
            return[(X_train,y_train)]
        


        