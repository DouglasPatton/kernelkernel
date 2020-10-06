import logging
import random
import numpy as np
from pisces_data_huc12 import PiscesDataTool
import os
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
        try:
            source=datagen_dict['source']
            PiscesDataTool.__init__(self,)
            species=datagen_dict['species']
            self.logger.info(f'datagen generating species:{species}')
            if type(species) is str:
                self.df=self.retrievespeciesdata(species_name=species)
            elif type(species) is int:
                self.df=self.retrievespeciesdata(species_idx=species)
            self.logger.info(f'type(self.df): {type(self.df)}')
            self.logger.info(f'self.df:{self.df}')
            all_vars=list(self.df.columns)

            col_convert={}
            for col in all_vars:
                if self.df[col].dtype=='object':
                    self.logger.warning(f'converting {col} to float')
                    col_convert[col]='float64'
            self.df=self.df.astype(col_convert)
            n=self.df.shape[0]
            self.n=n
            
            min_n=self.datagen_dict['min_sample']
            assert n>=min_n,f'aborting species:{species} because n:{n}<{min_n}'
            
            if self.datagen_dict['shuffle']:
                shuf=np.arange(n)
                np.random.shuffle(shuf)
                self.df=self.df.iloc[shuf]
            data_split_dict=self.datagen_dict['data_split']
            test_share=data_split_dict['test_share']

            y_name='presence'
            loc_vars=datagen_dict['loc_vars']
            drop_vars=datagen_dict['drop_vars']
            drop_vars.extend(loc_vars)
            drop_vars.append(y_name)
            self.x_vars=[var for var in all_vars if var not in drop_vars]
            self.datagen_dict['x_vars']=self.x_vars
            X_df=self.df.loc[:,self.x_vars]
            y_df=self.df.loc[:,y_name]
            try:
                min_1count=self.datagen_dict['min_1count']
            except KeyError:
                self.logger.warning(f'min_1count not in data_gen for species:{species}')
                min_1count=0
            except:
                assert False,'unexpected error!'
                
            count1=np.sum(y_df)
            assert count1>=min_1count,f'aborting species:{species} because count1:{count1}<min_1count{min_1count}'
            
            
            self.ymean=np.mean(y_df)
            try:random_state=self.datagen_dict['random_state']
            except KeyError:
                random_state=None
            if test_share:
                X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=test_share, random_state=random_state)
            else:
                X_train, X_test, y_train, y_test = (X_df, None, y_df, None)
            self.logger.info(f'for {species} X_train.shape:{X_train.shape},y_train.shape:{y_train.shape}')
            self.X_train=X_train
            self.X_test=X_test
            self.y_train=y_train
            self.y_test=y_test
            if data_split_dict['cv']:
                if count1<data_split_dict['cv']['n_splits']:
                    old_n_splits=data_split_dict['cv']['n_splits']
                    new_n_splits=count1//4
                    assert new_n_splits>=2,f'new_n_splits:{new_n_splits} must be >=2'
                    data_split_dict['cv']['n_splits']=new_n_splits # 
                    old_n_reps=data_split_dict['cv']['n_repeats']
                    new_n_reps=-(-(old_n_reps*old_n_splits)//new_n_splits) # ceiling divide
                    data_split_dict['cv']['n_repeats']=n_n_reps
                    self.logger.critical(f'split change for species:{species} new_n_splits:{new_n_splits}, new_n_reps:{new_n_reps},old_n_splits:{old_n_splits},old_n_reps:{old_n_reps}')
                cv_kwargs={key:val for key,val in data_split_dict['cv'].items() if not val is None}
                self.cv=RepeatedKFold(**cv_kwargs)  
                self.logger.info(f'cv set for species:{species}')
            else:
                self.cv=None
                self.logger.info(f'NO CV for species:{species}. datagen_dict:{datagen_dict}, data_split_dict:{data_split_dict}')
        except:
            self.logger.exception(f'datagen outer catch')
            
    def get_split_iterator(self,):
        if self.cv:
            return self.cv.split(self.X_train,self.y_train)
        else:
            return[(X_train,y_train)]
        


        
