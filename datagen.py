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
        
        
    def genpiscesbatchbatchlist(self, ydataarray,xdataarray,batch_n,batchcount,min_y=1):
        #needs updating!
        
        # test data already removed
        batchbatchcount=self.batchbatchcount
        n=ydataarray.shape[0]; p=xdataarray.shape[1]
        onecount=int(ydataarray.sum())
        zerocount=n-onecount
        countlist=[zerocount,onecount]
        if onecount<zerocount:
            smaller=1
        else:
            smaller=0
        
        if not min_y is None:
            if min_y<1:
                min_y=int(batch_n*min_y)
            batch01_n=[None,None]
            batch01_n[smaller]=min_y # this makes it max_y too....
            batch01_n[1-smaller]=batch_n-min_y
            max_batchcount=countlist[smaller]//min_y
            if max_batchcount<batchcount:
                #self.logger.info(f'for {self.species} batchcount changed from {batchcount} to {max_batchcount}')
                batchcount=max_batchcount
            subsample_n=batchcount*batch_n
            oneidx=np.arange(n)[ydataarray==1]
            zeroidx=np.arange(n)[ydataarray==0]
            bb_select_list=[]
            for bb_idx in range(batchbatchcount):
                ones=np.random.choice(oneidx,size=batch01_n[1]*batchcount,replace=False)
                zeros=np.random.choice(zeroidx,size=batch01_n[0]*batchcount,replace=False)
                bb_select_list.append(np.concatenate([ones,zeros],axis=0))
        else:
            max_batchcount=n//batch_n
            if max_batchcount<batchcount:
                #self.logger.info(f'for {self.species} batchcount changed from {batchcount} to {max_batchcount}')
                batchcount=max_batchcount
            subsample_n=batchcount*batch_n
            for bb_idx in range(batchbatchcount):
                bb_select_list.append(np.random.choice(np.arange(n),size=subsample_n),replace=False)
        batchbatchlist=[[None for __ in range(batchcount)] for _ in range(batchbatchcount)]
        SKF=StratifiedKFold(n_splits=batchcount, shuffle=False)
        for bb_idx in range(batchbatchcount):
            bb_x_subsample=xdataarray[bb_select_list[bb_idx],:]
            bb_y_subsample=ydataarray[bb_select_list[bb_idx]]
            for j,(train_index,test_index) in enumerate(SKF.split(bb_x_subsample,bb_y_subsample)):
                batchbatchlist[bb_idx][j]=(bb_y_subsample[test_index],bb_x_subsample[test_index,:])
        batchsize=batch_n*batchcount
        
        
        self.batchcount=batchcount
        self.expand_datagen_dict('batchcount',self.batchcount)
        fullbatchbatch_n=batchbatchcount*batchsize
        self.fullbatchbatch_n=fullbatchbatch_n
        self.expand_datagen_dict('fullbatchbatch_n',self.fullbatchbatch_n)
        self.logger.info(f'yxtup shapes:{[(yxtup[0].shape,yxtup[1].shape) for yxtuplist in batchbatchlist for yxtup in yxtuplist]}')
        self.yxtup_batchbatch=batchbatchlist
        
        
        