from pisces_data_huc12 import PiscesDataTool
import os
import logging
import pickle,json
import numpy as np

class XgboostProcessTool(PiscesDataTool):
    def __init__(self):
        
        self.cwd=os.getcwd()
        self.logdir=os.path.join(self.cwd,'log')

        
        
        
        if not os.path.exists(self.logdir):os.mkdir(self.logdir)
        handlername=os.path.join(self.logdir,f'XgboostProcessTool.log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=1)],
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        #self.logger = logging.getLogger(__name__)
        self.logger.info('started XGboostProcessTool log')
        PiscesDataTool.__init__(self)
        
        self.xgboost_data_path=os.path.join(self.savedir,'xgboost_data.pickle')
        self.species_mse_array_dict_path=os.path.join(self.savedir,'species_mse_array_dict.json')
        
        
        
    def preprocess_xgboost_csv(self,filename):
        try:
            with open(self.xgboost_data_path,'rb') as f:
                self.spec_row_dict,self.spec_idx_dict,self.mismatchlist=pickle.load(f)
                self.logger.info('sucessfully opened self.xgboost_data_path')
            return
        except FileNotFoundError:
            self.logger.info(f'no file found at self.xgboost_data_path:{self.xgboost_data_path}')
        except:
            self.logger.exception('pickle file found but cannot be opened')
        boostdatadict=self.getcsvfile(filename)
        print(type(boostdatadict))
        print([item for item in boostdatadict][0:10])
        try:self.specieslist
        except:
            self.returnspecieslist()
            
        spec_idx_dict={spec_name:[] for spec_name in self.specieslist}
        spec_row_dict={spec_name:[] for spec_name in self.specieslist}
        mismatchlist=[]
        for idx,row in enumerate(boostdatadict[1:]): #skip first row since it is just header
            spec_name=row['species_name']
            if spec_name not in spec_idx_dict:
                self.logger.critical(f'spec_name:{spec_name} not found in spec_idx_dict')
                mismatchlist.append((spec_name,idx))
                spec_row_dict[spec_name]=[]
                spec_idx_dict[spec_name]=[]
            spec_row_dict[spec_name].append([row['presence'],row['p_hat']]) #as an array dims:(obs,depvar)
            spec_idx_dict[spec_name].append(idx)
        if len(mismatchlist):
            self.logger.critical(f'mismatchlist(name,idx):{mismatchlist}')
        
        self.spec_row_dict=spec_row_dict
        self.spec_idx_dict=spec_idx_dict
        self.mismatchlist=mismatchlist
        
        with open(self.xgboost_data_path,'wb') as f:
            pickle.dump((spec_row_dict,spec_idx_dict,mismatchlist),f)
                                
                                     
    def process_xgboost(self,):
        try: spec_row_dict=self.spec_row_dict
        except:
            xgboost_data_path=os.path.join(self.savedir,'xgboost_data.pickle')
            with open(xgboost_data_path,'rb') as f:
                spec_row_dict,spec_idx_dict,mismatchlist=pickle.load(f)
        mse_spec_dict={spec_name:None for spec_name in spec_row_dict}
        for spec_name in spec_row_dict:
            specdatalist=spec_row_dict[spec_name]
            if not specdatalist: # empty list will evaluate as False, not False is True
                self.logger.critical(f'problem with {spec_name}, specdatalist:{specdatalist}')
            else:
                spec_data_array=np.array(specdatalist,dtype=np.float64) #dims:(obs,depvar) depvar0 is y, depvar1 is yhat
                avg_threshold0=np.mean(spec_data_array[:,1][spec_data_array[:,0]==0])
                avg_threshold1=np.mean(spec_data_array[:,1][spec_data_array[:,0]==1])
                med_threshold0=np.median(spec_data_array[:,1][spec_data_array[:,0]==0])
                med_threshold1=np.median(spec_data_array[:,1][spec_data_array[:,0]==1])
                avgavg=(avg_threshold0+avg_threshold1)/2
                avgmed=(med_threshold0+med_threshold1)/2
                wtavgavg=np.mean(spec_data_array[:,1])
                
                thresholds=np.array([45,50,55,avgavg,wtavgavg,avgmed,1,0])
                yhat_to_01_threshold=spec_data_array[None,:,1]#new lhs dim for each threshold
                mse_array=self.threshold_mse(spec_data_array,thresholds)
                mse_spec_dict[spec_name]={str(thresholds[i]):mse_array[i] for i in range(len(thresholds))}
                self.logger.debug(f'for spec_name:{spec_name}, after applying threshold_array with shape:{thresholds.shape}, mse_array has shape:{mse_array.shape}, and mse_array:{mse_array}')
        with open(self.species_mse_array_dict_path,'w') as f:
            json.dump(mse_spec_dict,f)
                
    
    def threshold_mse(self,data,threshold_array):
        newshape=(threshold_array.size,data.shape[0]) # add a dimension for each thrshold on lhs and drop dep var dim on rhs
        ydata=np.broadcast_to(data[None,:,0].astype(np.bool),newshape)
        yhatdata=np.broadcast_to(data[None,:,1],newshape)
        brdcst_threshold_array=np.broadcast_to(threshold_array[:,None],newshape)
        yhatdata_01=np.zeros(newshape,dtype=np.bool)
        yhatdata_01[yhatdata>brdcst_threshold_array]=1
        lossarray=np.not_equal(ydata,yhatdata_01)
        mse_array=np.mean(lossarray,axis=1)
        #mse_array=np.zeros(lossarray.shape, dtype=np.bool)
        #mse_array[lossarray]=lossarray[lossarray].size/ydata.size
        #mse_array=np.mean(np.power((ydata-yhatdata_01),2),axis=1)
        
        return mse_array
        
            
            
            
                