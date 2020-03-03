from pisces_data_huc12 import PiscesDataTool
import os
import logging

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
        
        PiscesDataTool.__init__(self)
        
        
    def preprocess_xgboost_csv(self,filename):
        boostdatadict=self.getcsvfile(filename)
        print(type(boostdatadict))
        print([item for item in boostdatadict][0:10])
        try:self.specieslist
        except:
            self.returnspecieslist()
            
        spec_idx_dict={spec_name:[] for spec_name in self.specieslist}
        spec_row_dict=spec_idx_dict.copy() 
        mismatchlist=[]
        for idx,row in enumerate(boostdatadict):
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
        xgboost_data_path=os.path.join(self.savedir,'xgboost_data.pickle')
        self.spec_row_dict=spec_row_dict
        self.spec_idx_dict=spec_idx_dict
        
        with open(xgboost_data_path,'wb') as f:
            pickle.dump((spec_row_dict,spec_idx_dict,mismatchlist),f)
                                
                                     
    def process_xgboost():
        try: spec_row_dict=self.spec_row_dict
        except:
            xgboost_data_path=os.path.join(self.savedir,'xgboost_data.pickle')
            with open(xgboost_data_path,'rb') as f:
                spec_row_dict,spec_idx_dict,mismatchlist=pickle.load(f)
        for spec_name in spec_row_dict:
            if spec_row_dict[spec_name]: # empty list will evaluate as False
                np.array()
