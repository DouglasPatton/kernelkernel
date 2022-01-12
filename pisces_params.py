#from sk_tool import SkTool
from sk_estimators import sk_estimator
from pisces_data_huc12 import PiscesDataTool
from mylogger import myLogger
import logging
from copy import copy
import joblib
from random import randint
from pi_db_tool import DBTool
from pi_runners import FitRunner,PredictRunner, XPredictRunner
from pi_results import PiResults
#from pi_xresults import PiXResults #no longer using

class PiSetup(myLogger):
    def __init__(self,test=False,run_type='fit',linear_run=False,scorer='f1_micro',cv_run=True):
        self.scorer=scorer
        self.cv_run=cv_run
        self.test=test#True # reduces repeats to speed things up
        self.linear_run=linear_run
        splits=5
        if self.test:
            repeats=2
        else:
            repeats=10
        myLogger.__init__(self,name='PiSetup.log')
        self.logger.info('starting PiSetup logger')
        self.run_type=run_type#'fit'#'predict'# 'fit_fill'#'predict'# 'Xpredict'#,'single_fit'
        if self.run_type=='predict':
            self.db_kwargs=dict(db=DBTool(cv_run=cv_run).predictDBdict)# for saveqdumper addToDBDict and checkcomplete too! #{'predict':True} # for saveQdumper
        elif self.run_type=='Xpredict':
            self.db_kwargs={'Xpredict':True}
        else:
            self.db_kwargs={}
        rs=1
        
        
        """self.permutation_kwargs=dict(
            n_repeats=5,
            random_state=rs,
            #scoring:None
            )"""

        self.model_setup_dict=dict(
            gridpoints=5,
            inner_cv_splits=splits,
            inner_cv_reps=3,
            random_state=rs, # for inner cv and some estimators
            
            )
        if self.test:
            species=(0,20)
        else:
            species='all'
        if self.cv_run:
            cv_gen_dict=dict(n_splits=splits,
                        n_repeats=repeats,
                        strategy=None, # e.g., 'balanced-HUC8'
                        random_state=rs)
            data_split_dict=dict(
                test_share=0,
                cv=cv_gen_dict
            )
        else:data_split_dict=dict(test_share=0)
            
        self.datagen_dict_template=dict(
            min_sample=32,
            min_1count=8, # at least 4 ones per split since split can go down to 2
            shuffle=False,#generally leave this off
            source='Pisces',
            species=species,#'all',#(0,20),#'all', # or a range, i.e., (0,100) # set in data_setup
            data_split=data_split_dict,
            drop_vars=[],
            loc_vars=['HUC12','COMID'],
            )
    
    def model_setup(self,):
        #sk_tool uses model_gen to create the estimator
        model_gen_list=[]
        if self.linear_run:
            ests=['logistic-reg','linear-svc','linear-probability-model']
        else:
            ests=sk_estimator().get_est_dict().keys()
        for est_name in ests:
            
            kwargs=self.model_setup_dict
            model_gen={'kwargs':kwargs,'scorer':self.scorer,'name':est_name}
            model_gen_list.append(model_gen)
        return model_gen_list
            
    
    def data_setup(self,):
        species_list=PiscesDataTool().returnspecieslist()
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
     
    def build_dghash_hash_id_dict_from_run_records(self,run_record_dict):
        datagenhash_hash_id_run_records={}
        self.logger.info(f'building datagenhash_hash_id_run_records')
        for hash_id,run_record in run_record_dict.items(): 
            data_gen=run_record["data_gen"]
            datagenhash=joblib.hash(data_gen)
            try:
                datagenhash_hash_id_run_records[datagenhash][hash_id]=run_record # in case diff species have diff 
                    #     datagen_dicts. if wrong random_state passed to cv, split is wrong
            except KeyError:
                datagenhash_hash_id_run_records[datagenhash]={hash_id:run_record}
            except:
                self.logger.exception(f'not a keyerror, unexpected error')
                assert False,'halt'
        self.logger.info('datagen hash hash_id dict complete')
        return datagenhash_hash_id_run_records
    
    def setupRunners(self,):
        try: self.dbt
        except:self.dbt=DBTool(cv_run=self.cv_run)
        try:self.results_dict
        except: self.results_dict=self.dbt.resultsDBdict()
        try: self.fitfail_dict
        except: self.fitfail_dict=self.dbt.fitfailDBdict()
        #a run dict has a 
        #model_gen_dict of models per instance of data_gen
        #each of those model-data combos has a hash_id built from
        #the model_dict and data_gen
        
        if self.run_type=='fit_fill':
            no_results_run_record_dict=self.dbt.get_no_results_run_record_dict()
            datagenhash_hash_id_run_records=self.build_dghash_hash_id_dict_from_run_records(no_results_run_record_dict)
            self.logger.info(f'datagenhash_hash_id_run_records:{datagenhash_hash_id_run_records}')
            rundict_list=[]#;hash_id_list=[] #latter is for tracking completion (but no longer used)
            for _,hash_id_run_record_dict in datagenhash_hash_id_run_records.items():
                first=True
                for hash_id,run_record in hash_id_run_record_dict.items():
                    #hash_id_list.append(hash_id)
                    if first: 
                        self.logger.info(f'hash_id:{hash_id}')
                        rundict={'data_gen':run_record['data_gen'],
                                 'model_gen_dict':{hash_id:run_record['model_gen']}}
                    else:
                        rundict['model_gen_dict'][hash_id]=run_record['model_gen']
                    first=False
                rundict_list.append(rundict.copy()) #maybe copy is not necessary
            runlist=[]    
            for rundict in rundict_list:
                runlist.append(FitRunner(rundict))
        elif self.run_type in ['fit','single_fit']:
            rundict_list=[]
            run_record_dict={}
            data_gen_list=self.data_setup()
            #hash_id_list=[]
            for data_gen in data_gen_list:
                model_gen_dict={}
                model_gen_list=self.model_setup()
                for model_gen in model_gen_list:
                    run_record={'model_gen':model_gen,'data_gen':data_gen}
                    hash_id=joblib.hash(run_record)
                    if not (hash_id in self.results_dict or hash_id in self.fitfail_dict):
                        #self.logger.info(f'adding to rundict hash_id:{hash_id}')
                        #hash_id_list.append(hash_id)
                        run_record_dict[hash_id]=run_record # store the _gen dicts for reference
                        model_gen_dict[hash_id]=model_gen # 
                    else: self.logger.info(f'setupRunners skipping hash_id:{hash_id}')
                if len(model_gen_dict)>0:
                    rundict={'data_gen':data_gen, 'model_gen_dict':model_gen_dict}
                    rundict_list.append(rundict)

            self.dbt.addToDBDict(run_record_dict,gen=1) # create a record of the rundicts to check when it's all complete.
            self.logger.debug(f'len(rundict_list):{len(rundict_list)}')
            runlist=[]
            #rundict_list=self.checkComplete(rundict_list=rundict_list) # remove any that are already in resultsDB
            for rundict in rundict_list:
                runlist.append(FitRunner(rundict))
        elif self.run_type=='Xpredict':
            if self.test:
                test=20
            else:
                test=self.test
            #PiXResults.consolidateXpredict()
            #rundict_list,hash_id_list=PiResults().build_prediction_rundicts(test=test,XpredictDB=self.dbt.XpredictDBdict())
            rundict_list=PiResults().build_prediction_rundicts(test=test,XpredictDB=self.dbt.XpredictDBdict())
            runlist=[]
            self.logger.info(f'building list of Xpredict runners. len(rundict_list):{len(rundict_list)}')
            for rundict in rundict_list:
                runlist.append(XPredictRunner(rundict))
            self.logger.info(f'list of runners built. len(runlist):{len(runlist)}')
            self.logger.info(f'runlist:{runlist}')
        elif self.run_type=='predict':
            if self.test:
                test=20
            else:
                test=self.test
            #rundict_list,hash_id_list=PiResults().build_prediction_rundicts(test=test)
            rundict_list=PiResults().build_prediction_rundicts(test=test)
            runlist=[]
            self.logger.info(f'building list of runners. len(rundict_list):{len(rundict_list)}')
            for rundict in rundict_list:
                runlist.append(PredictRunner(rundict))
            self.logger.info(f'list of runners built. len(runlist):{len(runlist)}')
            self.logger.info(f'runlist:{runlist}')
        return runlist#,hash_id_list
    
        
    def checkComplete(self,db=None,rundict_list=None,hash_id_list=None):
        #rundict_list is provided at startup, and if not, useful for checking if all have been saved
        if type(db) is dict:
            if 'db' in db:
                db=db['db']
            elif len(db)==0:
                db=None
            
        if db is None:
            db=self.dbt.resultsDBdict()
        if callable(db):
            db=db()
        if type(db) is dict and 'Xpredict' in db:
            complete_hash_id_list=dict.fromkeys(self.dbt.XpredictHashIDComidHashResultsDB()) #tablenames
        else:
            complete_hash_id_list=dict.fromkeys(db.keys())
        rundrop_idx=[]
        if rundict_list:
            for r,run_dict in enumerate(rundict_list):
                model_gen_dict=run_dict['model_gen_dict']
                for hash_id in list(model_gen_dict.keys()): #so model_gen_dict can be changed
                    if hash_id in complete_hash_id_list:
                        del rundict_list[r]['model_gen_dict'][hash_id]
                        self.logger.info(f'checkComplete already completed hash_id:{hash_id}')
                if len(rundict_list[r]['model_gen_dict'])==0:
                    rundrop_idx.append(r)
            for r in rundrop_idx[::-1]: #from the rhs so remaining,lower indices to left don't change
                self.logger.info(f'deleting rundisk idx:{r} bc len=0')
                del rundict_list[r]
            return rundict_list
        else:
            for hash_id in hash_id_list:
                if not hash_id in complete_hash_id_list:
                    return False
        self.logger.warning(f'check complete returning True!!!!!')
        return True # must be done
        
    
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
        
