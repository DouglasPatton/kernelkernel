from sqlitedict import SqliteDict
import sys,os
#from mylogger import myLogger
import logging


class DBTool:
    def __init__(self):
        func_name='DBTool'
        #super.__init__(name=f'{func_name}.log')
        #myLogger.__init__(self,name=f'{func_name}.log')
        self.logger=logging.getLogger()
        self.logger.info(f'starting {func_name} logger')
        resultsdir=os.path.join(os.getcwd(),'results')
        self.resultsdir=resultsdir
        if not os.path.exists(resultsdir):
            os.mkdir(resultsdir)
        self.resultsDBdictpath=os.path.join(resultsdir,'resultsDB.sqlite')
        self.metadataDBdictpath=os.path.join(resultsdir,'metadataDB.sqlite')
        self.genDBdictpath=os.path.join(resultsdir,'genDB.sqlite')
        self.predictDBdictpath=os.path.join(resultsdir,'predictDB.sqlite')
        self.pidataDBdictpath=os.path.join(os.getcwd(),'data_tool','pidataDB.sqlite')
        self.postfitDBdictpath=os.path.join(resultsdir,'postfitDB.sqlite')
        #self.resultsDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='results') # contains sk_tool for each hash_id
        #self.genDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='gen')# gen for generate. contains {'model_gen':model_gen,'data_gen':data_gen} for each hash_id
        #self.predictDBdict=lambda name:SqliteDict(filename=self.predictDBdictpath,tablename=name)
    
    def postFitDBdict(self,name):
        return SqliteDict(filename=self.postfitDBdictpath,tablename=name)
    
    
    def resultsDBdict(self):
        return SqliteDict(filename=self.resultsDBdictpath,tablename='results')
    
    def metadataDBdict(self):
        return SqliteDict(filename=self.metadataDBdictpath,tablename='metadata')
    
    def resultsDBdict_backup(self):
        return SqliteDict(filename='results/resultsDB_backup.sqlite',tablename='results')
    
    def pidataDBdict(self,name='species01'):
        return SqliteDict(filename=self.pidataDBdictpath,tablename=name)
    
    def genDBdict(self):
        return SqliteDict(filename=self.genDBdictpath,tablename='gen')
    
    def predictDBdict(self,):
        return SqliteDict(filename=self.predictDBdictpath,tablename='predict01')
    
    def addToDBDict(self,save_list,db=None,gen=0,predict=0,pi_data=0):
        try:
            
            if db:
                pass        
            elif gen:
                db=self.genDBdict
            elif predict:
                db=self.predictDBdict
            elif pi_data:
                if type(pi_data) is str:
                    kwargs={'name':pi_data}
                else:
                    kwargs={}
                db=lambda: self.pidataDBdict(**kwargs)
            else:
                db=self.resultsDBdict
            
            with db() as dbdict:
                try:
                    if type(save_list) is dict:
                        save_list=[save_list]
                    for dict_i in save_list:
                        for key,val in dict_i.items():
                            if key in dbdict:
                                if not gen:
                                    self.logger.warning(f'overwriting val:{dbdict[key]} for key:{key}')
                                    dbdict[key]=val
                                else:
                                    self.logger.debug(f'key:{key} already exists in gen table in db dict')
                            else:
                                dbdict[key]=val
                except:
                    self.logger.exception('dbtool addtoDBDict error! gen:{gen}')
                dbdict.commit()
            return  
        except:
            self.logger.exception(f'addToDBDict outer catch')
    
    def purgeExtraGen(self):
        rdb=self.resultsDBdict()
        with self.genDBdict() as dbdict:
            for hash_id,run_record_dict in dbdict.items():
                if not hash_id in rdb:
                    self.logger.info(f'purging {hash_id} run_record_dict:{run_record_dict}')
                    del dbdict[hash_id]
            dbdict.commit()
            
    def add_x_vars_to_results(self):
        from datagen import dataGenerator

        rdb=self.resultsDBdict
        spec_xvar_dict={} # to store the xvars the first time a species comes up
        with rdb() as rdb_dict:
            for hash_id in list(rdb_dict.keys()):# generator to list so rdb_dict can be modified.
                modeldict=rdb_dict[hash_id]
                data_gen=modeldict["data_gen"]
                species=data_gen["species"]
                est_name=modeldict["model_gen"]["name"]
                
                if type(modeldict['model']) is dict:
                    try:
                        modeldict['model']['estimator'][0].x_vars
                        self.logger.info(f'{species}:{est_name} already has x_vars')
                    except AttributeError: # add the attribute
                        try:
                            x_vars=spec_xvar_dict[species]
                        except KeyError:
                            self.logger.info(f'key error for {species}:{est_name}, so calling dataGenerator')
                            data=dataGenerator(data_gen)
                            x_vars=data.x_vars
                            spec_xvar_dict[species]=x_vars
                        except:
                            self.logger.exception(f'not a keyerror, unexpected error')
                            assert False,'halt'
                        self.logger.warning(f'adding x_vars for {species}:{est_name}')
                        for m in range(len(modeldict['model']['estimator'])): # cross_validate stores a list of the estimators
                            modeldict['model']['estimator'][m].x_vars=x_vars 
                            #    ^this should be the sktool object that contains the sk_estimator at sktool.model_
                        rdb_dict[hash_id]=modeldict # db only updated if attribute added
                    except:
                        self.logger.exception(f'error for hash_id:{hash_id},modeldict:{modeldict}!')
                        assert False, 'halt'
                    
                else:
                    assert False,'expecting cross_validate dict '

            rdb_dict.commit()
        return
            
            