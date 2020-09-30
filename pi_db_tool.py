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
        self.genDBdictpath=os.path.join(resultsdir,'genDB.sqlite')
        self.postfitDBdictpath=os.path.join(resultsdir,'postfitDB.sqlite')
        self.pidataDBdictpath=os.path.join(os.getcwd(),'data_tool','pidataDB.sqlite')
        #self.resultsDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='results') # contains sk_tool for each hash_id
        #self.genDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='gen')# gen for generate. contains {'model_gen':model_gen,'data_gen':data_gen} for each hash_id
        #self.postfitDBdict=lambda name:SqliteDict(filename=self.postfitDBdictpath,tablename=name)
    
    def resultsDBdict(self):
        return SqliteDict(filename=self.resultsDBdictpath,tablename='results')
    
    def pidataDBdict(self,name='species01'):
        return SqliteDict(filename=self.pidataDBdictpath,tablename=name)
    
    def genDBdict(self):
        return SqliteDict(filename=self.genDBdictpath,tablename='gen')
    
    def postfitDBdict(self,name):
        return SqliteDict(filename=self.postfitDBdictpath,tablename=name)
    
    def addToDBDict(self,save_list,gen=0,post_fit_tablename=0,pi_data=0):
        try:
            if gen:
                db=self.genDBdict
            elif post_fit_tablename:
                db=lambda: self.postfitDBdict(post_fit_tablename)
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