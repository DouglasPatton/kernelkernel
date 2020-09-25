from sqlitedict import SqliteDict
import sys,os
from mylogger import myLogger


class DBTool(myLogger,):
    def __init__(self):
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        resultsdir=os.path.join(os.getcwd(),'results')
        if not os.path.exists(resultsdir):
            os.mkdir(resultsdir)
        self.resultsDBdictpath=os.path.join(resultsdir,'resultsDB.sqlite')
        self.postfitDBdictpath=os.path.join(resultsdir,'postfitDB.sqlite')
        #self.resultsDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='results') # contains sk_tool for each hash_id
        #self.genDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='gen')# gen for generate. contains {'model_gen':model_gen,'data_gen':data_gen} for each hash_id
        #self.postfitDBdict=lambda name:SqliteDict(filename=self.postfitDBdictpath,tablename=name)
    
    def resultsDBdict(self):
        return SqliteDict(filename=self.resultsDBdictpath,tablename='results')
    
    def genDBdict(self):
        return SqliteDict(filename=self.resultsDBdictpath,tablename='gen')
    
    def postfitDBdict(self,name):
        return SqliteDict(filename=self.postfitDBdictpath,tablename=name)
    
    def addToDBDict(self,save_list,gen=0,post_fit_tablename=0):
        if gen:
            db=self.genDBdict
        elif post_fit_tablename:
            db=self.postfitDBdict(post_fit_tablename)
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
            except:
                self.logger.exception('dbtool addtoDBDict error! gen:{gen}')
            dbdict.commit()
        return  
    
    def purgeExtraGen(self):
        rdb=self.resultsDBdict()
        with self.genDBdict() as dbdict:
            for hash_id,run_record_dict in dbdict.items():
                if not hash_id in rdb:
                    self.logger.info(f'purging {hash_id} run_record_dict:{run_record_dict}')
                    del dbdict[hash_id]
            dbdict.commit()