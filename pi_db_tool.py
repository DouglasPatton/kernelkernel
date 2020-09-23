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
        self.resultsDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='results') # contains sk_tool for each hash_id
        self.genDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='gen')# gen for generate. contains {'model_gen':model_gen,'data_gen':data_gen} for each hash_id
        self.postfitDBdict=lambda name:SqliteDict(filename=self.postfitDBdictpath,tablename=name)
    
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
                    for key,val in save_list.items():
                        dbdict[key]=val
                if type(save_list) is list:
                    if type(save_list[0]) is tuple:
                        for key,val in save_list:
                            dbdict[key]=val
                    else:
                        assert False, f'expecting tuple for save_list first item, but type(save_list[0]):{type(save_list[0])}'
            except:
                self.logger.exception('')
            dbdict.commit()
        return  