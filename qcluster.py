import pickle
import os,sys,psutil
import re
from time import strftime,sleep
import datetime
import traceback
import shutil
from random import randint,seed,shuffle
from helpers import Helper
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from queue import Queue
import numpy as np
from sqlitedict import SqliteDict
from  sk_tool import SKToolInitializer
from datagen import dataGenerator 
from pisces_params import PiSetup,MonteSetup
from pi_db_tool import DBTool
from mylogger import myLogger
#class QueueManager(BaseManager): pass

        

class TheQManager(mp.Process,BaseManager,myLogger):
    def __init__(self,address,qdict):
        self.netaddress=address
        self.qdict=qdict
        self.BaseManager=BaseManager
        super(TheQManager,self).__init__()
        
    def run(self):
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        #for qname in self.qdict:
        #    q=self.qdict[qname]
        #    self.BaseManager.register(qname, callable=lambda:q)
        if self.qdict is None:
            qdict={'jobq':mp.Queue(),'saveq':mp.Queue()}
        else:
            qdict=self.qdict
        jobq = qdict['jobq']
        saveq = qdict['saveq']
        self.BaseManager.register('jobq', callable=lambda:jobq)
        self.BaseManager.register('saveq', callable=lambda:saveq)
        m = self.BaseManager(address=self.netaddress, authkey=b'qkey')
        s = m.get_server()
        self.logger.info('TheQManager starting')
        s.serve_forever()
        
class SaveQDumper(mp.Process,DBTool,myLogger):
    def __init__(self,q):
        self.q=q
        #fself.netaddress=address
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        #self.BaseManager=BaseManager
        #super(SaveQDumper,self).__init__()
        super().__init__()
        DBTool.__init__(self)
    
    
    def run(self):
        #self.BaseManager.register('saveq')
        #m = self.BaseManager(address=self.netaddress, authkey=b'qkey')
        #m.connect()
        #queue = m.saveq()
        queue=self.q
        keepgoing=1
        
        while keepgoing:
            try:
                success=0
                try:
                    savedict=queue.get(True,5)
                    
                    success=1
                except:
                    if queue.empty():
                        self.logger.debug('SaveQDumper saveq is empty, returning')
                        return
                    else:
                        self.logger.exception('SaveQDumper unexpected error!')
                if success:
                    if type(savedict) is str:
                        if savedict=='shutdown':
                            self.logger.DEBUG(f'SaveQDumper shutting down')
                            return
                    assert type(savedict) is dict, f'SaveQDumper expecting a dict for savedict but got {type(savedict)}'
                    for hash_id,model_dict in savedict.items():
                        try:
                            species=model_dict['data_gen']['species']
                        except:
                            self.logger.info("failed trying to get species from savedict")
                            species='error'
                        try:
                            model_name=model_dict['model_gen']['name']
                        except:
                            self.logger.exception('failed try to get model name from model_dict')
                            model_name='error'
                        self.logger.info(f'saveqdumper is adding to DB dict species:{species}, model_name:{model_name}')
                    save_list=[savedict] # b/c addToDBDict expects a list of dicts.
                    self.addToDBDict(save_list)
            except:
                self.logger.exception('unexpected error in SaveQDumper while outer try')
            
            
            
class JobQFiller(mp.Process,myLogger):
    '''
    runmaster calls this and passes the full list_of_rundicts to it
    '''
    def __init__(self,q,joblist):
        self.q=q
        self.joblist=joblist
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        super().__init__()
    
    
    def run(self):
        #QueueManager.register('jobq')
        #m = QueueManager(address=self.netaddress, authkey=b'qkey')
        #m.connect()
        #queue = m.jobq()
        queue=self.q
        i=1
        while len(self.joblist):
            job=self.joblist.pop()
            try:
                jobcount=len(self.joblist)
                self.logger.debug(f'adding job:{i}/{jobcount} to job queue')
                queue.put(job)
                self.logger.debug(f'job:{i}/{jobcount} succesfully added to queue')
                i+=1
            except:
                self.joblist.append(job)
                if queue.full():
                    self.logger.DEBUG('jobq full, waiting 4s')
                    sleep(4)
                else:
                    self.logger.exception(f'jobq error for i:{i}')
        self.logger.debug('all jobs added to jobq.')
        return

                

class RunNode(mp.Process,BaseManager,myLogger):
    def __init__(self,local_run=None,source=None,qdict=None):
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        self.qdict=qdict
        self.source=source
        if not local_run:
            self.netaddress=('192.168.1.83',50002)
            self.BaseManager=BaseManager
        super().__init__()
    
    
    
        
    
    def build_from_rundict(self,rundict):
        data_gen=rundict['data_gen'] #how to generate the data
        data=dataGenerator(data_gen)
        model_gen_dict=rundict['model_gen_dict'] # {hash_id:data_gen...}
        hash_id_model_dict={}
        for hash_id,model_gen in model_gen_dict.items():
            model_dict={'model':SKToolInitializer(model_gen),'data_gen':data_gen,'model_gen':model_gen}
            hash_id_model_dict[hash_id]=model_dict# hashid based on model_gen and data_gen
        return data,hash_id_model_dict
    
    def run(self,):
        self.logger.info('RunNode running')
        platform=sys.platform
        p=psutil.Process(os.getpid())
        if platform=='win32':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(6)
        if self.qdict:
            jobq=self.qdict['jobq']
            saveq=self.qdict['saveq']
        else:
            self.BaseManager.register('jobq')
            self.BaseManager.register('saveq')
            m = self.BaseManager(address=self.netaddress, authkey=b'qkey')
            m.connect()
            jobq = m.jobq()
            saveq = m.saveq()
        #kc=kernelcompare.KernelCompare(source=self.source) # a new one every run
        while True:
            try:
                havejob=0
                jobsuccess=0
                try:
                    self.logger.debug('RunNode about to check jobq')
                    rundict=jobq.get(True,10)
                    self.logger.debug(f'RunNode has job, rundict: {rundict}')
                    havejob=1
                except:
                    self.logger.exception('')
                if havejob:
                    if type(rundict) is str:
                        if rundict=='shutdown':
                            jobq.put(rundict)
                            return
                    data,hash_id_model_dict=self.build_from_rundict(rundict) # each estimator contains rundict
                    for hash_id,model_dict in hash_id_model_dict.items():
                        try:
                            success=0
                            model_dict['model'].run(data)
                            success=1
                        except:
                            self.logger.exception('error for model_dict:{model_dict}')
                        savedict={hash_id:model_dict}
                        qtry=0
                        while success:
                            self.logger.debug(f'adding savedict to saveq')
                            try:
                                qtry+=1
                                saveq.put(savetup)
                                self.logger.debug(f'savedict sucesfully added to saveq')
                                break
                            except:
                                if not saveq.full() and qtry>3:
                                    self.logger.exception('error adding to saveq')
                                else:
                                    sleep(1)
                                    
                        
            except:
                self.logger.exception('')           
    
    
class RunCluster(mp.Process,DBTool,myLogger):
    '''
    '''
    
    def __init__(self,source=None,local_run=None,nodecount=0,qdict=None):
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        
        
        if local_run:
            assert type(qdict) is dict,'qdict expected to be dict b/c local_run is true'
        else:
            self.netaddress=('192.168.1.83',50002)
            qm=TheQManager(self.netaddress,None)
            qm.start()
            sleep(1)
        self.qdict=qdict 
        if self.qdict is None:
            self.qdict=self.getqdict()
        if nodecount:
            #self.nodelist=[RunNode(source=source,local_run=local_run,qdict=self.qdict) for _ in range(nodecount)]
            self.nodelist=[RunNode(source=source,local_run=local_run,qdict=self.qdict) for _ in range(nodecount)]
            [node.start() for node in self.nodelist]
        if source is None:
            self.source='pisces'
        else:
            self.source=source
        if self.source=='pisces':
            self.setup=PiSetup()
        if self.source=='monte':
            self.setup=MonteSetup()
            
        seed(1)  
        self.nodecount=nodecount
        
        super().__init__()
        DBTool.__init__(self)
        
        
        
    def run(self,):
        self.logger.debug('master starting up')
        try:
            
            list_of_run_dicts,run_record_dict=self.setup.setupRundictList()
            self.addToDBDict(run_record_dict,gen=1) # create a record of the rundicts to check when it's all complete.
            list_of_run_dicts=self.checkComplete(run_dict_list=list_of_run_dicts) # remove any that are already in resultsDB
        
            self.logger.debug(f'len(list_of_run_dicts):{len(list_of_run_dicts)}')
            jobqfiller=JobQFiller(self.qdict['jobq'],list_of_run_dicts)
            jobqfiller.run()
            saveqdumper=SaveQDumper(self.qdict['saveq'])
            
            check_complete=0
            while not check_complete:
                sleep(30)
                saveqdumper.run()
                check_complete=self.checkComplete()
            jobqfiller.join() 
            saveqdumper.join()
            [node.join() for node in self.nodelist]
            self.qdict['jobq'].put('shutdown')
            return
        except:
            self.logger.exception('')
        
 
        
        
    def checkComplete(self,run_dict_list=None):
        #run_dict_list is provided at startup, and if not, useful for checking if all have been saved
        resultsDBdict=self.resultsDBdict()
        complete_hash_id_list=list(resultsDBdict.keys())
        if run_dict_list:
            for r,run_dict in enumerate(run_dict_list):
                model_gen_dict=run_dict['model_gen_dict']
                for hash_id in list(model_gen_dict.keys()): #so model_gen_dict can be changed
                    if hash_id in complete_hash_id_list:
                        del model_gen_dict[hash_id]
                        self.logger.info(f'checkComplete already completed hash_id:{hash_id}')
            return run_dict_list
        else:
            gen_hash_id_list=self.genDBdict().keys()
            for hash_id in gen_hash_id_list:
                if not hash_id in complete_hash_id_list:
                    return False
        return True # must be done
                        
                        
        hash_id_list=list(self.genDBdict().keys())
        
        '''
        if not run_dict_list:
            hash_id_iter=self.genDBdict().keys()
        else:
            hash_id_iter={hash_id:r for r,rundict in enumerate(run_dict_list) for hash_id in rundict['model_gen_dict'].keys()}
        for hash_id in hash_id_iter:
            if hash_id in resultsDBdict:
                if run_dict_list: # only retu
                    r=hash_id_iter[hash_id]
                    run_dict_list[r].pop(hash_id)
        '''            
        if return_list:
            return run_dict_list
        else:
            if run_dict_list:
                return True
            else:
                return False
                    
            
    def getqdict(self):
        BaseManager.register('jobq')
        BaseManager.register('saveq')
        m = BaseManager(address=self.netaddress, authkey=b'qkey')
        m.connect()
        jobq = m.jobq()
        saveq = m.saveq()
        return {'saveq':saveq,'jobq':jobq}    
            
                

if __name__=="__main__":
    RunNode(local_run=0)
