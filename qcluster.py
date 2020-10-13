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
#from  sk_tool import SKToolInitializer
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
    def __init__(self,q,db_kwargs={}):
        self.q=q
        self.db_kwargs=db_kwargs
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
                            self.logger.debug(f'SaveQDumper shutting down')
                            return
                    assert type(savedict) is dict, f'SaveQDumper expecting a dict for savedict but got {type(savedict)}'
                    for hash_id,thing_to_save in savedict.items():
                        if type(thing_to_save) is dict:
                            try:
                                species=thing_to_save['data_gen']['species']
                            except:

                                self.logger.info("failed trying to get species from savedict")
                                species='error'
                            try:
                                model_name=thing_to_save['model_gen']['name']
                            except:
                                self.logger.exception('failed try to get model name from thing_to_save')
                                model_name='error'
                            self.logger.info(f'saveqdumper is adding to DB dict species:{species}, model_name:{model_name}, hash_id:{hash_id}')
                        else:
                            self.logger.info(f'saveqdumper has data with type:{type(thing_to_save)} and self.db_kwargs:{self.db_kwargs}')
                    save_list=[savedict] # b/c addToDBDict expects a list of dicts.
                    self.addToDBDict(save_list,**self.db_kwargs)
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
        max_q_size=10
        q_size=0;tries=0 # for startup
        while len(self.joblist):
            if q_size<max_q_size:
                tries=0
                for i in range(max_q_size): #fill queue back up to max_q_size
                    job=self.joblist.pop()
                    try:job.build()
                    except:self.logger.exception(f'error building job')
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
            else:
                tries+=1
            q_size=queue.qsize()
            sleep(1+tries*10)
            
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
                    runner=jobq.get(True,10)
                    self.logger.debug(f'RunNode has job') #:runner.rundict:{runner.rundict}')
                    havejob=1
                except:
                    self.logger.exception('')
                if havejob:
                    if type(runner) is str:
                        if runner=='shutdown':
                            jobq.put(runner)
                            return
                     # each estimator contains rundict
                    runner.passQ(saveq)
                    runner.run()
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
            self.setup=PiSetup() # in pisces_params, this file/object determines how things run
        if self.source=='monte':
            self.setup=MonteSetup()
            
        seed(1)  
        self.nodecount=nodecount
        
        super().__init__()
        DBTool.__init__(self)
        
        
        
    def run(self,):
        self.logger.debug('master starting up')
        try:
            runlist,hash_id_list=self.setup.setupRunners()
            jobqfiller=JobQFiller(self.qdict['jobq'],runlist)
            jobqfiller.start()
            jobqfiller.run()
            self.logger.info(f'back from jobqfiller, initializing saveqdumper')
            saveqdumper=SaveQDumper(self.qdict['saveq'],db_kwargs=self.setup.db_kwargs)
            check_complete=0
            while not check_complete:
                sleep(60)
                saveqdumper.run()#
                check_complete=self.setup.checkComplete(db=self.setup.db_kwargs,hash_id_list=hash_id_list)
            #jobqfiller.join() 
            saveqdumper.join()
            [node.join() for node in self.nodelist]
            self.qdict['jobq'].put('shutdown')
            return
        except:
            self.logger.exception('')
            assert False,'unexpected error'
        
        """hash_id_list=list(self.genDBdict().keys())
        
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
                return False"""
                    
            
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
