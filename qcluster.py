import pickle
import os,sys,psutil
import re
from time import strftime,sleep
import datetime
import kernelcompare 
import traceback
import shutil
from random import randint,seed,shuffle
import logging
from helpers import Helper
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from queue import Queue
import numpy as np
from sqlitedict import SqliteDict
import sk_tool
import data_gen as dg
from pisces_params import PiSetup,MonteSetup

#class QueueManager(BaseManager): pass
class DBTool:
    def __init__(self):
        resultsdir=os.path.join(os.getcwd(),'results')
        if not os.path.exists(resultsdir):
            os.mkdir(resultsdir)
        self.resultsDBdictpath=os.path.join(resultsdir,'resultsDB.sqlite')
        self.resultsDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='results') # contains sk_tool for each hash_id
        self.genDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='gen')# gen for generate. contains {'model_gen':model_gen,'data_gen':data_gen} for each hash_id
    
    
    def addToDBDict(self,save_list,gen=0):
        if gen:
            db=self.genDBdict
        else:
            db=self.resultsDBdict
        with db() as dbdict:
            try:
                if type(save_list) is dict:
                    save_list=[save_list]
                for result in save_list:
                    for key,val in result.items():
                        dbdict[key]=val
            except:
                self.logger.exception('')
            dbdict.commit()
        return  
        

class TheQManager(mp.Process,BaseManager):
    def __init__(self,address,qdict):
        self.netaddress=address
        self.qdict=qdict
        self.BaseManager=BaseManager
        super(TheQManager,self).__init__()
        
    def run(self):
        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'TheQManager-log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**6, backupCount=20)],
            level=logging.WARNING,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
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
        
class SaveQDumper(mp.Process,DBTool):
    def __init__(self,q):
        self.q=q
        #fself.netaddress=address
        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'SaveQDumper-log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
            level=logging.WARNING,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.logger.info('SaveQDumper starting')
        #self.BaseManager=BaseManager
        #super(SaveQDumper,self).__init__()
        super().__init__()
        #DBTool.__init__(self)
    
    
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
                    model_save_list=queue.get(True,5)
                    
                    success=1
                except:
                    if queue.empty():
                        self.logger.debug('SaveQDumper saveq is empty, returning')
                        return
                    else:
                        self.logger.exception('SaveQDumper unexpected error!')
                if success:
                    if type(model_save_list) is str:
                        if model_save_list=='shutdown':
                            self.logger.DEBUG(f'SaveQDumper shutting down')
                            return
                    self.addtoDBdict(model_save_list)
            except:
                self.logger.exception('unexpected error in SaveQDumper while outer try')
            
            
            
class JobQFiller(mp.Process):
    '''
    runmaster calls this and passes the full list_of_rundicts to it
    '''
    def __init__(self,q,joblist):
        self.q=q
        self.joblist=joblist
        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'JobQFiller-log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
            level=logging.WARNING,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.logger.info('JobQFiller starting')
        super().__init__()
    
    
    def run(self):
        #QueueManager.register('jobq')
        #m = QueueManager(address=self.netaddress, authkey=b'qkey')
        #m.connect()
        #queue = m.jobq()
        queue=self.q
        '''jobcount=len(self.joblist)
        a_rundict=self.joblist[-1]
        a_savepath=a_rundict['savepath']
        savedir=os.path.split(a_savepath)[0]
        savedset=set(os.listdir(savedir))
        print(f'len(savedset):{len(savedset)}')
        shuffle(self.joblist)'''
        i=1
        while len(self.joblist):
            job=self.joblist.pop()
            try:
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

                

class RunNode(mp.Process,BaseManager):
    def __init__(self,local_run=None,source=None,qdict=None):
        try:
            self.logger=logging.getLogger(__name__)
            self.logger.info('starting RunNode object')
        except:
            logdir=os.path.join(os.getcwd(),'log')
            if not os.path.exists(logdir): os.mkdir(logdir)
            handlername=os.path.join(logdir,f'RunNode-log')
            logging.basicConfig(
                handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
                level=logging.WARNING,
                format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%Y-%m-%dT%H:%M:%S')
            self.logger = logging.getLogger(handlername)
            self.logger.info('RunNode logging')
        
        self.qdict=qdict
        self.source=source
        if not local_run:
            self.netaddress=('192.168.1.45',50002)
            self.BaseManager=BaseManager
        super().__init__()
    
    
    
        
    
    def build_from_rundict(self,rundict):
        data_gen=rundict['data_gen'] #how to generate the data
        data=dg.data_gen(data_gen)
        model_gen_dict=rundict['model_gen_dict']
        hash_id_model_dict={}
        for model_gen in model_gen_dict:
            hash_id_model_dict[model_gen['hash_id']]=sk_tool(model_gen)
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
                    rundict=jobq.get()
                    self.logger.debug(f'RunNode has job, rundict: {rundict}')
                    havejob=1
                except:
                    self.logger.exception('')
                if havejob:
                    if type(rundict) is str:
                        if rundict=='shutdown':
                            return
                    data,hash_id_model_dict=self.build_from_rundict(rundict) # each estimator contains rundict
                    for hash_id,model in hash_id_model_dict.items():
                        try:
                            model.fit(data.X_train,data.y_train)
                        except:
                            self.logger.exception('error for rundict:{rundict}')
                        savetup=(hash_id,model)
                        qtry=0
                        while True:
                            self.logger.debug(f'adding model_save_list to saveq')
                            try:
                                qtry+=1
                                saveq.put(savetup)
                                self.logger.debug(f'model_save_list sucesfully added to saveq')
                                break
                            except:
                                if not saveq.full() and qtry>3:
                                    self.logger.exception('error adding to saveq')
                                else:
                                    sleep(1)
                                    
                        
            except:
                self.logger.exception('')           
    
    
class RunCluster(mp.Process,kernelcompare.KernelCompare,DBTool):
    '''
    '''
    
    def __init__(self,source=None,local_run=None,nodecount=0,qdict=None):
        try:
            self.logger=logging.getLogger(__name__)
            self.logger.info('starting RunCluster object')
        except:
            logdir=os.path.join(os.getcwd(),'log')
            if not os.path.exists(logdir): os.mkdir(logdir)
            handlername=os.path.join(logdir,f'mycluster_.log')
            logging.basicConfig(
                handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
                level=logging.WARNING,
                format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%Y-%m-%dT%H:%M:%S')
            self.logger = logging.getLogger(handlername)
        
        self.qdict=qdict 
        if local_run:
            assert type(qdict) is dict,'qdict expected to be dict b/c local_run is true'
        else:
            self.netaddress=('192.168.1.45',50002)
            qm=TheQManager(self.netaddress,None)
            qm.start()
            sleep(1)
        if nodecount:
            #self.nodelist=[RunNode(source=source,local_run=local_run,qdict=self.qdict) for _ in range(nodecount)]
            self.nodelist=[RunNode(source=source,local_run=local_run,qdict=qdict) for _ in range(nodecount)]
            [node.start() for node in self.nodelist]
        if source is None:
            self.source='pisces'
        else:
            self.source=source
        if self.source=='pisces':
            self.setup=PiSetup()
        if self.source='monte':
            self.setup=MonteSetup()
            
        seed(1)  
        self.nodecount=nodecount
        
        super(RunCluster,self).__init__()
        
        
    def run(self,):
        self.logger.debug('master starting up')
        try:
            model_setup=self.setup.model_setup
            data_setup=self.setup.data_setup
            list_of_run_dicts=self.setupRundictList(model_setup,data_setup)
        
            self.logger.debug(f'len(list_of_run_dicts):{len(list_of_run_dicts)}')
            jobqfiller=JobQFiller(self.qdict['jobq'],list_of_run_dicts)
            jobqfiller.run()
            saveqdumper=SaveQDumper(self.qdict['saveq'])
            
            self.logger.debug('back from jobqfiller.run()')
            check_complete=0
            while not check_complete:
                sleep(5)
                saveqdumper.run()
                check_complete=self.checkComplete():
            return
        except:
            self.logger.exception('')
        
    def setupRundictList(self,model_setup,data_setup):
        #a run dict has a 
        #model_gen_dict of models per instance of data_gen
        #each of those model-data combos has a hash_id built from
        #the model_dict and data_gen
        run_dict_list=[]
        run_record_dict={}
        data_gen_list=data_setup()
        
        for data_gen in data_gen_list
            model_gen_dict={}
            model_gen_list=model_setup()
            for model_gen in model_gen_list:
                run_record={'model_gen':model_gen,'data_gen':data_gen}
                hash_id=joblib.hash(run_record)
                run_record_dict[hash_id]=run_record
                model_gen_dict[hash_id]=model_gen
            run_dict={'data_gen':data_gen, 'model_gen_dict':model_gen_dict}
            run_dict_list.append(run_dict)
        self.addToDBDict(run_record_dict,gen=1)
        return run_dict_list
        
        
    def checkComplete(self,run_dict_list=None):
        resultsDBdict=self.resultsDBdict()
        if not run_dict_list:
            hash_id_iter=self.genDBdict().keys()
        else:
            hash_id_iter={hash_id:r for r,rundict in enumerate(run_dict_list) for hash_id in rundict['model_gen_dict'].keys()}
        for hash_id in hash_id_iter:
            if hash_id in resultsDBdict:
                if run_dict_list:
                    r=hash_id_iter[hash_id]
                    run_dict_list[r].pop(hash_id)
            else:
                if not run_dict_list:
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
