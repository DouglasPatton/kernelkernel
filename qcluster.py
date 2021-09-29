
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
from pi_runners import FitRunner
from pi_db_tool import DBTool
from mylogger import myLogger
#class QueueManager(BaseManager): pass
import json
        

class TheQManager(mp.Process,BaseManager,myLogger):
    def __init__(self,address,qdict):
        self.netaddress=address
        self.qdict=qdict
        self.BaseManager=BaseManager
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        super(TheQManager,self).__init__()
        
    def run(self):
        self.logger.info(f'qmanager pid: {os.getpid()}')
        #func_name=f'{sys._getframe().f_code.co_name}'
        #myLogger.__init__(self,name=f'{func_name}.log')
        #self.logger.info(f'starting {func_name} logger')
        #for qname in self.qdict:
        #    q=self.qdict[qname]
        #    self.BaseManager.register(qname, callable=lambda:q)
        if self.qdict is None:
            qdict={'jobq':mp.Queue(),'saveq':mp.Queue()}
        else:
            qdict=self.qdict
        jobq = qdict['jobq']
        saveq = qdict['saveq']
        #pipeq=qdict['pipeq']
        self.BaseManager.register('jobq', callable=lambda:jobq)
        self.BaseManager.register('saveq', callable=lambda:saveq)
        m = self.BaseManager(address=self.netaddress, authkey=b'qkey')
        s = m.get_server()
        self.logger.info('TheQManager starting')
        s.serve_forever()
        
class SaveQDumper(mp.Process,myLogger): #DBTool removed
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
        #DBTool.__init__(self)
    
    
    def run(self):
        self.logger.info(f'saveqdumper pid: {os.getpid()}')
        #self.BaseManager.register('saveq')
        #m = self.BaseManager(address=self.netaddress, authkey=b'qkey')
        #m.connect()
        #queue = m.saveq()
        queue=self.q
        keepgoing=1
        self.logger.info('saveqdumper is running') 
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
                    if 'fitfail' in savedict:
                        savedict.pop('fitfail')
                        fitfail=True
                    else:
                        fitfail=False
                    '''for hash_id,thing_to_save in savedict.items():
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
                    '''    #else:
                    #self.logger.info(f'saveqdumper has data with type:{type(thing_to_save)} and self.db_kwargs:{self.db_kwargs}')
                    
                    save_list=[savedict] # b/c addToDBDict expects a list of dicts.
                    s=0
                    while True:
                        s+=1
                        try:
                            if fitfail:
                                DBTool().addToDBDict(save_list,db=self.fitfailDBdict)
                            else:
                                #self.logger.info(f'saveqdumper db_kwargs:{db_kwargs}')
                                DBTool().addToDBDict(save_list,**self.db_kwargs)
                            break
                        except:
                            self.logger.exception(f'error adding to DB. try:{s}')
                            sleep(5)
            except:
                self.logger.exception('unexpected error in SaveQDumper while outer try')
                sleep()
            
            
            
class JobQFiller(mp.Process,myLogger):
    '''
    runmaster calls this and passes the full list_of_rundicts to it
    '''
    def __init__(self,q,joblist,do_mp=True,address=None):
        self.q=q
        self.netaddress=address
        self.joblist=joblist
        self.do_mp=do_mp
        if self.do_mp:
            super().__init__()
        #func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name='jobqfiller.log')
        self.logger.info(f'starting jobqfiller logger')
        
    def addjobs(self,morejobs):
        assert not self.do_mp,f'only use addjobs if do_mp is False'
        if not type(morejobs) is list:
            morejobs=[morejobs]
        self.joblist.extend(morejobs)
        
    
    def run(self):
        self.logger.info(f'jobqfiller pid: {os.getpid()}')
        #QueueManager.register('jobq')
        #m = QueueManager(address=self.netaddress, authkey=b'qkey')
        #m.connect()
        #queue = m.jobq()
        queue=self.q
        i=1
        max_q_size=1 #not really the max
        if not self.netaddress is None:
            self.BM=BaseManager(address=self.netaddress,authkey=b'qkey')
            self.M=self.BM.connect()
        tries=0;p_i=0#for naming pipes
        while len(self.joblist):
            if queue.empty():
                #if i>2 and q_size==0 and q_size<len(self.joblist): 
                #    self.logger.info(f'jobq is empty, so max_q_size doubling from {max_q_size}')
                #    max_q_size*=2 # double max q since it is being consumed
                tries=0
                for i in range(max_q_size): #fill queue back up to 2*max_q_size
                    if len(self.joblist):
                        #n_sel=np.random.randint(0,len(self.joblist))
                        job=self.joblist.pop()
                    else: 
                        self.logger.critical("jobqfiller's joblist is empty, returning")
                        return
                    try:
                        finished=job.build()
                    except:
                        self.logger.exception(f'error building job')
                        finished=False
                    try:
                        if finished:
                            self.logger.info(f'skipping finished job ({i})')
                            i+=1
                        else:
                            jobcount=len(self.joblist)
                            self.logger.debug(f'adding job:{i}/{jobcount} to job queue')
                            p_rcv,p_snd=mp.Pipe(False)#False for one way, rcv-recieve,snd-send
                            #self.M.register(f'p_snd_{p_i}', callable=lambda:p_snd)
                            self.M.register(f'p_rcv_{p_i}', callable=lambda:p_rcv)
                            p_i+=1
                            p_snd.send(job)
                            queue.put(f'p_rcv_{p_i}')
                            self.logger.debug(f'job:{i}/{jobcount} succesfully added to queue')
                            i+=1
                    except:
                        self.joblist.append(job)
                        self.logger.exception(f'jobq error for i:{i}, adding back to joblist, which has len: {len(self.joblist)}')
            else:
                tries+=1
                sleep(2)
                if tries>10: sleep(5)
            
        self.logger.debug('all jobs added to jobq.')
        return

                

class RunNode(mp.Process,BaseManager,myLogger):
    def __init__(self,local_run=None,source=None,qdict=None,run_type='fit',cv_n_jobs=None):
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        self.qdict=qdict
        self.source=source
        self.run_type=run_type
        self.cv_n_jobs=cv_n_jobs
        if not local_run:
            try:
                with open('ip.json','r') as f:
                    ipdict=json.load(f)
                self.netaddress=(ipdict['ip'],ipdict[f'port_{run_type}'])
            except:
                self.logger.exception(f'ip address error')
                assert False, 'Halt'
                self.netaddress=('10.0.0.3',50002)
            self.BaseManager=BaseManager
        super().__init__()
    
    
    def run(self,):
        self.logger.info(f'runnode pid: {os.getpid()}')
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
                tries=0
                try:
                    pipe_str=jobq.get(True,20)
                    rcv_pipe=getattr(m,pipe_str)
                    runner=rcv_pipe.recv()
                    rcv_pipe.close()
                    #runner=jobq.get(True,20)
                    #self.logger.debug('RunNode about to check jobq')
                    #pipe=jobq.get(True,20)
                    #pipe.send('ready')
                    #runner=pipe.recv()
                    self.logger.debug(f'RunNode has job') #:runner.rundict:{runner.rundict}')
                    havejob=1
                    tries=0
                except:
                    tries+=1
                    if tries>10:
                        self.logger.exception('tried 10 times')
                        tries=0
                if havejob:
                    if type(runner) is str:
                        if runner=='shutdown':
                            jobq.put(runner)
                            return
                    # each estimator contains rundict
                    #runner.passPipe(saveq)
                    runner.passQ(saveq)
                    if type(runner) is FitRunner:
                        runner.run(cv_n_jobs=self.cv_n_jobs)
                    else:
                        runner.run()
            except:
                self.logger.exception('runnode outer catch!')           
    
    
class RunCluster(mp.Process,DBTool,myLogger):
    '''
    '''
    
    def __init__(self,source=None,local_run=None,nodecount=0,qdict=None,run_type='fit',cv_run=True):
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        
        
        if local_run:
            assert type(qdict) is dict,'qdict expected to be dict b/c local_run is true'
            self.netaddress=None
        else:
            try:
                with open('ip.json','r') as f:
                    ipdict=json.load(f)
                self.netaddress=(ipdict['ip'],ipdict[f'port_{run_type}'])
            except:
                self.logger.exception(f'ip address error')
                assert False, 'Halt'
                self.netaddress=('10.0.0.3',50002)
            self.BaseManager=BaseManager
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
        else:self.nodelist=[]
        if source is None:
            self.source='pisces'
        else:
            self.source=source
        if self.source=='pisces':
            self.setup=PiSetup(run_type=run_type,cv_run=cv_run) # in pisces_params, this file/object determines how things run
        if self.source=='monte':
            self.setup=MonteSetup()
            
        seed(1)  
        self.nodecount=nodecount
        
        super().__init__()
        DBTool.__init__(self)
        
        
        
    def run(self,):
        self.logger.info(f'runcluster pid: {os.getpid()}')
        self.logger.debug('master starting up')
        try:
            runlist,hash_id_list=self.setup.setupRunners()
            order=list(range(len(runlist)))
            shuffle(order)
            runlist=[runlist[i] for i in order]
            if len(runlist)==0:
                self.logger.critical(f'runlist is empty from the start')
                return
            hash_id_list=[hash_id_list[i] for i in order]
            run_jobfiller_as_proc=True
            if run_jobfiller_as_proc:
                jobqfiller=JobQFiller(self.qdict['jobq'],runlist,do_mp=True,address=self.netaddress)
                jobqfiller.start()
            else:
                jobs_at_a_time=40 if 40<len(runlist) else len(runlist)
                jobqfiller=JobQFiller(self.qdict['jobq'],[runlist.pop() for _ in range(jobs_at_a_time)],do_mp=False,address=self.netaddress)
                jobqfiller.run()
            self.logger.info(f'back from jobqfiller, initializing saveqdumper')
            saveqdumper=SaveQDumper(self.qdict['saveq'],db_kwargs=self.setup.db_kwargs)
            check_complete=0
            while not check_complete:
                if not run_jobfiller_as_proc:
                    while len(runlist)>0:
                        if jobs_at_a_time>len(runlist):jobs_at_a_time=len(runlist)
                        if jobs_at_a_time>0:
                            jobqfiller.addjobs([runlist.pop() for _ in range(jobs_at_a_time)])
                            jobqfiller.run()
                        saveqdumper.run()#
                sleep(40)
                saveqdumper.run()
                #check_complete=self.setup.checkComplete(db=self.setup.db_kwargs,hash_id_list=hash_id_list)
            try:jobqfiller.join()
            except: self.logger.exception(f'jobqfiller join error, moving on.')
            #jobqfiller.joblist=['shutdown']
            #jobqfiller.run()
            
            #saveqdumper.join()
            
            self.qdict['jobq'].put('shutdown')
            [node.join() for node in self.nodelist]
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
