import pickle
import os,sys,psutil
import re
from time import strftime,sleep
import datetime
import kernelcompare 
import traceback
import shutil
from numpy import log
from random import randint,seed,shuffle
import logging
from helpers import Helper
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from queue import Queue

#class QueueManager(BaseManager): pass

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
        
class SaveQDumper(mp.Process):
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
        super(SaveQDumper,self).__init__()
        
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
                    model_save=queue.get_nowait()
                    #self.logger.debug(f'SaveQDumper got: {model_save}')
                    loss=model_save[-1]['loss']
                    lossdict=model_save[-1]['lossdict']
                    naiveloss=model_save[-1]['naiveloss']
                    binary_y_result=model_save[-1]['binary_y_result']
                    message=f"lossdict:{lossdict}, naiveloss:{naiveloss}, binary_y_result:{binary_y_result}"
                    print(message)
                    self.logger.debug(message)
                    success=1
                except:
                    if queue.empty():
                        self.logger.debug('SaveQDumper saveq is empty')
                        break
                    else:
                        self.logger.exception('SaveQDumper unexpected error!')
                if success:
                    if type(model_save) is str:
                        if model_save=='shutdown':
                            self.logger.DEBUG(f'SaveQDumper shutting down')
                            return
                    nodesavepath=model_save[-1]['savepath']
                    '''mastersavepath=os.path.join('master_save',nodesavepath)
                    mastersavedir,stem=os.path.split(mastersavepath)
                    if not os.path.exists(mastersavedir):os.makedirs(mastersavedir)
                    self.logger.debug(f'saveqdumper saving to mastersavepath:{mastersavepath}')'''
                    with open(nodesavepath,'wb') as f:
                        pickle.dump(model_save,f)
            except:
                self.logger.exception('unexpeted error in SaveQDumper while outer try')
            
            
            
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
        super(JobQFiller,self).__init__()
    
    
    
    def run(self):
        #QueueManager.register('jobq')
        #m = QueueManager(address=self.netaddress, authkey=b'qkey')
        #m.connect()
        #queue = m.jobq()
        queue=self.q
        jobcount=len(self.joblist)
        i=1
        while self.joblist:
            job=self.joblist.pop()
            savepath=job['savepath']
            #masterpath=os.path.join('master_save',savepath)
            if not os.path.exists(savepath):
                with open(job['jobpath'],'wb') as f:
                    pickle.dump(job,f)
                try:
                    self.logger.debug(f'adding job:{i}/{jobcount} to job queue')
                    queue.put(job)
                    i+=1
                    self.logger.debug(f'job:{i}/{jobcount} succesfully added to queue')
                except:
                    self.joblist.append(job)
                    if queue.empty():
                        sleep(4)
                    else:
                        self.logger.exception(f'jobq error for i:{i}')
            else:
                self.logger.info(f'JobQFiller is skipping job b/c saved at savepath:{job["savepath"]}')
        self.logger.debug('all jobs added to jobq.')
        return

                

class RunNode(mp.Process,BaseManager):
    def __init__(self,local_run=None,source=None,qdict=None):

        
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
            self.netaddress=('192.168.1.100',50002)
            self.BaseManager=BaseManager
        super(RunNode,self).__init__()
        
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
        kc=kernelcompare.KernelCompare(source=self.source)
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
                    my_optimizedict=rundict['optimizedict']
                    my_datagen_dict=rundict['datagen_dict']
                    my_optimizedict['savepath']=rundict['savepath']
                    my_optimizedict['jobpath']=rundict['jobpath']
                    try:
                        '''jobsavepath=rundict['savepath']
                        jobsavefolder=os.path.split(jobsavepath)[0]
                        if not os.path.exists(jobsavefolder):os.makedirs(jobsavefolder)'''
                        kc.run_model_as_node(my_optimizedict,my_datagen_dict,force_start_params=1)
                        jobsuccess=1
                    except:
                        self.logger.exception('putting job back in jobq with savepath:{jobsavepath}')
                        jobq.put(rundict)
                        self.logger.debug('job back in jobq')
                    if jobsuccess:
                        with open(kc.nodesavepath,'rb') as f:
                            model_save_list=pickle.load(f)
                        qtry=0
                        while True:
                            self.logger.debug(f'adding model_save_list to saveq, kc.nodesavepath:{kc.nodesavepath}')
                            try:
                                qtry+=1
                                saveq.put(model_save_list)
                                self.logger.debug(f'model_save_list sucesfully added to saveq, savepath: {kc.nodesavepath}')
                                break
                            except:
                                if not saveq.full() and qtry>3:
                                    self.logger.exception('')
                                else:
                                    sleep(1)
                                    
                        
            except:
                self.logger.exception('')           
    
    
class RunCluster(mp.Process,kernelcompare.KernelCompare):
    '''
    '''
    
    def __init__(self,source=None,optdict_variation_list=None,datagen_variation_list=None,dosteps=1,local_run=None,nodecount=0,qdict=None):
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
            self.netaddress=('192.168.1.100',50002)
            qm=TheQManager(self.netaddress,None)
            qm.start()
            sleep(1)
        #self.qdict={'saveq':mp.Queue(),'jobq':mp.Queue()}
        #qm=TheQManager(self.netaddress,self.qdict)
        
        
        
        #self.SaveQDumper=SaveQDumper(self.qdict['saveq'])#run by runmaster, never started
        #saveqdumper=SaveQDumper(self.qdict['saveq'],None)
        #saveqdumper=SaveQDumper(None,self.netaddress)
        #saveqdumper.start()
        if nodecount:
            #self.nodelist=[RunNode(source=source,local_run=local_run,qdict=self.qdict) for _ in range(nodecount)]
            self.nodelist=[RunNode(source=source,local_run=local_run,qdict=qdict) for _ in range(nodecount)]
            [node.start() for node in self.nodelist]
        #according to the docs:https://docs.python.org/3.7/library/multiprocessing.html#using-a-remote-manager,
        #     I need to get local queue users started up before getting the network going
        #qm=TheQManager(self.netaddress,self.qdict)
        #qm.start()
            
        self.savechecktimeout_hours=2
        seed(1)  
        self.nodecount=nodecount
        self.dosteps=dosteps
        self.optdict_variation_list=optdict_variation_list
        self.datagen_variation_list=datagen_variation_list
        if source is None: source='monte'
        self.source=source
        self.savedirectory='results'
        if not os.path.exists(self.savedirectory):os.mkdir(self.savedirectory)
        
       
        kernelcompare.KernelCompare.__init__(self,directory=self.savedirectory,source=source)
        self.jobdirectory=os.path.join(self.savedirectory,'jobs')
        self.modelsavedirectory=os.path.join(self.savedirectory,'saves')
        if not os.path.exists(self.jobdirectory): os.mkdir(self.jobdirectory)
        if not os.path.exists(self.modelsavedirectory): os.mkdir(self.modelsavedirectory)
        super(RunCluster,self).__init__()
        
        
    def run(self,):
        self.mastermaster() 
        

    def generate_rundicts_from_variations(self,step=None):
        if self.optdict_variation_list is None:
            self.optdict_variation_list=self.getoptdictvariations(source=self.source)
        if self.datagen_variation_list is None:
            self.datagen_variation_list=self.getdatagenvariations(source=self.source)
        initial_datagen_dict=self.setdata(self.source)
        list_of_run_dicts=self.prep_model_list(optdict_variation_list=self.optdict_variation_list,datagen_variation_list=self.datagen_variation_list,datagen_dict=initial_datagen_dict)
        #list_of_run_dicts=list_of_run_dicts[-1::-1]#reverse the order of the list
        self.setup_save_paths(list_of_run_dicts,step=step)
        #p#rint(f'list_of_run_dicts[0:2]:{list_of_run_dicts[0:2]},{list_of_run_dicts[-2:]}')
        return list_of_run_dicts
    
    
    def setup_save_paths(self,rundictlist,step=None):
        if step is None:
            step=0
        for idx,rundict in enumerate(rundictlist):
            jobstepdir=os.path.join(self.jobdirectory,'step'+str(step))
            if not os.path.exists(jobstepdir): os.mkdir(jobstepdir)
            jobpath=os.path.join(jobstepdir,str(idx)+'_jobdict')
            try: species=rundict['datagen_dict']['species']
            except: 
                self.logger.exception('')
                species=''
            savestepdir=os.path.join(self.modelsavedirectory,'step'+str(step))
            if not os.path.exists(savestepdir): os.mkdir(savestepdir)
            savepath=os.path.join(savestepdir,'species-'+species+'_model_save_'+str(idx))
            rundict['jobpath']=jobpath
            rundict['savepath']=savepath
            
            
    def getqdict(self):
        BaseManager.register('jobq')
        BaseManager.register('saveq')
        m = BaseManager(address=self.netaddress, authkey=b'qkey')
        m.connect()
        jobq = m.jobq()
        saveq = m.saveq()
        return {'saveq':saveq,'jobq':jobq}    
            
    def mastermaster(self,):
        if not self.dosteps:
            list_of_run_dicts=self.generate_rundicts_from_variations()
            return self.runmaster(list_of_run_dicts)
        
        if self.qdict is None:
            self.qdict=self.getqdict()
        
        model_run_stepdict_list=self.build_stepdict_list() #parameters set in kernelparams
        
        for i,stepdict in enumerate(model_run_stepdict_list):
            
            self.logger.debug(f'i:{i}, stepdict:{stepdict}')
            #stepfolders=stepdict['stepfolders']
            try:
                self.logger.debug(f'i:{i},stepdict:{stepdict}')
                if 'variations' in stepdict:
                    list_of_run_dicts=self.generate_rundicts_from_variations()
                    runmasterresult=self.runmaster(list_of_run_dicts)
                    #self.logger.info(f'step#:{i} completed, runmasterresult:{runmasterresult}')
                else:
                    resultslist=[]

                    for functup in stepdict['functions']:
                        args=functup[1]
                        if args==[]:
                            args=[resultslist[-1]]
                        kwargs=functup[2]
                        result=functup[0](*args,**kwargs)
                        resultslist.append(result)
                    list_of_run_dicts=resultslist[-1]
                    self.logger.debug(f'step:{i} len(list_of_run_dicts):{len(list_of_run_dicts)}')
                    #self.rundict_advance_path(list_of_run_dicts,i,stepfolders)
                    runmasterresult=self.runmaster(list_of_run_dicts)
                self.logger.info(f'step#:{i} completed, runmasterresult:{runmasterresult}')
            except:
                self.logger.exception(f'i:{i},stepdict:{stepdict}')
                assert False,'halt'
        self.qdict['saveq'].put('shutdown')
        #saveqdumper.join()
                
                
                
    def runmaster(self,list_of_run_dicts):
        try:
            pathlist=[run_dict['savepath'] for run_dict in list_of_run_dicts]
            self.logger.debug(f"len(pathlist):{len(pathlist)}")
            
            self.logger.debug(f'len(list_of_run_dicts):{len(list_of_run_dicts)}')
            jobqfiller=JobQFiller(self.qdict['jobq'],list_of_run_dicts)
            #jobqfiller.start()
            #jobqfiller.join()
            jobqfiller.run()
            self.logger.debug('back from jobqfiller')
            saveqdumper=SaveQDumper(self.qdict['saveq'])
            while pathlist:
                sleep(5)
                saveqdumper.run()
                pathcount1=len(pathlist)
                pathlist=self.savecheck(pathlist)
                pathcount2=len(pathlist)
                if pathcount1>pathcount2:
                    self.logger.debug(f'saveqdumper returned {pathcount1-pathcount2} fewer paths. remaining paths:{pathcount2}')
                    print(f'saveqdumper returned {pathcount1-pathcount2} fewer paths. remaining paths:{pathcount2}')
            return
        except:
            self.logger.exception('')
    
    
    
    def savecheck(self,pathlist):
        try:
            i=0
            sleeptime=5 # seconds
            while pathlist:
                path=pathlist.pop()
                if not os.path.exists(path):
                    pathlist.append(path)
                    return pathlist
        
            return pathlist
        except:
            self.logger.exception('')
        
           

if __name__=="__main__":
    RunNode(local_run=0)
