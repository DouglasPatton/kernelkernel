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
            level=logging.DEBUG,
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
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.logger.info('SaveQDumper starting')
        #self.BaseManager=BaseManager
        super(SaveQDumper,self).__init__()
    
    def summarizeModelSaveList(self,model_save_list):
        try:
            model0=model_save_list[0]
            model_save_summary={}
            model_save_summary['savepath']=model0['savepath']
            model_save_summary['naiveloss']=model0['naiveloss']
            modeldict0=model0['modeldict']
            loss_function=modeldict0['loss_function']
            lossdictlist=[savedict['lossdict'] for savedict in model_save_list]
            try:
                validate=modeldict0['validate']
            except:
                validate=0
            if not validate:
                losslist=[lossdict[loss_function] for lossdict in lossdictlist]
                minloss=min(losslist)
                minlosspos=losslist.index(minloss)
                model_save_summary['loss']=minloss
                model_save_summary['lossdict']=lossdictlist[minlosspos]
                model_save_summary['params']=model_save_list[minlosspos]['params']
                model_save_summary['binary_y_result']=model_save_list[minlosspos]['binary_y_result']
            else:
                keylist=[key for key in lossdictlist[0]]
                meanlossdict={key:np.mean(np.array([lossdict[key] for lossdict in lossdictlist]))for key in keylist}
                meanloss=meanlossdict[loss_function]

                by_keylist=[res[0] for res in model_save_list[0]['binary_y_result']]
                meanbinary_y_result=[]
                for k,key in enumerate(by_keylist):
                    resultlist=[]
                    for modelsave in model_save_list:
                        byr=modelsave['binary_y_result'][k][1]
                        resultlist.append(byr)
                        meanresult=np.mean(np.array(resultlist))
                    meanbinary_y_result.append((key,meanresult))
                model_save_summary['loss']=meanloss
                model_save_summary['lossdict']=meanlossdict
                model_save_summary['binary_y_result']=meanbinary_y_result
                model_save_summary['params']=model_save_list[0]['params'] #should all have same params for validation
            return model_save_summary
        except:
            self.logger.exception('summary error')
                    
    
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
                    model_save_summary=self.summarizeModelSaveList(model_save_list)
                    #self.logger.debug(f'SaveQDumper got: {model_save_list}')
                    loss=model_save_summary['loss']
                    lossdict=model_save_summary['lossdict']
                    naiveloss=model_save_summary['naiveloss']
                    binary_y_result=model_save_summary['binary_y_result']
                    params=model_save_summary['params']
                    message=f"SaveQDumper has {model_save_summary['savepath']} with lossdict:{lossdict}, naiveloss:{naiveloss}, binary_y_result:{binary_y_result} for params:{params}"
                    print(message)
                    self.logger.debug(message)
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
                    nodesavepath=model_save_summary['savepath']
                    with open(nodesavepath,'wb') as f:
                        pickle.dump(model_save_list,f)
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
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.logger.info('JobQFiller starting')
        super(JobQFiller,self).__init__()
    
    '''def run(self):
        #QueueManager.register('jobq')
        #m = QueueManager(address=self.netaddress, authkey=b'qkey')
        #m.connect()
        #queue = m.jobq()
        queue=self.q
        a_rundict=self.joblist[-1]
        a_savepath=a_rundict['savepath']
        savedir=os.path.split(a_savepath)[0]
        savedset=set(os.listdir(savedir))
        print(f'len(savedset):{len(savedset)}')
        
        print('creating notsavedset')
        notsaved_joblist=[job for job in self.joblist if os.path.split(job['savepath'])[-1] not in savedset]
        print(f'len(notsaved_joblist):{len(notsaved_joblist)}')
        print(f'notsaved_joblist[0]:{notsaved_joblist[0]}')
        shuffle(notsaved_joblist)
        jobcount=len(notsaved_joblist)
        i=1
        while len(notsaved_joblist):
            job=notsaved_joblist.pop()
            try:
                self.logger.debug(f'adding job:{i}/{jobcount} to job queue')
                queue.put(job)
                self.logger.debug(f'job:{i}/{jobcount} succesfully added to queue')
                i+=1
            except:
                notsaved_joblist.append(job)
                if queue.full():
                    self.logger.DEBUG('jobq full, waiting 4s')
                    sleep(4)
                else:
                    self.logger.exception(f'jobq error for i:{i}')
        self.logger.debug('all jobs added to jobq.')
        print('all jobs added to jobq.')
        return'''
    
    def run(self):
        #QueueManager.register('jobq')
        #m = QueueManager(address=self.netaddress, authkey=b'qkey')
        #m.connect()
        #queue = m.jobq()
        queue=self.q
        jobcount=len(self.joblist)
        a_rundict=self.joblist[-1]
        a_savepath=a_rundict['savepath']
        savedir=os.path.split(a_savepath)[0]
        savedset=set(os.listdir(savedir))
        print(f'len(savedset):{len(savedset)}')
        shuffle(self.joblist)
        i=1
        while len(self.joblist):
            job=self.joblist.pop()
            savepath=job['savepath']
            if os.path.split(savepath)[1] not in savedset:
            #if not os.path.exists(savepath):
                #with open(job['jobpath'],'wb') as f:
                #    pickle.dump(job,f)
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
            else:
                self.logger.info(f'JobQFiller is skipping job b/c saved at savepath:{job["savepath"]}')
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
                level=logging.DEBUG,
                format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%Y-%m-%dT%H:%M:%S')
            self.logger = logging.getLogger(handlername)
            self.logger.info('RunNode logging')
        
        self.qdict=qdict
        self.source=source
        if not local_run:
            self.netaddress=('192.168.1.45',50002)
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
        kc=kernelcompare.KernelCompare(source=self.source) # a new one every run
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
        try:
            self.logger=logging.getLogger(__name__)
            self.logger.info('starting RunCluster object')
        except:
            logdir=os.path.join(os.getcwd(),'log')
            if not os.path.exists(logdir): os.mkdir(logdir)
            handlername=os.path.join(logdir,f'mycluster_.log')
            logging.basicConfig(
                handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
                level=logging.DEBUG,
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
        self.logger.debug('mastermaster starting up')
        self.mastermaster() 
        

    def generate_rundicts_from_variations(self,step=None,optdict_variation_list=None,datagen_variation_list=None):
        if optdict_variation_list is None:
            if self.optdict_variation_list is None:
                optdict_variation_list=self.getoptdictvariations(source=self.source)
            else:optdict_variaton_list=self.optdict_variation_list
        if datagen_variation_list is None:
            if self.datagen_variation_list is None:
                self.datagen_variation_list=self.getdatagenvariations(source=self.source)
            else: datagen_variation_list=self.datagen_variation_list
        initial_datagen_dict=self.setdata(self.source)
        list_of_run_dicts=self.prep_model_list(optdict_variation_list=optdict_variation_list,datagen_variation_list=datagen_variation_list,datagen_dict=initial_datagen_dict)
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
        try:
            if not self.dosteps:
                list_of_run_dicts=self.generate_rundicts_from_variations()
                return self.runmaster(list_of_run_dicts)

            if self.qdict is None:
                self.qdict=self.getqdict()
            
            self.logger.debug('saveQdumper started, now building pipeline')
            pipelinesteps=self.build_pipeline()
            self.logger.debug(f'pipelinesteps:{pipelinesteps}')
            for pipestepdict in pipelinesteps:
                self.logger.debug(f'pipestepdict:{pipestepdict}')
                try:
                    list_of_run_dicts=self.processPipeStep(pipestepdict)
                    self.logger.debug(f'after processPipeStep, len(list_of_run_dicts):{len(list_of_run_dicts)}')
                    runmasterresult=self.runmaster(list_of_run_dicts)
                except:
                    self.logger.exception(f'error in mastermaster. pipestepdict:{pipestepdict}')
                    assert False,'halt'
            self.qdict['saveq'].put('shutdown')
        except:
            self.logger.exception('')
            assert False, 'Halt'
        #saveqdumper.join()
                
                
                
    def runmaster(self,list_of_run_dicts):
        try:
            pathlist=[run_dict['savepath'] for run_dict in list_of_run_dicts]
            self.logger.debug(f"len(pathlist):{len(pathlist)}")
            self.logger.debug(f'len(list_of_run_dicts):{len(list_of_run_dicts)}')
            jobqfiller=JobQFiller(self.qdict['jobq'],list_of_run_dicts)
            jobqfiller.run()
            saveqdumper=SaveQDumper(self.qdict['saveq'])
            
            self.logger.debug('back from jobqfiller.run()')
            while pathlist:
                sleep(5)
                saveqdumper.run()
                pathcount1=len(pathlist)
                pathlist=self.savecheck(pathlist)
                pathcount2=len(pathlist)
                if pathcount1>pathcount2:
                    status_string=f'pathlist has {pathcount1-pathcount2} fewer paths. remaining paths:{pathcount2}'
                    self.logger.debug(status_string)
                    print(status_string)
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
