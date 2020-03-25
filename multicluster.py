from time import sleep
import multiprocessing as mp
from random import randint,seed
import os,sys,psutil
import mycluster
from datetime import datetime
import logging,logging.config
import yaml
from numpy import log


class mypool:
    def __init__(self,source='monte', nodecount=1,includemaster=1,local_run='no'):

        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'multicluster.log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(os.path.join(logdir,handlername), maxBytes=10**7, backupCount=0)],
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
      
        #handler=logging.RotatingFileHandler(os.path.join(logdir,handlername),maxBytes=8000, backupCount=5)
        self.logger = logging.getLogger(handlername)
        self.includemaster=includemaster
        self.source=source

        self.sleepfactor=0.1 #0.2->4min. 0.1->6sec, 0.224 ->10min
        
        #logging.basicConfig(level=logging.INFO)
        #with open(os.path.join(os.getcwd(),'logconfig.yaml'),'rt') as f:
        #    configfile=yaml.safe_load(f.read())
        #logging.config.dictConfig(configfile)
        #self.logger = logging.getLogger('multiClusterLogger')

        ''' platform=sys.platform
        p=psutil.Process(os.getpid())
        if platform=='win32':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(6)'''
        #seed(1)
        #seed(datetime.now())
        self.local_run=local_run
        self.i=0
        self.id=randint(0,100000000)
        self.nodecount=nodecount
        self.workercount=nodecount
        
        #if includemaster==1:
        #    self.workercount=nodecount+1
        #    self.arg_list=['master']+['node']*(nodecount)
        #else:
        #    self.arg_list = ['node'] * (nodecount)
        self.arg_list = ['node'] * (nodecount)
        self.runpool(self.arg_list,self.workercount)
        

    def runpool0(self,arg_list,workercount):
        with mp.Pool(processes=workercount) as pool:
            pool.map(self.runworker,arg_list)
            
            
    def runpool(self,arg_list,workercount):
        if self.includemaster:
            master_proc=mp.Process(target=self.runworker,args=('master',))
            master_proc.start()
            
        process_list=[None]*workercount
        for i in range(workercount):
            self.i+=1
            process_list[i]=mp.Process(target=self.runworker,args=(arg_list[i],))
            process_list[i].start()
        for i in range(workercount):
            process_list[i].join()
        if self.includemaster:
            master_proc.join()
        print('============================================')
        print('==========multicluster has completed=========')
        print('============================================')
    
        '''     processes = [None] * 4
        for i in range(4):
            processes[i] = multiprocessing.Process(target=child_process.run, args=(i,))
            processes[i].start()
        for i in range(4):
            processes[i].join()
        '''
    
    def runworker(self,startname):
        
        if startname=='master':
            rerun=True
            while rerun:
                try:
                    mycluster.run_cluster(myname=startname, source=self.source, local_run=self.local_run)
                except KeyboardInterrupt:
                    rerun=False
                except:
                    sleep(1)
                    #self.i+=500
                    print(f'restarting:{startname}')
                    self.logger.exception(f'error in {__name__}')

        sleeptime=(self.sleepfactor*log(float(randint(5000000,99999999))/2000))**8

        print(f'sleeping for {sleeptime/60} minutes')
        sleep(sleeptime)#make nodes start at different times
        rerun=True
        while rerun:
            try:
                self.i+=1#increments with start/restart of nodes or master
                name=startname+str(self.id)+'-'+str(self.i)
                mycluster.run_cluster(myname=name,local_run=self.local_run,source=self.source)
            except KeyboardInterrupt:
                rerun=False

            except:
                sleep(1)
                print(f'restarting:{name}')
                self.logger.exception(f'error in {__name__}')


if __name__=='__main__':
    #test = mypool(nodecount=1, includemaster=1,local_run='yes')
    choose_source=int(input(('0 for monte carlo, 1 for pisces')))
    if choose_source==0:
        source='monte'
    elif choose_source==1:
        source='pisces'
    else: 
        assert False, 'choose_source not 0 or 1'
    local_run=int(input('1 for local_run or 0 for network run'))
    includemaster=int(input('1 for include master, 0 for not'))
    nodecount=int(input('node count:'))
    

    test=mypool(nodecount=nodecount,
                source=source,
                includemaster=includemaster,
                local_run=local_run
               )
