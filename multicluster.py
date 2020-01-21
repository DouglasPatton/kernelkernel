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
    def __init__(self,data_source='monte' nodecount=1,includemaster=1,local_run='no'):
        '''logging.basicConfig(level=logging.INFO)
        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=f'multicluster.log'
        handler=logging.FileHandler(os.path.join(logdir,handlername))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)'''
        self.data_source=data_source
        logging.basicConfig(level=logging.INFO)
        with open(os.path.join(os.getcwd(),'logconfig.yaml'),'rt') as f:
            configfile=yaml.safe_load(f.read())
        logging.config.dictConfig(configfile)
        self.logger = logging.getLogger('multiClusterLogger')

        platform=sys.platform
        p=psutil.Process(os.getpid())
        if platform=='win32':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(6)

        #seed(datetime.now())
        self.local_run=local_run
        self.i=0
        self.id=randint(0,100000000)
        self.nodecount=nodecount
        self.workercount=nodecount
        if includemaster==1:
            self.workercount=nodecount+1
            self.arg_list=['master']+['node']*(nodecount)
        else:
            self.arg_list = ['node'] * (nodecount)
        self.runpool(self.arg_list,self.workercount)
        

    def runpool0(self,arg_list,workercount):
        with mp.Pool(processes=workercount) as pool:
            pool.map(self.runworker,arg_list)
            
            
    def runpool(self,arg_list,workercount):
        process_list=[None]*workercount
        for i in range(workercount):
            self.i+=1
            process_list[i]=mp.Process(target=self.runworker,args=(arg_list[i],))
            process_list[i].start()
        for i in range(workercount):
            process_list[i].join()
    
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
                    mycluster.run_cluster(startname, data_source=self.data_source, local_run=self.local_run)
                except KeyboardInterrupt:
                    rerun=False
                except:
                    #self.i+=500
                    print(f'restarting:{startname}')
                    self.logger.exception(f'error in {__name__}')

        sleeptime=(.2*log(float(randint(5000000,99999999))/2000))**8#0.2->4min. 0.1->6sec

        print(f'sleeping for {sleeptime/60} minutes')
        sleep(sleeptime)#make nodes start at different times
        rerun=True
        while rerun:
            try:
                self.i+=1#increments with start/restart of nodes or master
                name=startname+str(self.id)+'-'+str(self.i)
                mycluster.run_cluster(name,local_run=self.local_run)
            except KeyboardInterrupt:
                rerun=False

            except:
                print(f'restarting:{name}')
                self.logger.exception(f'error in {__name__}')


if __name__=='__main__':
    #test = mypool(nodecount=1, includemaster=1,local_run='yes')
    choose_data_source=int(input(('0 for monte carlo, 1 for pisces')))
    if choose_data_source==0:
        data_source='monte'
    elif choose_data_source==1:
        data_source='pisces'
    else: 
        assert False, 'choose_data_source not 0 or 1'
    local_run=int(input('1 for local_run or 0 for network run'))
    includemaster=int(input('1 for include master, 0 for not'))
    nodecount=int(input('node count:'))
    

    test=mypool(nodecount=nodecount,
                data_source=data_source,
                includemaster=includemaster,
                local_run=local_run
               )
