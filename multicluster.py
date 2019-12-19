from time import sleep
import multiprocessing as mp
from random import randint,seed
import os
import mycluster
from datetime import datetime
import logging
import logging.config
import yaml
from numpy import log



class mypool:
    def __init__(self, nodecount=1,includemaster=1,local_test='no'):
        with open(os.path.join(os.getcwd(),'logconfig.yaml'),'rt') as f:
            configfile=yaml.safe_load(f.read())
        logging.config.dictConfig(configfile)
        self.logger = logging.getLogger('multiClusterLogger')
        seed(datetime.now())
        self.local_test=local_test
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
                    mycluster.run_cluster(startname, local_test=self.local_test)
                except KeyboardInterrupt:
                    rerun=False
                except:
                    self.i+=1
                    print(f'restarting:{startname}')
                    self.logger.exception(f'error in {__name__}')
        sleeptime=3*log(25*float(self.i))
        print(f'sleeping for {sleeptime/60} minutes')
        sleep(sleeptime)#make nodes start at different times
        rerun=True
        while rerun:
            try:
                self.i+=1#increments with start/restart of nodes or master
                name=startname+str(self.id)+'-'+str(self.i)
                mycluster.run_cluster(name,local_test=self.local_test)
            except KeyboardInterrupt:
                rerun=False

            except:
                print(f'restarting:{name}')
                self.logger.exception(f'error in {__name__}')


if __name__=='__main__':
    #test = mypool(nodecount=1, includemaster=1,local_test='yes')
    local_test=int(input('1 for local_test or 0 for network run'))
    includemaster=int(input('1 for include master, 0 for not'))
    nodecount=int(input('node count:'))
    

    test=mypool(nodecount=nodecount,includemaster=includemaster,local_test=local_test)
