from time import sleep
import multiprocessing as mp
from random import randint,seed
import os,sys#,psutil
import qcluster
from datetime import datetime
from mylogger import myLogger
#import yaml
from numpy import log


class mypool(myLogger):
    def __init__(self,source='pisces', nodecount=1,includemaster=1,local_run='no',run_type='fit',cv_run=False,cv_n_jobs=None,local_nodes=False):
        func_name='multicluster'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        self.run_type=run_type
        self.includemaster=includemaster
        self.source=source
        self.cv_run=cv_run
        self.cv_n_jobs=cv_n_jobs
        self.local_nodes=local_nodes

        self.sleepbetweennodes=1#8 # seconds
        self.local_run=local_run
        self.i=0
        self.id=randint(0,100000000)
        self.nodecount=nodecount
        self.runpool()
            
    def runpool(self,):
        try:
            if self.local_run:
                qdict={'jobq':mp.Queue(),'saveq':mp.Queue()}
                sleep(2)
            else:
                qdict=None
            if self.includemaster:
                master=qcluster.RunCluster(local_run=self.local_run,qdict=qdict,source=self.source,nodecount=0,run_type=self.run_type,cv_run=self.cv_run)
                proclist=[master]
            else:proclist=[]
                #master.join()
            self.logger.debug('creating nodes')
            proclist.extend([
                qcluster.RunNode(
                    local_run=self.local_run,qdict=qdict,source=self.source,
                    run_type=self.run_type,cv_n_jobs=self.cv_n_jobs,
                    local_node=self.local_nodes
                    ) for _ in range(self.nodecount)
                ])
            self.logger.debug('starting nodes')
            for proc in proclist:
                proc.start()
                sleep(self.sleepbetweennodes)
            [proc.join() for proc in proclist[::-1]]
        except:
             self.logger.exception('')
        print('============================================')
        print('==========multicluster has completed=========')
        print('============================================')
        
    
    

if __name__=='__main__':
    #test = mypool(nodecount=1, includemaster=1,local_run='yes')
    fit_or_predict=int(input('0 cv-fit, 1 for single-fit, 2 for fullfit predict, 3 for out of sample Xpredict: '))
    cv_run=False
    if fit_or_predict==0:
        run_type='fit'
        cv_run=True
    elif fit_or_predict==1:
        run_type='single_fit'

    elif fit_or_predict==2:
        run_type='predict'
        
    elif fit_or_predict==3:
        run_type='Xpredict'
    else: 
        assert False, 'fit_or_predict not 0, 1, 2 or 3'
    local_run=int(input('1 for local_run or 0 for network run: '))
    includemaster=int(input('1 for include master, 0 for not: '))
    nodecount=int(input('node count: '))
    if nodecount>0 and cv_run:
        cv_n_jobs=int(input('n_jobs per node: '))
    else: 
        cv_n_jobs=None
    if nodecount>0 and includemaster==0:
        local_nodes=True if int(input('1 for local_nodes and 0 for remote'))==1 else False
    else:
        local_nodes=False
    

    test=mypool(nodecount=nodecount,
                run_type=run_type,
                includemaster=includemaster,
                local_run=local_run,
                cv_run=cv_run,
                cv_n_jobs=cv_n_jobs,
                local_nodes=local_nodes
               )
