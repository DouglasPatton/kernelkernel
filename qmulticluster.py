from time import sleep
import multiprocessing as mp
from random import randint,seed
import os,sys#,psutil
import qcluster
from datetime import datetime
import logging,logging.config
#import yaml
from numpy import log


class mypool:
    def __init__(self,source='monte', nodecount=1,includemaster=1,local_run='no'):

        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'multicluster.log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(os.path.join(logdir,handlername), maxBytes=10**7, backupCount=100)],
                level=logging.WARNING,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
      
        #handler=logging.RotatingFileHandler(os.path.join(logdir,handlername),maxBytes=8000, backupCount=5)
        self.logger = logging.getLogger(handlername)
        self.includemaster=includemaster
        self.source=source

        self.sleepbetweennodes=8 # seconds
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
                master=qcluster.RunCluster(local_run=self.local_run,qdict=qdict,source=self.source,nodecount=0)
                proclist=[master]
            else:proclist=[]
                #master.join()
            self.logger.debug('creating nodes')
            proclist.extend([qcluster.RunNode(local_run=self.local_run,qdict=qdict,source=self.source) for _ in range(self.nodecount)])
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
