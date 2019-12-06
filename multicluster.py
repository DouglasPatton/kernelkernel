from time import sleep
import multiprocessing as mp
from random import randint,seed
import traceback
import mycluster
from datetime import datetime

class mypool:
    def __init__(self, nodecount=1,includemaster=1,local_test='no'):
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
        self.i=0

    def runpool0(self,arg_list,workercount):
        with mp.Pool(processes=workercount) as pool:
            pool.map(self.runworker,arg_list)
            
            
    def runpool(self,arg_list,workercount):
        process_list=[None]*workercount
        for i in range(workercount):
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
    
    def runworker(self,name):
        if name=='master':
            rerun=True
            while rerun:
                try:
                    self.i+=1
                    mycluster.run_cluster(name, local_test=self.local_test)
                except KeyboardInterrupt:
                    rerun=False
                except:
                    print(f'restarting:{name}')
                    print(traceback.format_exc())
        sleeptime=randint(1,10000)*60/10000
        print(f'sleeping for {sleeptime/60} minutes')
        sleep(sleeptime)#make nodes start at different times
        rerun=True
        while rerun:
            try:
                self.i+=1#increments with start/restart of nodes or master
                mycluster.run_cluster(name+str(self.id)+'-'+str(self.i),local_test=self.local_test)
            except KeyboardInterrupt:
                rerun=False

            except:
                print(f'restarting:{name}')
                print(traceback.format_exc())


if __name__=='__main__':
    #test = mypool(nodecount=1, includemaster=1,local_test='yes')
    local_test='yes'
    includemaster=int(input('1 for include master, 0 for not'))
    nodecount=int(input('node count:'))
    

    test=mypool(nodecount=nodecount,includemaster=includemaster,local_test=local_test)
