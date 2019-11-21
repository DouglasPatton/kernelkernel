from time import sleep
import multiprocessing as mp
from random import randint
import traceback
import mycluster

class mypool:
    def __init__(self, nodecount=2,includemaster=0,local_test='no'):
        self.local_test=local_test
        self.i=0
        self.id=randint(0,100000000)
        self.nodecount=nodecount
        if includemaster==1:
            self.arg_list=['master']+['node']*(nodecount)
        else:
            self.arg_list = ['node'] * (nodecount)
        self.runpool(self.arg_list,self.nodecount)
        self.i=0

    def runpool(self,arg_list,nodecount):
        with mp.Pool(processes=nodecount) as pool:
            pool.map(self.runcluster,arg_list)

    def runcluster(self,name):
        if not name=='master':
            sleep(randint(25,100)/25)#make nodes start at different times
        while True:
            try:
                self.i+=1#increments with start/restart of nodes or master
                if name==master:
                    mycluster.run_cluster(name, local_test='no')
                else:
                    mycluster.run_cluster(name+str(self.id)+'-'+str(self.i),local_test='no')

            except:
                print(f'restarting:{name}')
                print(traceback.format_exc())


if __name__=='__main__':
    include_master=int(input('1 for include master, 0 for not'))
    nodecount=int(input('worker count:'))
    
    if include_master==0:
        test=mypool(nodecount)
    if include_master==1:
        test=mypool(nodecount,1)
