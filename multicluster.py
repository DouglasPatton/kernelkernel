import mycluster
import multiprocessing as mp
from random import randint
from time import sleep


class mypool:
    def __init__(self, workercount=2,includemaster=0):
        self.workercount=workercount
        if includemaster==1:
            self.arg_list=['master']+['node'+str(randint(0,100))]*(workercount-1)
        else:
            self.arg_list = ['node'] * (workercount)
        self.runpool(self.arg_list,self.workercount)
        self.i=0

    def runpool_async(self,arg_list,workercount):
        with mp.Pool(processes=workercount) as pool:
            [pool.apply_async(self.runcluster,self.arg_list) for i in range(self.workercount)]

    def runpool(self,arg_list,workercount):
        with mp.Pool(processes=workercount) as pool:
            pool.map(self.runcluster,arg_list)

    def runcluster(self,name):
        if not name=='master':
            sleep(randint(25,200)/25)#make nodes start at different times
        while True:
            try:
                mycluster.run_cluster(mytype=name)

            except:
                print(f'restarting:{name}')


if __name__=='__main__':
    include_master=input('1 for include master, 0 for not')
    workercount=input('worker count:')
    if type(include_master) is str:
        include_master=include_master.lower()

    assert type(workercount) is int and workercount>0, f"workercount should be positive int but is type:{type(workercount)}"
    if include_master=0 or include_master=='no':
        test=mypool(workercount,0)
    if include_master=1 or include_master='yes':
        test=mypool(workercount-1,1)
