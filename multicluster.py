from time import sleep
import multiprocessing as mp


class mypool:
    def __init__(self, workercount=2,includemaster=0):
        self.workercount=workercount
        if includemaster==1:
            self.arg_list=['master']+['node'+str(randint(0,100))]*(workercount-1)
        else:
            self.arg_list = ['node'] * (workercount)
        self.runpool(self.arg_list,self.workercount)
        self.i=0

    def runpool(self,arg_list,workercount):
        with mp.Pool(processes=workercount) as pool:
            pool.map(self.runcluster,arg_list)

    def runcluster(self,name):
        if not name=='master':
            sleep(randint(25,200)/25)#make nodes start at different times
        while True:
            try:
                mycluster.run_cluster(mytype=name,local_test='no')

            except:
                print(f'restarting:{name}')


if __name__=='__main__':
    include_master=int(input('1 for include master, 0 for not'))
    workercount=int(input('worker count:'))

    if include_master==0:
        test=mypool(workercount)
    if include_master==1:
        test=mypool(workercount-1,1)
