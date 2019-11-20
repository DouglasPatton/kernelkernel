import mycluster
import multiprocessing as mp
from random import randint


class mypool:
    def __init__(self, worker_count=2,includemaster=0):
        if includemaster==1:
            arg_list=['master']+['node'+str(randint(0,100))]*(worker_count-1)
        else
            arg_list = ['node'] * (worker_count)







