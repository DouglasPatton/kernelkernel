from multicluster import mypool
import psutil,sys
if __name__=="__main__":

    cores=int(psutil.cpu_count(logical=True))
    platform=sys.platform
    p=psutil.Process(os.getpid())
    if platform=='win32':
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        p.nice(6)
    print(f'choose a node count between 1 and {cores-2}')
    nodecount=int(input('node count:'))
    

    mypool(nodecount=nodecount,
                source='pisces',
                includemaster=0,
                local_run=0
               )
