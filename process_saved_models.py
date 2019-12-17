import os
import kernelcompare as kc

networkdir='o:/public/dpatton/kernel'
mydir=os.getcwd()
mydir2=os.path.join(mydir,"cluster_test")


if __name__=='__main__':
    print(f'type 0 for cluster_test directory:{mydir2}')
    print(f'type 1 for network directory,{networkdir}')
    whichdir=input('or enter a directory manually: ')
    if str(whichdir)==str(0):
        thedir=mydir2
    elif str(whichdir)==str(1):
        thedir=networkdir
    else:
        thedir=whichdir
    print(f'thedir:{thedir}')
    test=kc.KernelCompare(directory=thedir)
    test.recursive_merge(thedir,overwrite=0,verbose=1)    
