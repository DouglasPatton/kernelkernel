import os
import kernelcompare as kc

networkdir='o:/public/dpatton/kernel'
mydir=os.getcwd()
mydir2=os.path.join(mydir,"cluster_test")


if __name__=='__main__':
    print(f'type 1 for cluster_test directory:{mydir2}')
    print(f'type 0 for network directory,{networkdir}')
    whichdir=input('or enter a directory manually: ')
    wantprint=input('type 1 to print results to html or 0 to skip: ')
    condensechoice=int(input('type 1 to condense, 0 to not condense: '))
    if str(whichdir)==str(1):
        thedir=mydir2
    elif str(whichdir)==str(0):
        thedir=networkdir
    else:
        thedir=whichdir
    print(f'thedir:{thedir}')
    test=kc.KernelCompare(directory=thedir)
    test.merge_and_condense_saved_models(merge_directory=thedir,save_directory=None,condense=condensechoice,recondense=None,verbose=None,recursive=1)
    #test.recursive_merge(thedir,overwrite=0,verbose=1,condense=condensechoice)    
    if wantprint==str(1):
        test.print_model_save(filename=os.path.join(thedir,'mergedfiles',"condensed_model_save"))
