import os
import kernelcompare as kc

mydir=os.getcwd()
mydir0=os.path.join(mydir,'..')
mydir1=os.path.join(mydir,"cluster_test")


if __name__=='__main__':
    print(f'type 1 for {mydir1}')
    print(f'type 0 for {mydir0}')
    whichdir=input('or enter a directory manually: ')
    wantprint=input('type 1 to print results to html or 0 to skip: ')
    condensechoice=int(input('type 1 to condense, 0 to not condense: '))
    if str(whichdir)==str(1):
        thedir=mydir1
    elif str(whichdir)==str(0):
        thedir=mydir0
    else:
        thedir=whichdir
    print(f'thedir:{thedir}')
    test=kc.KernelCompare(directory=thedir)
    test.process_pisces_models(thedir)
    
    #test.merge_and_condense_saved_models(merge_directory=thedir,save_directory=None,condense=condensechoice,recondense=None,verbose=None,recursive=1)
    #test.recursive_merge(thedir,overwrite=0,verbose=1,condense=condensechoice)    
    if wantprint==str(1):
        test.print_model_save(filename=os.path.join(thedir,'mergedfiles',"condensed_model_save"))
