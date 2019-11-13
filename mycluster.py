import pickle
import os
import re
from time import strftime
import datetime

class run_cluster:
    def __init__(self,mytype):
        self.savedirectory=O:\Public\DPatton\kernel
        self.namelist=self.getnamelist()
        self.initialize(mytype)
        
        
def initialize(self,mytype):
    if mytype="master":
        self.runmaster()
    else:
        myname=mytype
        mytype="node"
        name_regex=r'^'+re.escape(myname)
        
        namematch=[1 for name in self.namelist if re.search(name_regex,name)]
        namecount=len(namematch)
        if namecount>0
            nameset=0
            while nameset==0:
                mynametry=myname+f'{namecount}'
                name_regex2=r'^'+mynametry
                namematch2=[1 for name in self.namelist if re.search(name_regex2,name)]
                if len(namematch2)>0:
                    namecount+=1
                if len(namematch2)==0:
                    myname=mynametry
                    print()
                    nameset=1
        try:
            self.add_to_namelist(myname):
        except:
            return self.initialize(myname) #this should re-start the process if something went wrong
        self.runnode(myname)
        

                
        

def add_to_namelist(self,newname):
    try: 
        with open(self.savedirectory+'namelist','rb') as savednamelist:
            namelist=pickle.load(savednamelist)
            
    except: 
        namelist=[]
    namelist.append(newname)
    try: 
        with open(self.savedirectory+'namelist','wb') as savednamelist:
            assert len([1 for name in self.namelist if name==newname])==0,"newname has  match"
            pickle.dump(savedname,savednamelist)
        with open(self.savedirectory+'namelist','rb') as savednamelist:
            self.namelist=pickle.load(savednamelist)
            assert len([1 for name in self.namelist if name==newname])>1,"newname has too many matches!!!"
            
    except:
        assert False,"naming failure"
            
def getnamelist(self):
    try: 
        with open(self.savedirectory+'namelist','rb') as namelist
            return pickle.load(namelist)
    except:
        print('no file called namelist found')
        return []
    
def runnode(self,myname):
    start_time=strftime("%Y%m%d-%H%M%S")
    i_have_opt_job=0
    while i_have_opt_job==0:
        my_opt_job=self.check_for_opt_job(myname,start_time)
        i_have_opt_job=1
    
    
def check_for_opt_job(self,myname,start_time):
    assert type(myname) is str,f"myname should be type str not type:{type(myname)}"
    try: 
        os.chdir(self.savedirectory)
    except:
        os.mkdir(self.savedirectory)
        os.chdir(self.savedirectory)
    
    mydir=self.savedirectory+myname
    myfile=mydir+myname+'_job'
    try:
        os.chdir(mydir)
    except:
        print("mydir:{mydir} doesn't exist, so creating it and waiting")
        os.mkdir(mydir)
        os.chdir(mydir)
        waiting=0
        i=0
        while waiting==0:
            try: with open(myfile,'rb') as myjob:
                    if myjob['job_to_do']==1:
                        return pickle.load(myjob)
                    else:
                        print('job_to_do:',myjob['job_to_do'])
                        waiting=1
            except:
                i+=1
                if i%500==0:
                    print("waiting...i=",i,end='. ')
                    now=strftime("%Y%m%d-%H%M%S")
                s_since_start=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(start_time,"%Y%m%d-%H%M%S"))
                if s_since_start>60*10:#10 minutes
                    print(f'time out after {s_since_start}')
                    waiting=1
        return None
                
            
                    

        
                    
                