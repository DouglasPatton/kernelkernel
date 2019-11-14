import pickle
import os
import re
from time import strftime
import datetime
import kernelcompare 

class run_cluster(kernelcompare.KernelCompare):
    def __init__(self,mytype):
        self.old_threshold=60*20#60 seconds times 20 times before master assumes name is old and removes from list
        self.savedirectory='O:/Public/DPatton/kernel/'
        kernelcompare.Kernelcompare.__init__(self,self.savedirectory)
        
        self.initialize(mytype)
        
        
def initialize(self,mytype):
    if mytype="master":
        self.runmaster()
    else:
        myname=mytype
        mytype="node"
        name_regex=r'^'+re.escape(myname)
        namelist=self.getnamelist()
        namematch=[1 for name in namelist if re.search(name_regex,name[0])]
        namecount=len(namematch)
        if namecount>0
            nameset=0
            while nameset==0:
                mynametry=myname+f'{namecount}'
                name_regex2=r'^'+mynametry
                namematch2=[1 for name in namelist if re.search(name_regex2,name[0])]
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
        

def runmaster(self,opt_model_variation_list)
    self.rebuild_current_namelist()
    namelist=self.getnamelist()
    self.build_opt_dict_variations()
    
        
def rebuild_current_namelist():
    namelist=self.getnamelist()#get a new copy just in case
    now=strftime("%Y%m%d-%H%M%S")
    now_s=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")
    name_last_update_list=[(name,times[-1]) for name,times in namelist]
    name_s_since_update_list=[(name,now_s-datetime.datetime.strptime(time,"%Y%m%d-%H%M%S")) for name,time in name_last_update_list]
    current_name_list=[name_times_tup for i,name_times_tup in enumerate(namelist) if name_s_since_update_list[i]<self.old_threshold]
    try: 
        with open(self.savedirectory+'namelist','wb') as savednamelist:
            pickle.dump(current_name_list,savednamelist)
    except:
        assert False,"could not write current_name_list to disc"
    

def add_to_namelist(self,newname):
    try: 
        namelist=self.getnamelist()
    except: 
        namelist=[]
    
    namelist.append((newname,node_checkin_list))
    try: 
        with open(self.savedirectory+'namelist','wb') as savednamelist:
            assert len([1 for name in namelist if name[0]==newname])==1,"newname has more than 1 match"
            pickle.dump(savedname,savednamelist)
        with open(self.savedirectory+'namelist','rb') as savednamelist:
            namelist_check=pickle.load(savednamelist)
            assert len([1 for name in namelist_check if name[0]==newname])>1,"newname has too many matches!!!"
            
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
    mydir=self.savedirectory+myname
    
    try: 
        os.chdir(self.savedirectory)
    except:
        os.mkdir(self.savedirectory)
        os.chdir(self.savedirectory)
    try:
        os.chdir(mydir)
    except:
        print("mydir:{mydir} doesn't exist, so creating it")
        os.mkdir(mydir)
        os.chdir(mydir)
    start_time=strftime("%Y%m%d-%H%M%S")
    i_have_opt_job=0
    i=0
    while i_have_opt_job==0:
        my_opt_job=self.check_for_opt_job(myname,start_time,mydir)
        if try:type(my_opt_job) is dict:
                i_have_opt_job=1
        else:
            print(f'checking_for_opt_job repeated {i} times')
    if i_have_opt_job==1:
        data_gen_variation_list=my_opt_job['data_gen_variation_list']
        opt_model_variation_list=my_opt_job['opt_model_variation_list']
        my_job_save_file=my_opt_job['my_job_save_file']
        self.update_node_status(myname,working='starting'mydir=mydir,my_job_save_file=my_job_save_file)
        try:
            opt_results=kc.KernelCompare().run_kernel_list(opt_model_variation_list=opt_model_variation_list,data_gen_variation_list=data_gen_variation_list)
            #was_successful_list=[minimize_obj.success for minimize_obj in opt_result]
            self.update_node_status(myname,working=opt_results,mydir=mydir,my_job_save_file=my_job_save_file)
        except:
            self.update_node_status(myname,working='failed',mydir=mydir,my_job_save_file=my_job_save_file)
    
    
    
def check_for_opt_job(self,myname,start_time,mydir):
    assert type(myname) is str,f"myname should be type str not type:{type(myname)}"
    
    myjobcheck_file=mydir+'_jobcheck'
    waiting=0
    i=0
    while waiting==0:
        try: with open(myjobcheck_file,'rb') as myjob:
                if myjob['job_to_do']==1:
                    myjob_id=myjob['job_id']
                    my_job_save_file=mydir+myname+'myjob_id'
                    self.update_node_status(myname,working='accepted',mydir=mydir,my_job_save_file=my_job_save_file)
                    my_opt_job=pickle.load(myjob)
                    my_opt_job['my_job_save_file']=my_job_save_file
                    return my_opt_job
                else:
                    print('job_to_do:',myjob['job_to_do'])
                    waiting=1#need to develop
        except:
            i+=1
            if i%500==0:
                print("waiting...i=",i,end='. ')
            now=strftime("%Y%m%d-%H%M%S")
            s_since_start=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(start_time,"%Y%m%d-%H%M%S"))
            if s_since_start<60*10:
                self.update_node_status(myname,checktime=s_since_start,time_out=0,working=0)
                
            if s_since_start>60*10:#10 minutes
                self.update_node_status(myname,checktime=s_since_start,time_out=1,working=0)
                self.update_myname_in_namelist(myname)
                waiting=1
    return None

def update_myname_in_namelist(self, myname)
    namelist=self.getnamelist()
    myname_match_list=[i for i,name in enumerate(namelist) if name[0]==myname]
    assert len(myname_match_list)==1, f'somehow len(myname_match_list) is not 1 but len:{len(myname_match_list)}'
    myname_tup=namelist[i]
    now=strftime("%Y%m%d-%H%M%S")
    myname_tup[1].append(now)
    namelist[i]=myname_tup
    try:
        with open(self.savedirectory+'namelist','wb') as savednamelist:
            pickle.dump(namelist,savednamelist)
    except:
        assert False, 'update_of namelist_failed'
    
def update_node_status(self,myname,checktime=None,time_out=None,working=None,mydir=None,my_job_save_file=None):
    self.update_myname_in_namelist(myname)
    success=0
    now=strftime("%Y%m%d-%H%M%S")
    if type(working)==list:
        now=strftime("%Y%m%d-%H%M%S")
        job_save_dict['node_status'].append("finished",now)
        try:
            os.chdir(mydir)
        except: assert False, "chdir failed"
        try:
            with open(my_job_save_file,'rb') as job_save_file
                job_save_dict=pickle.load(job_save_file)
        except:
            assert False,"could not open my_job_save_file when updating node status to started"
        now=strftime("%Y%m%d-%H%M%S")
        job_save_dict['node_status'].append("finished",now)
        job_save_dict['optimize_obj_list']=working
        now=strftime("%Y%m%d-%H%M%S")
        job_save_dict['node_status'].append("waiting",now)
        success=1
    if working=='accepted':
        try:
            os.chdir(mydir)
        except: assert False, "chdir failed"
        try:
            with open(my_job_save_file,'rb') as job_save_file
                job_save_dict=pickle.load(job_save_file)
            print(f'my_job_save_file:{my_job_save_file} already exists.') 
            assert type(job_save_dict) is dict, f'job_save_dict type is {type(job_save_dict)}'
        except:
            job_save_dict={'node_status':[('created',now)]}
            print(f'my_job_save_file:{my_job_save_file}  does not exist')
        now=strftime("%Y%m%d-%H%M%S")
        job_save_dict['node_status'].append("accepted",now)
        success=1
    
    if working=='started':
        now=strftime("%Y%m%d-%H%M%S")
        try:
            os.chdir(mydir)
        except: assert False, "chdir failed"
        try:
            with open(my_job_save_file,'rb') as job_save_file
                job_save_dict=pickle.load(job_save_file)
        except:
            assert False,"could not open my_job_save_file when updating node status to started"
        job_save_dict['node_status'].append("started",now)
        success=1
    
    if working=='failed':
        now=strftime("%Y%m%d-%H%M%S")
        try:
            os.chdir(mydir)
        except: assert False, "chdir failed"
        try:
            with open(my_job_save_file,'rb') as job_save_file
                job_save_dict=pickle.load(job_save_file)
        except:
            assert False,"could not open my_job_save_file when updating node status to started"
        job_save_dict['node_status'].append("failed",now)
        success=1
    
    assert success==1,f"something went wrong when updating node. working:{working}"
    try:
        with open(my_job_save_file,'wb') as job_save_file
            pickle.dump(job_save_dict,job_save_file)
    except:
        assert False,'could not dump my_job_save_file'
            
    return
            

                    

        
                    
                