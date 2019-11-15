import pickle
import os
import re
from time import strftime
import datetime
import kernelcompare 

class run_cluster(kernelcompare.KernelCompare):
    def __init__(self,mytype):
        self.oldnode_threshold=60*20#60 seconds times 20 times before master assumes name is old and removes from list
        self.savedirectory='O:/Public/DPatton/kernel/'
        kernelcompare.Kernelcompare.__init__(self,self.savedirectory)
        self.initialize(mytype)
        
        
def initialize(self,mytype):
    os.chdir(self.savedirectory)
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
    self.savedirectory+myname
    
    try: 
        os.chdir(self.savedirectory)
    except:
        os.mkdir(self.savedirectory)
        os.chdir(self.savedirectory)
    
    self.rebuild_current_namelist()
    namelist=self.getnamelist()
    list_of_opt_dicts=self.build_opt_dict_variations()
    model_run_count=len(list_of_opt_dicts)
    i=0
    nodecount=len(namelist)
    while i<model_run_count:
        for name in namelist:
            try:
                job_status=self.check_node_job_status(name[0])
                if status=="no file found":
                    try:
                        self.setup_job_for_node(name[0],list_of_opt_dicts[i])
                        i+=1
                    except:
                        print(f'setup_job_for_node named:{name}, opt_dict:{i} has failed')
                elif status="finished" or status="waiting":
                    
                    
                    #print(f'status of node:{name} is:{status} not "no file found"')
            except:
                print(f'status check for_node named:{name} has failed')
        
    assert i==model_run_count, f"i={i}but model_run_count={model_run_count}"
    
    
def setup_job_for_node(self,name,optimizedict)
    jobdict={}
    jobdict['optimizedict']=optimizedict
    jobdict['node_status']='ready for node'
    
    

            
        
        
    
        
def rebuild_current_namelist():
    namelist=self.getnamelist()#get a new copy just in case
    now=strftime("%Y%m%d-%H%M%S")
    now_s=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")
    name_last_update_list=[(name,times_status_tup_list[-1][0]) for name,times_status_tup_list in namelist]#times_status_tup_list[-1][0]:-1 for last item in list, and 0 for first item in time_status tuple
    name_s_since_update_list=[(name,now_s-datetime.datetime.strptime(time,"%Y%m%d-%H%M%S")) for name,time in name_last_update_list]
    current_name_list=
                  [name_times_tup for i,name_times_tup in enumerate(namelist) if name_s_since_update_list[i]<self.oldnode_threshold]
    try: 
        with open(self.savedirectory+'namelist','wb') as savednamelist:
            pickle.dump(current_name_list,savednamelist)
    except:
        assert False,"could not write current_name_list to disc"
    

def add_to_namelist(self,newname):
    os.chdir(self.savedirectory)
    try: 
        namelist=self.getnamelist()
    except: 
        print('namelist not found')
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
    my_job_file=mydir+'_job'
    my_node_status_file=mydir+'status'
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
        
    if i_have_opt_job==1:
        my_optimizedict=my_opt_job['optimizedict']
        
        
        self.update_node_status(myname,status='starting'mydir=mydir)
        try:
            opt_results=KernelCompare(directory=mydir).run_model_as_node(my_optimizedict,force_start_params=0)
            #was_successful_list=[minimize_obj.success for minimize_obj in opt_result]
            self.update_node_status(myname,status=opt_results,mydir=mydir)
        except:
            self.update_node_status(myname,status='failed',mydir=mydir)
    
    
    
def check_for_opt_job(self,myname,start_time,mydir):
    assert type(myname) is str,f"myname should be type str not type:{type(myname)}"
    os.chdir(mydir)
    my_node_status_file=mydir+'status'
    my_job_file=mydir+'_job'
    waiting=0
    i=0
    while waiting==0:
        try:
            with open(my_job_save_file,'rb') as myjob_save:
                myjob=pickled.load(myjob_save)
            last_node_status=myjob['node_status'][-1][0]
            if last_node_status=='ready for node':
                    self.update_node_status(myname,status='accepted',mydir=mydir)
                    return myjob
            else:
                print('myjob status:',myjob['node_status'])
                waiting=1#need to develop
        except:
            i+=1
            if i%100==0:
                print("waiting...i=",i,end='. ')
                self.update_myname_in_namelist(myname)#signal that this node is still active
            now=strftime("%Y%m%d-%H%M%S")
            s_since_start=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(start_time,"%Y%m%d-%H%M%S"))
            if s_since_start>self.oldnode_threshold
                waiting=1
        print(f'myname:{myname} timed out after finding no jobs')
                
    return None

def update_myname_in_namelist(self, myname,status=None)
    namelist=self.getnamelist()
    myname_match_list=[i for i,name in enumerate(namelist) if name[0]==myname]
    assert len(myname_match_list)==1, f'somehow len(myname_match_list) is not 1 but len:{len(myname_match_list)}'
    i=myname_match_list[0]
    myname_tup=namelist[i][1]
    now=strftime("%Y%m%d-%H%M%S")
    time_status_tup=(now,status)
    myname_tup[1].append(time_status_tup)
    namelist[i]=myname_tup
    try:
        with open(self.savedirectory+'namelist','wb') as savednamelist:
            pickle.dump(namelist,savednamelist)
    except:
        assert False, 'update_of namelist_failed'
        
        
def check_node_job_status(self,name):
    #returns the status of the job in the directory of a node
    nodes_dir=self.savedirectory+name
    
    os.chdir(nodes_dir)
    
        
    nodes_job_filename=nodes_dir+'_job'
    try:
        with: open(nodes_job_filename,'rb') as save_job_file:
                nodesjob_dict=pickled.load(saved_job_file)
            print(f'check_node_job_status found: nodes_jobdict["status"]:{nodesjob_dict["node_status"]}'
        os.chdir(self.savedirectory)
        return nodesjob_dict['status']
    except:
        os.chdir(self.savedirectory)          
        return "no file found"#if the file doesn't exist, then assign the job
        
    #files=os.listdir():
    #jobfile=[filename in files if name+'_job'=filename]
        
def update_node_job_status(self,myname,status=None,mydir=None):
    self.update_myname_in_namelist(myname)
    success=0
    now=strftime("%Y%m%d-%H%M%S")
    
    
    my_node_status_file=mydir+'status'
    my_job_file=mydir+'_job'
    with open(my_job_file,'rb') as job_save_file
        job_save_dict=pickle.load(job_save_file)
    with open(my_node_status_file) as node_status_file
        node_status_list=pickle.load(node_status_file)
    
    
    if type(status)==list:
        now=strftime("%Y%m%d-%H%M%S")
        node_status_list.append("finished",now)
        
       
        
        
        
        now=strftime("%Y%m%d-%H%M%S")
        job_save_dict['node_status'].append("finished",now)
        job_save_dict['optimize_obj_list']=status
        now=strftime("%Y%m%d-%H%M%S")
        job_save_dict['node_status'].append("waiting",now)
        success=1
    if status=='accepted':
                
        try:
            with open(my_job_save_file,'rb') as job_save_file
                job_save_dict=pickle.load(job_save_file)
            print(f'my_job_save_file:{my_job_save_file} already exists.') 
            assert type(job_save_dict) is dict, f'job_save_dict type is {type(job_save_dict)}'
        except:
            job_save_dict={'node_status':[('created',now)]}
            print(f'my_job_save_file:{my_job_save_file}  does not exist')
        now=strftime("%Y%m%d-%H%M%S")
        node_status_list.append("accepted",now)
        success=1
    
    if status=='started':
        now=strftime("%Y%m%d-%H%M%S")
        try:
            os.chdir(mydir)
        except: assert False, "chdir failed"
        try:
            with open(my_job_save_file,'rb') as job_save_file
                job_save_dict=pickle.load(job_save_file)
        except:
            assert False,"could not open my_job_save_file when updating node_status to started"
        node_status_list.append("started",now)
        success=1
    
    if status=='failed':
        now=strftime("%Y%m%d-%H%M%S")
        try:
            os.chdir(mydir)
        except: assert False, "chdir failed"
        try:
            with open(my_job_save_file,'rb') as job_save_file
                job_save_dict=pickle.load(job_save_file)
        except:
            assert False,"could not open my_job_save_file when updating node_status to failed"
        node_status_list.append("failed",now)
        success=1
    
    assert success==1,f"something went wrong when updating node. status:{status}"
    try:
        with open(my_job_save_file,'wb') as job_save_file
            pickle.dump(job_save_dict,job_save_file)
    except:
        assert False,'could not dump my_job_save_file'
            
    return
            

                    

        
                    
                