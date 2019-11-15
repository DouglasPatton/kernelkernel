import pickle
import os
import re
from time import strftime
import datetime
import kernelcompare 

class run_cluster(kernelcompare.KernelCompare):
    def __init__(self,mytype=None):
        if mytype==None:
            mytype='master'
        self.oldnode_threshold=datetime.timedelta(minutes=3)
        self.savedirectory='O:/Public/DPatton/kernel/'
        kernelcompare.KernelCompare.__init__(self,self.savedirectory)
        self.initialize(mytype)
        
        
    def initialize(self,mytype):
        os.chdir(self.savedirectory)
        if mytype=="master":
            self.runmaster()
        else:
            myname=mytype
            mytype="node"
            name_regex=r'^'+re.escape(myname)
            namelist=self.getnamelist()
            namematch=[1 for name_tup in namelist if re.search(name_regex,name_tup[0])]
            namecount=len(namematch)
            if namecount>0:
                nameset=0
                while nameset==0:
                    mynametry=myname+f'{namecount}'
                    name_regex2=r'^'+mynametry
                    namematch2=[1 for name_tup in namelist if re.search(name_regex2,name_tup[0])]
                    if len(namematch2)>0:
                        namecount+=1
                    if len(namematch2)==0:
                        myname=mynametry
                        print()
                        nameset=1
            self.add_to_namelist(myname)
            self.runnode(myname)

    def runmaster(self,opt_model_variation_list):
        try: 
            os.chdir(self.savedirectory)
        except:
            os.mkdir(self.savedirectory)
            os.chdir(self.savedirectory)

        self.rebuild_current_namelist()#get rid of the old names that are inactive
        namelist=self.getnamelist()
        list_of_opt_dicts=self.build_opt_dict_variations(opt_model_variation_list)
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
                            print(f'setup_job_for_node named:{name[0]}, opt_dict:{i} has failed')
                    '''elif status="finished" or status="waiting":
                        merge_directory=self.savedirectory+name[0]+'/'
                        self.merge_and_condense_saved_models(
                            merge_directory=merge_directory,
                            save_directory=self.savedirectory,
                            condense=None,
                            verbose=None)
                       '''#removed above merge call and will have node do it when finished. 

                        #print(f'status of node:{name} is:{status} not "no file found"')
                except:
                    print(f'status check for_node named:{name} has failed')

        assert i==model_run_count, f"i={i}but model_run_count={model_run_count}"


    def setup_job_for_node(self,name,optimizedict):
        jobdict={}
        jobdict['optimizedict']=optimizedict
        jobdict['node_status']='ready for node'
        with open(self.savedirectory++name+'/_job','wb') as newjob:
            pickled.dump(jobdict,newjob)
        print(f'job setup for node:{name}')
        return


    def rebuild_current_namelist():
        namelist=self.getnamelist()#get a new copy just in case
        now=strftime("%Y%m%d-%H%M%S")
        now_s=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")
        name_last_update_list=[(name,times_status_tup_list[-1][0]) for name,times_status_tup_list in namelist]#times_status_tup_list[-1][0]:-1 for last item in list, and 0 for first item in time_status tuple
        s_since_update_list=[now_s-datetime.datetime.strptime(time,"%Y%m%d-%H%M%S") for _,time in name_last_update_list]
        current_name_list=[name_times_tup for i,name_times_tup in enumerate(namelist) if s_since_update_list[i]<self.oldnode_threshold]
        try: 
            with open(self.savedirectory+'namelist','wb') as savednamelist:
                pickle.dump(current_name_list,savednamelist)
        except:
            assert False,"could not write current_name_list to disc"


    def add_to_namelist(self,newname):
        os.chdir(self.savedirectory)
        namelist=self.getnamelist()
        assert len([1 for name in namelist if name[0]==newname])==0,"newname has a match already!"
        now=strftime("%Y%m%d-%H%M%S")
        time_status_tup_list=[(now,"created")]
        namelist.append((newname,time_status_tup_list))
        with open(self.savedirectory+'namelist','wb') as savednamelist:
            pickle.dump(namelist,savednamelist)
        namelist_check=self.getnamelist()
        matches=len([1 for name in namelist_check if name[0]==newname])
        assert matches==1,f"newname has too many matches:{matches}!!!"


    def getnamelist(self):
        os.chdir(self.savedirectory)
        try: 
            with open('namelist','rb') as namelist:
                return pickle.load(namelist)
        except:
            print('getnamelist found no namelist')
            return []

    def runnode(self,myname):
        mydir=self.savedirectory+myname+'/'
        my_job_file=mydir+'_job'
        
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
        print(f'{myname} ischecking for jobs')
        while i_have_opt_job==0:
            my_opt_job=self.check_for_opt_job(myname,start_time,mydir)
            if type(my_opt_job) is dict:
                i_have_opt_job=1

        if i_have_opt_job==1:
            my_optimizedict=my_opt_job['optimizedict']


            self.update_node_job_status(myname,status='starting',mydir=mydir)
            try:
                KernelCompare(directory=mydir).run_model_as_node(my_optimizedict,force_start_params=0)
                #was_successful_list=[minimize_obj.success for minimize_obj in opt_result]
                self.update_node_job_status(myname,status="finished",mydir=mydir)

            except:
                self.update_node_job_status(myname,status='failed',mydir=mydir)



    def check_for_opt_job(self,myname,start_time,mydir):
        assert type(myname) is str,f"myname should be type str not type:{type(myname)}"
        os.chdir(mydir)
        my_job_file=mydir+'_job'
        waiting=0
        i=0
        while waiting==0:
            try:
                with open(my_job_save_file,'rb') as myjob_save:
                    myjob=pickled.load(myjob_save)
                last_node_status=myjob['node_status'][-1][1]
                if last_node_status=='ready for node':
                        self.update_node_job_status(myname,status='accepted',mydir=mydir)
                        return myjob
                else:
                    print('myjob status:',myjob['node_status'])
                    waiting=1#need to develop
            except:
                i+=1
                now=strftime("%Y%m%d-%H%M%S")
                s_since_start=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(start_time,"%Y%m%d-%H%M%S")
                if i%10000==0:
                    print("s_since_start=",s_since_start,end='. ')
                    self.update_myname_in_namelist(myname)#signal that this node is still active
                
                
                if s_since_start>self.oldnode_threshold:
                    print(s_since_start-self.oldnode_threshold)
                    waiting=1
        print(f'myname:{myname} timed out after finding no jobs')

        return None

    def update_myname_in_namelist(self, myname,status=None):
        namelist=self.getnamelist()
        myname_match_list=[i for i,name_tup in enumerate(namelist) if name_tup[0]==myname]
        assert len(myname_match_list)==1, f'somehow len(myname_match_list) is not 1 but len:{len(myname_match_list)}'
        i=myname_match_list[0]
        myname_tup=namelist[i]
        now=strftime("%Y%m%d-%H%M%S")
        time_status_tup=(now,status)
        myname_tup[1].append(time_status_tup)
        namelist[i]=myname_tup
        with open(self.savedirectory+'namelist','wb') as savednamelist:
            pickle.dump(namelist,savednamelist)



    def check_node_job_status(self,name):
        #returns the status of the job in the directory of a node
        nodes_dir=self.savedirectory+name+'/'

        os.chdir(nodes_dir)


        nodes_job_filename=nodes_dir+'_job'
        try:
            with open(nodes_job_filename,'rb') as save_job_file:
                nodesjob_dict=pickled.load(saved_job_file)
            print(f'check_node_job_status found: nodes_jobdict["status"]:{nodesjob_dict["node_status"]}')
            os.chdir(self.savedirectory)
            return nodesjob_dict['status']
        except:
            os.chdir(self.savedirectory)          
            return "no file found"#if the file doesn't exist, then assign the job

        #files=os.listdir():
        #jobfile=[filename in files if name+'_job'=filename]

    def update_node_job_status(self,myname,status=None,mydir=None):
        self.update_myname_in_namelist(myname,status)

        my_job_file=mydir+'_job'

        os.chdir(mydir)
        with open(my_job_file,'rb') as job_save_file:
            job_save_dict=pickle.load(job_save_file)

        if type(status) is str:
            now=strftime("%Y%m%d-%H%M%S")
            job_save_dict['node_status'].append(now,status)

        with open(my_job_save_file,'wb') as job_save_file:
            pickle.dump(job_save_dict,job_save_file)

        return






