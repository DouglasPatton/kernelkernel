import pickle
import os
import re
from time import strftime,sleep
import datetime
import kernelcompare 
import traceback

'''to do:
master needs to maintain a modelrun status list to make sure models don't get left behind.

master and node can communicate with each other by adding a third item to namelist tupple
    or by having distinct update status options for master/node

master needs to maintain a node list and reassign/update finished/failed appropriately

add master record file and give restart option

let nodes decide if job is too big and refuse and get another

have mycluster search for and merge model_save files in parent and child directories to its own directory
'''

class run_cluster(kernelcompare.KernelCompare):
    '''
    There should be 1 master and 1 node. each node makes sure its assigned name is not already in the namelist
    appending a number for the number of matches+1. Then nodelist appends and posts back to namelist its name and a time,status tupple with
    status='created'(this process should keep namelist open, so other nodes can't do the exact same thing at the
    same time and both think they got the right name....
    When the master is started, it checks the namelist for anynodes that have not posted for awhile (need to add this next bit)
    and that are not working on a job. The master can also check the nodes model_save file
    '''
    def __init__(self,mytype=None,optdict_variation_list=None,data_gen_variation_list=None,local_test='yes'):
        
        if mytype==None:
            mytype='node'
        self.oldnode_threshold=datetime.timedelta(minutes=1,seconds=10)
        
        self.savedirectory=self.setdirectory(local_test=local_test)
        kernelcompare.KernelCompare.__init__(self,self.savedirectory)
        self.initialize(mytype,optdict_variation_list=optdict_variation_list,data_gen_variation_list=data_gen_variation_list)

        
    def setdirectory(self,local_test='yes'):
        if local_test=='No' or local_test=='no' or local_test==0:
            os.chdir('O:/Public/DPatton/kernel/')
        elif local_test=='yes' or local_test==None or local_test=='Yes' or local_test==1:
            try:
                os.chdir(cluster_test)
            except:
                os.mkdir(cluster_test)
                os.chdir(cluster_test)
        else: 
            assert False,f"local_test not understood. value:{local_test}"
        return os.getcwd()
        
        
        
        
        
    def initialize(self,mytype,optdict_variation_list=None,data_gen_variation_list=None):
        os.chdir(self.savedirectory)
        if mytype=="master":
            self.runmaster(optdict_variation_list,data_gen_variation_list)
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
                    name_regex2=r'^'+re.escape(mynametry)
                    namematch2=[1 for name_tup in namelist if re.search(name_regex2,name_tup[0])]
                    if len(namematch2)>0:
                        namecount+=1
                    if len(namematch2)==0:
                        myname=mynametry
                        break
                        nameset=1
            self.add_to_namelist(myname)
            self.runnode(myname)

    def runmaster(self,optdict_variation_list,data_gen_variation_list):
        try: 
            os.chdir(self.savedirectory)
        except:
            os.mkdir(self.savedirectory)
            os.chdir(self.savedirectory)

        
        list_of_run_dicts=self.prep_model_list(optdict_variation_list=optdict_variation_list,data_gen_variation_list=data_gen_variation_list)
        model_run_count=len(list_of_run_dicts)
        run_dict_status=['ready for node']*model_run_count
        
        i=0
        while all([status=='finished' for status in run_dict_status])==False:
            sleep(.95)
            
            self.rebuild_current_namelist()#get rid of the old names that are inactive
            namelist=self.getnamelist()
            assignment_tracker=[]
            for name in namelist:
                ready_dict_idx=[i for i in range(model_run_count) if run_dict_status[i]=='ready for node']
                
                #ready_dicts=[dict_i for i,dict_i in enumerate(list_of_run_dicts) if run_dict_status[i]=='ready for node']
                try:
                    job_time,job_status=self.check_node_job_status(name[0],time=1)
                    print(f"job_time:{job_time},job_status:{job_status}")
                    now=strftime("%Y%m%d-%H%M%S")
                    #elapsed=now-job_time
                    #late=elapsed>datetime.timedelta(seconds=30)
                    
                    if job_status=="no file found":# and (not late):
                        print(f'about to setup the job for node:{name}')
                        try:
                            first_ready_dict_idx=ready_dict_idx[0]
                            self.setup_job_for_node(name[0],list_of_run_dicts[first_ready_dict_idx])
                            i+=1
                            run_dict_status[first_ready_dict_idx]='assigned'
                            assignment_tracker.append((name[0],first_ready_dict_idx))
                        except:
                            print(traceback.format_exc())
                            print(f'setup_job_for_node named:{name[0]}, i:{i} has failed')
                    if job_status=='failed':
                        job_idx=[name_idx_tup[1] for name_idx_tup in assignment_tracker if name_idx_tup[0]==name[0]]
                        assignment_tracker=[name_idx_tup for name_idx_tup in assignment_tracker if not name_idx_tup[0]==name[0]]
                        try:
                            self.discard_job_for_node(name[0],list_of_run_dicts)
                            run_dict_status[job_idx]='ready for node'
                            self.update_myname_in_namelist(name[0],status='ready')
                        except:
                            print(traceback.format_exc())
                            print(f'node:{name[0]} has failed but could not free-up node and/or job_idx:{job_idx}')
                    if job_status=='finished':
                        job_idx=[name_idx_tup[1] for name_idx_tup in assignment_tracker if name_idx_tup[0]==name[0]]
                        assignment_tracker=[name_idx_tup for name_idx_tup in assignment_tracker if not name_idx_tup[0]==name[0]]
                        try:
                            self.discard_job_for_node(name[0],list_of_run_dicts)
                            run_dict_status[job_idx]='finished'
                            self.update_myname_in_namelist(name[0],status='ready')
                        except:
                            print(traceback.format_exc())
                            print(f'node:{name[0]} has finished but could not free-up node and/or job_idx:{job_idx}')
                        
                except:
                    print(traceback.format_exc())
                    
                    #print(Exception)
                    #print(f'status check for_node named:{name} has failed')
            sleep(3)
            self.find_failed_dicts

        assert i==model_run_count, f"i={i}but model_run_count={model_run_count}"
        print('all jobs finished')
        
    def discard_job_for_node(self,name):
        os.remove(self.savedirectory+name+'/'+name+'_job')
        return
    
    

    def setup_job_for_node(self,name,rundict,repeat=None):
        jobdict={}
        jobdict['optimizedict']=rundict['optimizedict']
        jobdict['datagen_dict']=rundict['datagen_dict']
        now=strftime("%Y%m%d-%H%M%S")
        jobdict['node_status']=[(now,'ready for node')]
        try:
            with open(self.savedirectory+name+'/'+name+'_job','wb') as newjob:
                pickle.dump(jobdict,newjob)
            print(f'job setup for node:{name}')
        except:
            if repeat==None:
                repeat=1
            elif repeat>10:
                assert False,"too many repeats"
            else:
                repeat+=1
                sleep(.05)
            self.setup_job_for_node(name,rundict,repeat=repeat)
        return


    def rebuild_current_namelist(self):
        os.chdir(self.savedirectory)
        namelist=self.getnamelist()#get a new copy just in case
        now=strftime("%Y%m%d-%H%M%S")
        now_s=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")
        name_last_update_list=[(name,times_status_tup_list[-1][0]) for name,times_status_tup_list in namelist]#times_status_tup_list[-1][0]:-1 for last item in list, and 0 for first item in time_status tuple
        s_since_update_list=[now_s-datetime.datetime.strptime(time,"%Y%m%d-%H%M%S") for _,time in name_last_update_list]
        current_name_list=[name_times_tup for i,name_times_tup in enumerate(namelist) if s_since_update_list[i]<self.oldnode_threshold]
        old_name_list=[name_times_tup for i,name_times_tup in enumerate(namelist) if not s_since_update_list[i]<self.oldnode_threshold]
        try:
            [os.rmdir(name[0]) for name in old_name_list]
        except Exception as e: print(e)
            
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
            sleep(5)

        if i_have_opt_job==1:
            my_optimizedict=my_opt_job['optimizedict']
            my_datagen_dict=my_opt_job['datagen_dict']


            self.update_node_job_status(myname,status='starting',mydir=mydir)
            try:
                kernelcompare.KernelCompare(directory=mydir).run_model_as_node(my_optimizedict,my_datagen_dict,force_start_params=0)
            except:
                print(traceback.format_exc())
            try:
                self.update_node_job_status(myname,status="finished",mydir=mydir)

            except:
                print(traceback.format_exc())
                self.update_node_job_status(myname,status='failed',mydir=mydir)



    def check_for_opt_job(self,myname,start_time,mydir):
        assert type(myname) is str,f"myname should be type str not type:{type(myname)}"
        os.chdir(mydir)
        my_job_file=mydir+myname+'_job'
        waiting=0
        i=0
        while waiting==0:
            sleep(1.5)
            try:
                with open(my_job_file,'rb') as myjob_save:
                    myjob=pickle.load(myjob_save)
                last_node_status=myjob['node_status'][-1][1]
                if last_node_status=='ready for node':
                    self.update_node_job_status(myname,status='accepted',mydir=mydir)
                    return myjob
                else:
                    print('myjob status:',myjob['node_status'])
                    waiting=1#need to develop
            except:
                print(traceback.format_exc())
                i+=1
                now=strftime("%Y%m%d-%H%M%S")
                s_since_start=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(start_time,"%Y%m%d-%H%M%S")
                if s_since_start%datetime.timedelta(seconds=10)==0:
                    print("s_since_start=",s_since_start,end='. ')
                    self.update_myname_in_namelist(myname,'waiting for job')#signal that this node is still active
                
                
                if s_since_start>self.oldnode_threshold:
                    print(s_since_start-self.oldnode_threshold)
                    waiting=1
        print(f'myname:{myname} timed out after finding no jobs')

        return None

    def update_myname_in_namelist(self, myname,status=None):
        namelist=self.getnamelist()
        print(f'namelist:{namelist},myname:{myname}')
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
        return


    def check_node_job_status(self,name,time=None):
        #returns the status of the job in the directory of a node
        if time==None or time=='no':
            time=0
        if time=='yes':
            time=1
        nodes_dir=self.savedirectory+name+'/'

        os.chdir(nodes_dir)


        nodes_job_filename=nodes_dir+name+'_job'
        try:
            with open(nodes_job_filename,'rb') as saved_job_file:
                nodesjob_dict=pickle.load(saved_job_file)
            print(f'check_node_job_status found: nodes_jobdict["status"]:{nodesjob_dict["node_status"]}')
            os.chdir(self.savedirectory)
            if time==0:
                return nodesjob_dict['node_status'][-1][1]
            if time==1:
                print(f"nodesjob_dict['node_status'][-1]:{nodesjob_dict['node_status'][-1]}")
                return nodesjob_dict['node_status'][-1][0],nodesjob_dict['node_status'][-1][1]#time_status tup
        except:
            print(traceback.format_exc())
            os.chdir(self.savedirectory)          
            if time==0:
                return "no file found"#if the file doesn't exist, then assign the job
            if time==1:
                
                return strftime("%Y%m%d-%H%M%S"), "no file found"


    def update_node_job_status(self,myname,status=None,mydir=None):
        self.update_myname_in_namelist(myname,status)

        my_job_file=mydir+myname+'_job'

        os.chdir(mydir)
        with open(my_job_file,'rb') as job_save_file:
            job_save_dict=pickle.load(job_save_file)

        if type(status) is str:
            now=strftime("%Y%m%d-%H%M%S")
            job_save_dict['node_status'].append((now,status))

        with open(my_job_file,'wb') as job_save_file:
            pickle.dump(job_save_dict,job_save_file)

        return

if __name__=="__main__":
    import kernelcompare as kc
    import mycluster
    data_gen_variation_list=[kc.KernelCompare().build_quadratic_datagen_dict_override()]
    Ndiff_type_variations=('modeldict:Ndiff_type',['recursive','product'])
    max_bw_Ndiff_variations=('modeldict:max_bw_Ndiff',[2])
    Ndiff_start_variations=('modeldict:Ndiff_start',[2])
    product_kern_norm_variations=('modeldict:product_kern_norm',['self','own_n'])#include None too?
    normalize_Ndiffwtsum_variations=('modeldict:normalize_Ndiffwtsum',['own_n','across'])
    optdict_variation_list=[Ndiff_type_variations,max_bw_Ndiff_variations,Ndiff_start_variations]
    mycluster.run_cluster(mytype='master',optdict_variation_list=optdict_variation_list)





