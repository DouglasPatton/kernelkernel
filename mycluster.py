import pickle
import os
import re
from time import strftime,sleep
import datetime
import kernelcompare 
import traceback
import shutil
from numpy import log
from random import randint

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
    status='ready for job'(this process should keep namelist open, so other nodes can't do the exact same thing at the
    same time and both think they got the right name....
    When the master is started, it checks the namelist for anynodes that have not posted for awhile (need to add this next bit)
    and that are not working on a job. The master can also check the nodes model_save file
    '''
    def __init__(self,myname=None,optdict_variation_list=None,datagen_variation_list=None,local_test='yes'):
        
        if myname==None:
            myname='node'
        if optdict_variation_list==None:
            optdict_variation_list=self.getoptdictvariations()
        if datagen_variation_list==None:
            datagen_variation_list=self.getdatagenvariations()

        self.oldnode_threshold=datetime.timedelta(minutes=10,seconds=1)
        self.savedirectory=self.setdirectory(local_test=local_test)
        self.masterdirectory=self.setmasterdir(self.savedirectory)
        self.masterfilefilename=os.path.join(self.masterdirectory, 'masterfile')

        print(f'self.savedirectory{self.savedirectory}')
        kernelcompare.KernelCompare.__init__(self,self.savedirectory)
        self.initialize(myname,optdict_variation_list=optdict_variation_list,datagen_variation_list=datagen_variation_list)




    def getoptdictvariations(self):
        Ndiff_type_variations = ('modeldict:Ndiff_type', ['recursive', 'product'])
        max_bw_Ndiff_variations = ('modeldict:max_bw_Ndiff', [2])
        Ndiff_start_variations = ('modeldict:Ndiff_start', [1, 2])
        #product_kern_norm_variations = ('modeldict:product_kern_norm', ['self', 'own_n'])
        #normalize_Ndiffwtsum_variations = ('modeldict:normalize_Ndiffwtsum', ['own_n', 'across'])
        # ykern_grid_variations=('ykern_grid',[31,46,61])
        optdict_variation_list = [Ndiff_type_variations, max_bw_Ndiff_variations, Ndiff_start_variations]

        return optdict_variation_list

    def getdatagenvariations(self):
        train_n_variations = ('train_n', [30, 45, 60])
        ftype_variations = ('ftype', ['linear', 'quadratic'])
        param_count_variations = ('param_count', [1, 2])
        datagen_variation_list = [train_n_variations, ftype_variations, param_count_variations]
        return datagen_variation_list

    def setmasterdir(self,savedirectory):
        masterdir=os.path.join(savedirectory,'master')
        if not os.path.exists(masterdir):
            for i in range(10):
                try:
                    os.mkdir(masterdir)
                    break
                except:
                    if i==9:
                        print(traceback.format_exc())
        return masterdir
        
    def setdirectory(self,local_test='yes'):
        if local_test=='No' or local_test=='no' or local_test==0:
            savedirectory='O:/Public/DPatton/kernel/'
        elif local_test=='yes' or local_test==None or local_test=='Yes' or local_test==1:
            savedirectory=os.path.join(os.getcwd(),'cluster_test')
            if not os.path.exists(savedirectory):
                for i in range(10):
                    try:
                        os.mkdir(savedirectory)
                        os.chdir(savedirectory)
                        break
                    except:
                        if i==9:
                            print(traceback.format_exc())

        else: 
            assert False,f"local_test not understood. value:{local_test}"
        return savedirectory

        
    def initialize(self,myname,optdict_variation_list=None,datagen_variation_list=None):
        os.chdir(self.savedirectory)
        if myname=="master":
            self.runmaster(optdict_variation_list,datagen_variation_list)
        else:
            myname=self.createnamefile(myname)
            self.nodedirectory = os.path.join(self.savedirectory, myname)
            self.nodenamefilename=os.path.join(self.masterdirectory,myname+'.name')
            self.nodejobfilename=os.path.join(self.nodedirectory,myname+'_job')
            self.runnode(myname)


    def createnamefile(self,name):
        namefilename=os.path.join(self.masterdirectory,name+'.name')
        if os.path.exists(namefilename):
            oldname=name
            nameset=0
            while nameset==0:
                name=oldname+str(randint(0,9))
                namefilename = os.path.join(self.masterdirectory, name+'.name')
                if not os.path.exists(namefilename):
                    break
            print(f' {oldname} taken; new name is {name}')

        now = strftime("%Y%m%d-%H%M%S")
        time_status_tup_list = [(now, 'created')]
        for i in range(10):
            try:
                with open(namefilename,'wb') as savednamefile:
                    pickle.dump(time_status_tup_list,savednamefile)
                break
            except:
                if i==9:
                    print(traceback.format_exc())
                    print(f'problem creating:{name}, restarting createnamefile')
                    name=self.createnamefile(name)
        return name




    '''def namenode(self,myname):

        name_regex=r'^'+re.escape(myname)
        namelist=self.getnamelist()
        namematch=[1 for name_tup in namelist if re.search(name_regex,name_tup[0])]
        namecount=len(namematch)
        if namecount>0:
            nameset=0
            while nameset==0:
                mynametry=myname+f'{namecount}'
                name_regex2=r'^'+re.escape(mynametry)
                namelist=self.getnamelist()
                namematch2=[1 for name_tup in namelist if name_tup[0]==mynametry]
                if len(namematch2)>0:
                    namecount+=1
                if len(namematch2)==0:
                    myname=mynametry
                    myname=self.add_to_namelist(myname)
                    namelist=self.getnamelist()
                    namematch3=[1 for name_tup in namelist if name_tup[0]==myname]
                    if len(namematch3)>1:
                        self.namenode(myname+str(1))
                    break

        self.runnode(myname)'''

            
    def getreadynames(self,namelist):
        readylist=[]
        for name_i in namelist:
            last_time_status_tup=self.namefile_statuscheck(name_i)
            if last_time_status_tup[1]=='ready for job':
                readylist.append(name_i)
        return readylist
    
    def checkmaster(self):
        return os.path.exists(self.masterfilefilename)
    
    def getmaster(self):
        for i in range(10):
            try:
                with open(self.masterfilefilename,'rb') as themasterfile:
                    masterfile=pickle.load(themasterfile)
                return masterfile
            except(FileNotFoundError):
                print(traceback.format_exc())
                return False
            except:
                if i==9:
                    sleep(.5)
                    assert False, "getmaster can't open masterfile"

     
    def archivemaster(self):
        masterfile=self.getmaster()
        for i in range(10):
            try:
                with open('masterfile_archive','wb') as savefile:
                    pickle.dump(masterfile,savefile)
                break
            except:
                if i==9:
                    print(traceback.format_exc())
                    print('arechivemaster has failed')
                    return
        for i in range(10): 
            try:
                os.remove(self.masterfilefilename)
                return
            except:
                if i==9:
                    print('masterfile_archive created, but removal of old file has failed')
                    print(traceback.format_exc())
                    return
                    
            
                
    
    def savemasterstatus(self,assignment_tracker,run_dict_status,list_of_run_dicts):
        savedict={'assignment_tracker':assignment_tracker,'run_dict_status':run_dict_status,'list_of_run_dicts':list_of_run_dicts}
        for i in range(10):
            try:
                with open(self.masterfilefilename,'wb') as themasterfile:
                    pickle.dump(savedict,themasterfile)
                break
            except:
                sleep(.2)
                if i==9:
                    print(traceback.format_exc())
                    assert False, 'masterfile problem'

    def rebuild_namefiles(self, run_dict_status, assignment_tracker):
        namelist = self.getnamelist()  # get a new copy just in case
        namefile_tuplist = [self.namefile_statuscheck(name) for name in namelist]
        # print(f'namefile_tuplist:{namefile_tuplist}')
        s_since_update_list = [self.s_before_now(time) for time, status in namefile_tuplist]

        current_name_list = [name for i, name in enumerate(namelist) if s_since_update_list[i] < self.oldnode_threshold]
        old_name_list1 = [name for i, name in enumerate(namelist) if
                          not s_since_update_list[i] < self.oldnode_threshold]

        old_name_list = []
        for name_i in old_name_list1:
            for j in range(10):
                try:
                    time_i = self.model_save_activitycheck(name_i)
                    print(f'time_i{time_i}')
                    if not type(time_i) is datetime.timedelta:
                        old_name_list.append(name_i)
                    elif time_i < self.oldnode_threshold:
                        current_name_list.append(name_i)
                    break
                except:
                    if j == 9:
                        print('timeout', traceback.format_exc())
                        old_name_list.append(name_i)

        if len(old_name_list) > 0:
            print(f'old_name_list:{old_name_list}')
        for j, name in enumerate(old_name_list):
            for i in range(10):
                try:
                    self.mergethisnode(name)
                    try:
                        os.remove(os.path.join(self.masterdirectory, name + '.name'))
                    except:
                        pass
                    try:
                        shutil.rmtree(os.path.join(self.savedirectory, name))
                    except:
                        pass
                    break
                except:
                    if i == 9:
                        print(f'failed to merge node named:{name}')
                        print(traceback.format_exc())
        if len(assignment_tracker) > 0:
            assigned_to_not_current_name_idx=[]
            #ignment_tracker',assignment_tracker)
            for name_i,idx in assignment_tracker.items():
                for name_j in current_name_list:
                    if name_i==name_j:
                        assigned_to_not_current_name_idx.append(idx)
        else:
            assigned_to_not_current_name_idx=[idx for name_i,idx in assignment_tracker.items()]
        the_not_current_names=[name_i for name_i,idx in assignment_tracker.items() if not any([name_j==name_i for name_j in current_name_list])]
        for idx in assigned_to_not_current_name_idx:
            run_dict_status[idx]='ready for node'
        for name_i in the_not_current_names:
            try:del assignment_tracker[name_i]
            except:pass
        '''print('assignment_tracker',assignment_tracker)
        print('current_name_list',current_name_list)
        print('the_not_current_names',the_not_current_names)'''
        #assigned_names = [name_i for name_i in current_name_list if name_i in assignment_tracker]
        assigned_names_idx = [assignment_tracker[name_i] for name_i in current_name_list if
                              name_i in assignment_tracker]
        status_assigned_idx = [i for i, status in enumerate(run_dict_status) if status == 'assigned']

        release_status_idx = [idx for idx in status_assigned_idx if not any([idx == j for j in assigned_names_idx])]
        for idx in release_status_idx:
            run_dict_status[idx]='ready for node'
        return run_dict_status, assignment_tracker


        

    def runmaster(self,optdict_variation_list,datagen_variation_list):
        if self.checkmaster(): 
            masterfile=self.getmaster()
        try: 
            assignment_tracker=masterfile['assignment_tracker']
            list_of_run_dicts=masterfile['list_of_run_dicts']
            run_dict_status=masterfile['run_dict_status']
            model_run_count=len(list_of_run_dicts)
        except:
            assignment_tracker={}
            list_of_run_dicts=self.prep_model_list(optdict_variation_list=optdict_variation_list,datagen_variation_list=datagen_variation_list)
            model_run_count=len(list_of_run_dicts)
            run_dict_status=['ready for node']*model_run_count
        os.chdir(self.masterdirectory)

        i=0
        while all([status=='finished' for status in run_dict_status])==False:
            self.savemasterstatus(assignment_tracker,run_dict_status,list_of_run_dicts)

            run_dict_status, assignment_tracker=self.rebuild_namefiles(run_dict_status, assignment_tracker)#get rid of the old names that are inactive
            namelist=self.getnamelist()
            readynamelist=self.getreadynames(namelist)

            if len(readynamelist)>1:
                print(f'readynamelist:{readynamelist}')
            
            for name in readynamelist:
                ready_dict_idx=[i for i in range(model_run_count) if run_dict_status[i]=='ready for node']
                
                #ready_dicts=[dict_i for i,dict_i in enumerate(list_of_run_dicts) if run_dict_status[i]=='ready for node']

                try:
                    job_time,job_status=self.check_node_job_status(name,time=1)
                except:
                    print(f'check_node_job_status failed for node:{name}')
                    break

                #print(f"job_time:{job_time},job_status:{job_status}")
                now=strftime("%Y%m%d-%H%M%S")
                #elapsed=now-job_time
                #late=elapsed>datetime.timedelta(seconds=30)

                if job_status=="no file found":
                    print(f'about to setup the job for node:{name}')
                    #print('len(ready_dict_idx)',len(ready_dict_idx))
                    if len(ready_dict_idx) == 0:
                        print('run_dict_status',run_dict_status)
                    if len(ready_dict_idx)>0:
                        random_ready_dict_idx=ready_dict_idx[randint(0,len(ready_dict_idx))]
                        try:
                            run_dict_status[random_ready_dict_idx] = 'assigned'
                            ready_dict_idx = [i for i in range(model_run_count) if run_dict_status[i] == 'ready for node']
                            self.setup_job_for_node(name,list_of_run_dicts[random_ready_dict_idx])
                            assignment_tracker[name] = random_ready_dict_idx
                            #print('assignment_tracker', assignment_tracker)
                            i+=1
                        except:
                            run_dict_status[random_ready_dict_idx] = 'ready for node'
                            ready_dict_idx = [i for i in range(model_run_count) if run_dict_status[i] == 'ready for node']

                            print(traceback.format_exc())
                            print(f'setup_job_for_node named:{name}, i:{i} has failed')



                if job_status=='failed':
                    job_idx=assignment_tracker[name]
                    self.discard_job_for_node(name)
                    print(f'deleting assignmen_tracker for key:{name} with job_status:{job_status}')
                    del assignment_tracker[name]
                    run_dict_status[job_idx]='ready for node'
                    ready_dict_idx=[i for i in range(model_run_count) if run_dict_status[i]=='ready for node']
                    self.update_my_namefile(name,status='ready for job')
                    self.mergethisnode(name)
                if job_status=='finished':
                    print(f'node:{name} has finished')
                    try:job_idx=assignment_tracker[name]
                    except:
                        print(f'assignment_tracker failed for key:{name}, job_status:{job_status} ')
                        print(f'assignment_tracker:{assignment_tracker}')
                    self.discard_job_for_node(name)
                    print(f'deleting assignment_tracker for key:{name} with job_status:{job_status}')
                    del assignment_tracker[name]
                    run_dict_status[job_idx]='finished'
                    self.update_my_namefile(name,status='ready for job')
                    self.mergethisnode(name)

            sleep(5)

            

        #assert i==model_run_count, f"i={i}but model_run_count={model_run_count}"
        self.savemasterstatus(assignment_tracker,run_dict_status,list_of_run_dicts)
        self.archivemaster()
        print('all jobs finished')
        return
    

    
    def mergethisnode(self,name):
        nodesdir=os.path.join(self.savedirectory,name)
        for i in range(10):
            try:
                self.merge_and_condense_saved_models(merge_directory=nodesdir,save_directory=self.savedirectory,condense=1,verbose=0)
                break
            except:
                if i==9:
                    try:
                        self.merge_and_condense_saved_models(merge_directory=nodesdir,save_directory=self.savedirectory,condense=1,verbose=1)
                    except:
                        print(f'merge this node failed for node named:{name}')
                        print(traceback.format_exc())
                
                
        
    def discard_job_for_node(self,name):
        for i in range(10):
            try:
                os.remove(os.path.join(self.savedirectory,name,name+'_job'))
                break
            except:
                if i==9:
                    print(traceback.format_exc())
        return
    
    

    def setup_job_for_node(self,name,rundict):
        jobdict={}
        jobdict['optimizedict']=rundict['optimizedict']
        jobdict['datagen_dict']=rundict['datagen_dict']
        now=strftime("%Y%m%d-%H%M%S")
        jobdict['node_status']=[(now,'ready for node')]
        for _ in range(10):
            try:
                with open(os.path.join(self.savedirectory,name,name+'_job'),'wb') as newjob:
                    pickle.dump(jobdict,newjob)
                print(f'job setup for node:{name}')
                print(f'newjob has jobdict:{jobdict}')
                break
            except:
                print(traceback.format_exc())
                sleep(0.35)
                
        return

    def namefile_statuscheck(self,name):
        nodesnamefilefilename=os.path.join(self.masterdirectory,name+'.name')
        for i in range(10):
            try:
                with open(nodesnamefilefilename,'rb') as savednamefile:
                    namefile=pickle.load(savednamefile)
                break
            except:
                sleep(0.25)
                if i==9:
                    print(traceback.format_exc())
                    namefile=[(None,None)]
        return namefile[-1]


    def s_before_now(self,then):
        now=strftime("%Y%m%d-%H%M%S")
        now_s=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")
        return now_s-datetime.datetime.strptime(then,"%Y%m%d-%H%M%S")                   
                        
    
    def model_save_activitycheck(self,name,filename=None):
        if filename==None: filename='model_save'
        nodedir=os.path.join(self.savedirectory,name)
        node_job=os.path.join(nodedir,name+'_job')
        node_model_save=os.path.join(self.savedirectory,name,filename)
        #print(node_model_save)
        for i in range(10):
            try:
                with open(node_model_save,'rb') as saved_model_save:
                    model_save=pickle.load(saved_model_save)
                return model_save[-1]['when_saved']
            except:
                if i==9:
                    print(traceback.format_exc())
                    if not filename=='final_model_save':
                        self.model_save_activitycheck(name,filename='final_model_save')
        return None


    '''def add_to_namelist(self,newname):
        os.chdir(self.savedirectory)
        namelist=self.getnamelist()
        if len([1 for name in namelist if name[0]==newname])>0:
            newname=newname+str(randint(0,9))
        now=strftime("%Y%m%d-%H%M%S")
        time_status_tup_list=[(now,'ready for job')]
        namelist.append((newname,time_status_tup_list))
        for _ in range(20):
            try:
                with open(os.path.join(self.savedirectory,'namelist'),'wb') as savednamelist:
                    pickle.dump(namelist,savednamelist)
                break
            except:
                print(traceback.format_exc())
                sleep(0.15)
            
        namelist_check=self.getnamelist()
        matches=len([1 for name in namelist_check if name[0]==newname])
        if not matches==1:
            self.add_to_namelist(newname+str(randint(0,9)))
        return newname'''


    def getnamelist(self):
        
        for i in range(20):
            try:
                name_regex=r'\.'+re.escape('name')+r'$'#matches anything ending with .name
                allfiles=os.listdir(self.masterdirectory)
                namefilelist=[name for name in allfiles if re.search(name_regex,name)]
                namelist=[name[:-5] for name in namefilelist]
                return namelist
            except:
                if i==19:
                    print('getnamelist failed')
                    print(traceback.format_exc())
                sleep(0.25)
            return []


    def runnode(self,myname):
        for i in range(10):
            try:
                masterfile_exists=self.checkmaster()
                break
            except:
                if i==9:
                    print(traceback.format_exc())
                    assert False,f"runnode named {myname} could not check master"
        if masterfile_exists==False:
            for i in range(120):
                sleep(30)
                try:
                    masterfile_exists = self.checkmaster()
                    if masterfile_exists:break
                except:
                    if i == 119:
                        print(traceback.format_exc())
                        assert False, f"runnode named {myname} could not check master"

        #mydir=os.path.join(self.savedirectory,myname)
        #my_job_file=os.path.join(mydir,myname+'_job')
        
        mydir=self.nodedirectory
        if not os.path.exists(mydir):
            os.mkdir(mydir)
        self.update_my_namefile(myname,status='ready for job')
        start_time=strftime("%Y%m%d-%H%M%S")
        i_have_opt_job=0
        i=0
        print(f'{myname} ischecking for jobs')
        while i_have_opt_job==0:
            my_opt_job=self.check_for_opt_job(myname,start_time,mydir)
            if type(my_opt_job) is dict:
                break
            else:
                sleep(5)


        my_optimizedict=my_opt_job['optimizedict']
        my_datagen_dict=my_opt_job['datagen_dict']


        self.update_node_job_status(myname,status='starting',mydir=mydir)
        try:
            kernelcompare.KernelCompare(directory=mydir).run_model_as_node(my_optimizedict,my_datagen_dict,force_start_params=0)
            success=1
        except:
            success=0
            try:
                self.update_node_job_status(myname,status='failed',mydir=mydir)
            except:
                print(traceback.format_exc())
                #self.runnode(myname+'0')#relying on wrapper function restarting the node
                return
            print(traceback.format_exc())

        if success==1:
            try:
                self.update_node_job_status(myname,status="finished",mydir=mydir)
            except:
                print(f'could not update node status for {myname} to finished')
                print(traceback.format_exc())
                #self.runnode(myname+'0')#relying on wrapper function restarting the node
                return

        #self.runnode(myname)#relying on wrapper function restarting the node
        return



    def check_for_opt_job(self,myname,start_time,mydir):
        assert type(myname) is str,f"myname should be type str not type:{type(myname)}"
        os.chdir(mydir)
        my_job_file=os.path.join(mydir,myname+'_job')
        waiting=0
        i=0
        while waiting==0:
            
            try:
                with open(my_job_file,'rb') as myjob_save:
                    myjob=pickle.load(myjob_save)
                last_node_status=myjob['node_status'][-1][1]
                if last_node_status=='ready for node':
                    self.update_node_job_status(myname,status='accepted',mydir=mydir)
                    return myjob
                else:
                    #print('myjob status:',myjob['node_status'])
                    i+=1
                    sleep(.25*log(10*i+1))
            except:

                i+=1
                now=strftime("%Y%m%d-%H%M%S")
                s_since_start=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(start_time,"%Y%m%d-%H%M%S")
                if s_since_start>datetime.timedelta(hours=120):#allowing 2 hours instead of using self.oldnode_threshold
                    print(f'node:{myname} checking for job s_since_start="{s_since_start}')
                    self.update_my_namefile(myname,'ready for node')#signal that this node is still active
                    print(traceback.format_exc())
                
                
                if s_since_start>self.oldnode_threshold:
                    print(s_since_start-self.oldnode_threshold)
                    assert False,f'myname:{myname} timed out after finding no jobs'#relying on wrapper to restart the node
                sleep(.25*log(10*i+1))


        return None

    def update_my_namefile(self, myname,status=None):
        ''' namelist=self.getnamelist()
        print(f'namelist:{namelist},myname:{myname}')
        myname_match_list=[i for i,name_tup in enumerate(namelist) if name_tup[0]==myname]
        assert len(myname_match_list)==1, f'somehow len(myname_match_list) is not 1 but len:{len(myname_match_list)}'
        i=myname_match_list[0]
        myname_tup=namelist[i]'''
        now=strftime("%Y%m%d-%H%M%S")
        time_status_tup=(now,status)
        namefilename=os.path.join(self.masterdirectory,myname+'.name')
        for i in range(10):
            try:
                with open(namefilename,'rb') as namefile:
                    listoftups=pickle.load(namefile)
                    listoftups.append(time_status_tup)
                with open(namefilename,'wb') as namefile:
                    pickle.dump(listoftups,namefile)
                break
            except:
                if i==9:
                    print(traceback.format_exc())
                sleep(.15)

        return


    def check_node_job_status(self,name,time=None):
        #returns the status of the job in the directory of a node
        if time==None or time=='no':
            time=0
        if time=='yes':
            time=1
        nodes_dir=os.path.join(self.savedirectory,name)
        nodes_job_filename=os.path.join(nodes_dir,name+'_job')
        for i in range(10):
            try:
                with open(nodes_job_filename,'rb') as saved_job_file:
                    nodesjob_dict=pickle.load(saved_job_file)
                #print(f'check_node_job_status found: nodes_jobdict["status"]:{nodesjob_dict["node_status"]}')
                if time==0:
                    return nodesjob_dict['node_status'][-1][1]
                if time==1:
                    #print(f"nodesjob_dict['node_status'][-1]:{nodesjob_dict['node_status'][-1]}")
                    return nodesjob_dict['node_status'][-1][0],nodesjob_dict['node_status'][-1][1]#time_status tup
            except(FileNotFoundError):
                if i==9:

                    #print(traceback.format_exc())
                    if time==0:
                        return "no file found"#if the file doesn't exist, then assign the job
                    if time==1:
                        return strftime("%Y%m%d-%H%M%S"), "no file found"
            sleep(1)
                    
            

    def update_node_job_status(self,myname,status=None,mydir=None):
        self.update_my_namefile(myname,status)
        if mydir==None:
            mydir=os.path.join(self.savedirectory,myname)
        my_job_file=os.path.join(mydir,myname+'_job')

        for i in range(10):
            try:
                with open(my_job_file,'rb') as job_save_file:
                    job_save_dict=pickle.load(job_save_file)
                break
            except:
                sleep(.25)
                if i==9:
                    print(traceback.format_exc())
        if type(status) is str:
            now=strftime("%Y%m%d-%H%M%S")
            job_save_dict['node_status'].append((now,status))
        for _ in range(10):
            try:
                with open(my_job_file,'wb') as job_save_file:
                    pickle.dump(job_save_dict,job_save_file)
                break
            except:pass

        return

if __name__=="__main__":
    import kernelcompare as kc
    import mycluster
    
    Ndiff_type_variations=('modeldict:Ndiff_type',['recursive','product'])
    max_bw_Ndiff_variations=('modeldict:max_bw_Ndiff',[2,3])
    Ndiff_start_variations=('modeldict:Ndiff_start',[1,2])
    product_kern_norm_variations=('modeldict:product_kern_norm',['self','own_n'])
    normalize_Ndiffwtsum_variations=('modeldict:normalize_Ndiffwtsum',['own_n','across'])
    optdict_variation_list=[Ndiff_type_variations,max_bw_Ndiff_variations,Ndiff_start_variations,product_kern_norm_variations,normalize_Ndiffwtsum_variations]
    
    train_n_variations=('train_n',[30,45,60])
    ykern_grid_variations=('ykern_grid',[31,46,61])
    ftype_variations=('ftype',['linear','quadratic'])
    param_count_variations=('param_count',[1,2])
    datagen_variation_list=[train_n_variations,ftype_variations,param_count_variations]
    
    
    mycluster.run_cluster(myname='master',optdict_variation_list=optdict_variation_list,datagen_variation_list=datagen_variation_list)





