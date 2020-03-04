import pickle
import os
import re
from time import strftime,sleep
import datetime
import kernelcompare 
import traceback
import shutil
from numpy import log
from random import randint,seed,shuffle
import logging
import logging.config
import yaml
from helpers import Helper
import multiprocessing as mp

#from pisces_data_huc12 import PiscesDataTool

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
    
    def __init__(self,source=None,myname=None,optdict_variation_list=None,datagen_variation_list=None,local_run=None):
        seed(1)
        if source==None:
            source='monte'
        self.source=source
        self.oldnodequeue=mp.Queue
        self.savedirectory=self.setdirectory(local_run=local_run)
        self.model_collection_directorylist=[os.path.join(os.getcwd(),'..','model_collection')]
        if not os.path.exists(self.model_collection_directorylist[-1]): os.mkdir(self.model_collection_directorylist[-1])
        
        logdir=os.path.join(os.getcwd(),'log')
        #logdir=os.path.join(self.savedirectory,'log')

        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'mycluster_{myname}.log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=1)],
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)

        '''if not myname=='master':
            self.Ndiff_list_of_masks_x=None
            self.Ndiff_list_of_masks_y=None'''
        kernelcompare.KernelCompare.__init__(self,directory=self.savedirectory,source=source,myname=myname)
        
        self.masterdirectory, self.masterfiledirectory=self.setmasterdir(self.savedirectory)
        self.jobdirectory=os.path.join(self.savedirectory,'jobs')
        
        self.oldnode_threshold=datetime.timedelta(minutes=65,seconds=1)
        self.masterfilefilename=os.path.join(self.masterfiledirectory, 'masterfile')
        
        if myname is None:
            myname='node'
        
        

        
        print(f'self.savedirectory{self.savedirectory}')
        
        if not myname=='master':
            self.initialize(myname)
        
        
        self.datagen_dict=self.setdata(source)#creates the initial datagen_dict
        
        
        if local_run==None or local_run=='yes' or local_run=='Yes':
            local_run=1
        if local_run=='no' or local_run=='No':
            local_run=0

        #self.n=32 #must be even if ykerngrid is 1 higher and ykerngrid_form:exp is used

                    
        
        if optdict_variation_list==None:
            
            optdict_variation_list=self.getoptdictvariations(source=source)
        if datagen_variation_list==None:
            datagen_variation_list=self.getdatagenvariations(source=source)
        


        #print(f'datagen_variation_list:{datagen_variation_list}')
        self.initialize(
            myname,optdict_variation_list=optdict_variation_list,datagen_variation_list=datagen_variation_list)

    
    
    
    def setmasterdir(self,savedirectory):
        masterdir=os.path.join(savedirectory,'master')
        masterfiledir=os.path.join(os.getcwd(),'master')
        if not os.path.exists(masterdir): os.mkdir(masterdir)
        if not os.path.exists(masterfiledir): os.mkdir(masterfiledir)
            
                
        return masterdir,masterfiledir
        
    def setdirectory(self,local_run='yes'):
        if local_run==0:
            savedirectory='O:/Public/DPatton/kernel/'
        elif local_run==1:
            print('------------local_run:',local_run)
            savedirectory=os.path.join(os.getcwd(),'cluster_test')
        else: 
            assert False,f"local_run not understood. value:{local_run}"
        if not os.path.exists(savedirectory):
                for i in range(10):
                    if not os.path.exists(savedirectory):
                        try:
                            os.mkdir(savedirectory)
                            break
                        except:
                            if i==9:
                                self.logger.exception(f'error in {__name__} attempting mkdir({savedirectory})')
        return savedirectory

        
    def initialize(self,myname,optdict_variation_list=None,datagen_variation_list=None):
        
        if myname=="master":
            try:
                self.runmaster(optdict_variation_list,datagen_variation_list)
            except:
                self.logger.exception('master failed')
        else:
            myname=self.createnamefile(myname)
            self.nodedirectory = os.path.join(self.savedirectory, myname)
            self.nodenamefilename=os.path.join(self.masterdirectory,myname+'.name')
            self.nodejobfilename=os.path.join(self.nodedirectory,myname+'_job')
            try:
                self.runnode(myname)
            except:
                self.logger.exception('node with myname:{myname} has failed')


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
                    self.logger.exception(f'error in {__name__}')
                    print(f'problem creating:{name}, restarting createnamefile')
                    name=self.createnamefile(name)
        return name

            
    def getreadynames(self,namelist):
        readylist=[];sortlist=[]
        for name_i in namelist:
            last_time_status_tup=self.namefile_statuscheck(name_i)
            if last_time_status_tup[1]=='ready for job':
                readylist.append(name_i)
                #sortlist.append(last_time_status_tup[0])
        shuffle(readylist)
        #sorted_readylist=[name_i for _, name_i in sorted(zip(sortlist,readylist), key=lambda pair: pair[0])]
        return readylist
    
    def checkmaster(self):
        return os.path.exists(self.masterfilefilename)
    
    def getmaster(self):
        for i in range(3):
            try:
                with open(self.masterfilefilename,'rb') as themasterfile:
                    masterfile=pickle.load(themasterfile)
                return masterfile
            except FileNotFoundError:
                self.logger.exception(f'error in {__name__}')
                return False
            except EOFError:
                self.logger.exception('masterfile is corrupt')
                try:
                    with open(self.masterfilefilename+'_backup','rb') as themasterfile:
                        masterfile=pickle.load(themasterfile)
                    return masterfile
                except:
                    self.logger.exception('could not open masterfile backup either')
                
                
                masterfile={}
                return masterfile
            except:
                if i==2:
                    sleep(.5)
                    assert False, "getmaster can't open masterfile"
        return

    
    def archivemaster(self):
        masterfile=self.getmaster()
        for i in range(3):
            try:
                with open('masterfile_archive','wb') as savefile:
                    pickle.dump(masterfile,savefile)
                break
            except:
                if i==2:
                    self.logger.exception(f'error in {__name__}')
                    print('arechivemaster has failed')
                    return
        for i in range(3): 
            try:
                os.remove(self.masterfilefilename)
                return
            except:
                if i==2:
                    print('masterfile_archive created, but removal of old file has failed')
                    self.logger.exception(f'error in {__name__}')
                    return
                    
    
    def savemasterstatus(self,assignment_tracker,run_dict_status,list_of_run_dicts):
        savedict={'assignment_tracker':assignment_tracker,'run_dict_status':run_dict_status,'list_of_run_dicts':list_of_run_dicts}
        for i in range(3):
            try:
                with open(self.masterfilefilename,'wb') as themasterfile:
                    pickle.dump(savedict,themasterfile)
                with open(self.masterfilefilename+'_backup','wb') as themasterfile:
                    pickle.dump(savedict,themasterfile)
                break
            except:
                sleep(.2)
                if i==2:
                    self.logger.exception(f'error in {__name__}')
                    assert False, 'masterfile problem'

    def rebuild_namefiles(self, run_dict_status, assignment_tracker):
        namelist = self.getnamelist()  # get a new copy just in case
        namefile_tuplist = [self.namefile_statuscheck(name) for name in namelist]
        # print(f'namefile_tuplist:{namefile_tuplist}')
        s_since_update_list = [self.s_before_now(time) for time, status in namefile_tuplist]

        current_name_list = [name for i, name in enumerate(namelist) if (not s_since_update_list[i]==None) and s_since_update_list[i] < self.oldnode_threshold]
        old_name_list1 = [name for i, name in enumerate(namelist) if s_since_update_list[i]==None or 
                          not s_since_update_list[i] < self.oldnode_threshold]

        old_name_list = []
        for name_i in old_name_list1:
            for j in range(10):
                try:
                    idx=assignment_tracker[name_i]
                except:
                    idx=None
                try:
                    time_i = self.model_save_activitycheck(name_i,idx=idx)
                    
                    if not type(time_i) is datetime.timedelta:
                        old_name_list.append(name_i)
                        print(f'1-rebuild_namefiles classifies name_i:{name_i} with time_i:{time_i} as old')
                    elif time_i < self.oldnode_threshold:
                        current_name_list.append(name_i)
                        self.update_my_namefile(name_i,status='working')
                    else: 
                        old_name_list.append(name_i)
                        print(f'2-rebuild_namefiles classifies name_i:{name_i} with time_i:{time_i} as old')
                    break
                except:
                    if j == 1:
                        print(f'-----rebuild namefiles timeout for name_i:{name_i} with time_i:{time_i}', traceback.format_exc())
                        old_name_list.append(name_i)

        if len(old_name_list) > 0:
            print(f'old_name_list:{old_name_list}')
        for j, name in enumerate(old_name_list):
            
            try:
                if self.mergethisnode(name,old=1,move=1):
                    
                    save_idx=[0,0]
                    
                    while not all([idx=='saved' for idx in save_idx]):
                        if not type(save_idx[0]) is str:
                            try:
                                shutil.move(os.path.join(self.masterdirectory, name + '.name'),self.model_collection_directorylist[save_idx[0]])
                                save_idx[0]='saved'
                            except shutil.Error:
                                self.logger.exception(f'save failed for namefile, idx:{save_idx[0]}')
                                try:
                                    self.logger.exception('')
                                    save_idx[0]+=1
                                    if len(self.model_collection_directorylist)-1<save_idx[0]:
                                        new_model_collection_dir=Helper().getname(self.model_collection_directorylist[-1])
                                        self.logger.debug(f'new_model_collection_dir:{new_model_collection_dir}')
                                        self.model_collection_directorylist.append(new_model_collection_dir)
                                        os.mkdir(self.model_collection_directorylist[-1])
                                except:self.logger.exception('')
                            except FileNotFoundError:
                                save_idx[0]='FileNotFoundError'
                            except: self.logger.exception('')
                        if not type(save_idx[1]) is str:
                            try:
                                shutil.move(os.path.join(self.savedirectory, name),self.model_collection_directorylist[save_idx[1]])
                                save_idx[1]='saved'
                            except shutil.Error:
                                self.logger.exception(f'save failed for namefile, idx:{save_idx[1]}')
                                try:
                                    save_idx[1]+=1
                                    if len(self.model_collection_directorylist)-1<save_idx[1]:
                                        new_model_collection_dir=Helper().getname(self.model_collection_directorylist[-1])
                                        self.logger.debug(f'new_model_collection_dir:{new_model_collection_dir}')
                                        self.model_collection_directorylist.append(new_model_collection_dir)
                                        os.mkdir(self.model_collection_directorylist[-1])
                                except:self.logger.exception('')
                            except FileNotFoundError:
                                save_idx[0]='FileNotFoundError'
                            except: self.logger.exception('')
                        #assert all([idx<100 for idx in save_idx if not type(idx) is str]), f'too many _model_collection_ directories. save_idx:{save_idx}'
                            
            except:
                print(f'failed to merge node named:{name}')
                self.logger.exception(f'error in {__name__}')
        
        assigned_to_not_current_name_idx=[]
        #ignment_tracker',assignment_tracker)
        for name_i,idx in assignment_tracker.items():
            for name_j in old_name_list:
                if name_i==name_j:
                    assigned_to_not_current_name_idx.append(idx)
                    break

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
        dorestart=1
        if self.checkmaster(): 
            masterfile=self.getmaster()
            
            
            try: 
                assignment_tracker=masterfile['assignment_tracker']
                list_of_run_dicts=masterfile['list_of_run_dicts']
                run_dict_status=masterfile['run_dict_status']
                model_run_count=len(list_of_run_dicts)
                
                dorestart=0
            except:
                self.logger.exception('restarting master')
                dorestart=1
        
        if dorestart==1:
            assignment_tracker={}
            list_of_run_dicts=self.prep_model_list(
                optdict_variation_list=optdict_variation_list,datagen_variation_list=datagen_variation_list,datagen_dict=self.datagen_dict)
            #list_of_run_dicts=list_of_run_dicts[-1::-1]#reverse the order of the list
            self.setupalljobs(list_of_run_dicts)
            #print(f'list_of_run_dicts[0:2]:{list_of_run_dicts[0:2]},{list_of_run_dicts[-2:]}')
            model_run_count=len(list_of_run_dicts)
            run_dict_status=['ready for node' for _ in range(model_run_count)]
        print('model_run_count',model_run_count)

        i=0;loopcount=0
        shutdownnodes=0
        keepgoing=1
        readynamelist=[]
        while keepgoing:
            if i%100==0:
                self.savemasterstatus(assignment_tracker,run_dict_status,list_of_run_dicts)
            
                
            if loopcount>1:
                loopcount=0

            run_dict_status, assignment_tracker=self.rebuild_namefiles(run_dict_status, assignment_tracker)#get rid of the old names that are inactive
            namelist=self.getnamelist()
            readynamelist=self.getreadynames(namelist)
            if shutdownnodes and len(readynamelist)==0:
                keepgoing=0
            #self.logger.debug('i:{i},loopcount:{loopcount}readynamelist:{readynamelist}')
            if len(readynamelist)>1:
                print(f'readynamelist:{readynamelist}')
            if all([status=='finished' for status in run_dict_status])==True:
                shutdownnodes=1
            ready_dict_idx=[i for i in range(model_run_count) if run_dict_status[i]=='ready for node']
            notanewjob_list=[]
            sleep(1)
            
            for name in readynamelist:
                loopcount+=1
                #name=readynamelist.pop(0)
                
                #ready_dicts=[dict_i for i,dict_i in enumerate(list_of_run_dicts) if run_dict_status[i]=='ready for node']

                try:
                    job_time,job_status=self.check_node_job_status(name,time=1)
                    #print(f'job_time:{job_time},job_status:{job_status} for name:{name}')
                except:
                    print(f'check_node_job_status failed for node:{name}')
                    job_status='failed'
                    job_time='failed'

                #print(f"job_time:{job_time},job_status:{job_status}")
                try:
                    now=strftime("%Y%m%d-%H%M%S")
                    
                    elapsed=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(job_time,"%Y%m%d-%H%M%S")
                    islate=elapsed>self.oldnode_threshold/2
                    
                except:
                    self.logger.exception('')
                    islate=0
                if islate:
                    self.logger.info(f'job_time:{job_time},job_status:{job_status} for name:{name}, islate:{islate}')
                    print(f'job_time:{job_time},job_status:{job_status} for name:{name}, islate:{islate}')

                if not(job_status=="no file found" or islate):
                    if not job_status=='finished':
                        notanewjob_list.append([name,job_status])
                    else:
                        notanewjob_list.append([name,job_status])
                        
                    #nonewjob_namelist=[i[0] for i in notanewjob_list]
                else:
                    print(f'about to setup the job for node:{name}')
                    #print('len(ready_dict_idx)',len(ready_dict_idx))
                    if len(ready_dict_idx) == 0:
                        print('run_dict_status',run_dict_status)
                    if len(ready_dict_idx)>0:
                        random_ready_dict_idx=ready_dict_idx.pop(randint(0,len(ready_dict_idx)-1))#-1 b/c random.randint(a,b) includes b
                        try:
                            run_dict_status[random_ready_dict_idx] = 'assigned'
                            #ready_dict_idx = [ii for ii in range(model_run_count) if run_dict_status[ii] == 'ready for node']
                            if shutdownnodes:
                                newjob='shutdown'
                            else:
                                newjob=list_of_run_dicts[random_ready_dict_idx]
                            #if not name in nonewjob_namelist:
                            self.setup_job_for_node(name,newjob)
                            if islate:
                                try:
                                    job_idx=assignment_tracker[name]
                                    run_dict_status[job_idx]='ready for node'
                                    assignment_tracker[name]=None
                                except:
                                    self.logger.debug('',exc_info=True)
                                    
                            
                            assignment_tracker[name] = random_ready_dict_idx
                            #print('assignment_tracker', assignment_tracker)
                            i+=1
                        except:
                            self.logger.exception('setup job for node failed')
                            run_dict_status[random_ready_dict_idx] = 'ready for node'
                            #ready_dict_idx = [ii for ii in range(model_run_count) if run_dict_status[ii] == 'ready for node']

                            self.logger.exception(f'error in {__name__}')
                            print(f'setup_job_for_node named:{name}, i:{i} has failed')

            for arglist in notanewjob_list:
                name=arglist[0]
                job_status=arglist[1]

                if job_status=='failed':
                    try:
                        job_idx=assignment_tracker[name]
                        tracked=1
                    except:
                        tracked=0
                    try:
                        self.discard_job_for_node(name)
                    except:
                        self.logger.exception(f'deleting assignment_tracker for key:{name} with job_status:{job_status}')
                    if tracked==1: 
                        self.logger.info(f'about to delete assignment_tracker[name]:{assignment_tracker[name]} witj job_idx:{job_idx}')
                        assignment_tracker[name]=None
                        
                        run_dict_status[job_idx]='ready for node'
                        
                    ready_dict_idx=[i for i in range(model_run_count) if run_dict_status[i]=='ready for node']
                    try:
                        self.logger.info(f'namefile for name:{name} is being updated: ready for job')
                        self.update_my_namefile(name,status='ready for job')
                    except:
                        self.logger.exception(f'could not update_my_namefile: name:{name}')
                    mergestatus=self.mergethisnode(name,move=1)
                elif job_status=='finished':
                    print(f'node:{name} has finished')
                    try:
                        job_idx=assignment_tracker[name]
                        tracked=1
                    except:
                        tracked=0
                        print(f'assignment_tracker failed for key:{name}, job_status:{job_status} ')
                        print(f'assignment_tracker:{assignment_tracker}')
                        self.logger.exception(f'assignment_tracker failed for key:{name}, job_status:{job_status} ')
                        self.logger.info(f'assignment_tracker:{assignment_tracker}')
                    try:
                        self.discard_job_for_node(name)
                        print(f'deleting assignment_tracker for key:{name} with job_status:{job_status}')
                        self.logger.info(f'deleting assignment_tracker for key:{name} with job_status:{job_status}')
                        assignment_tracker[name]=None
                        if tracked:
                            run_dict_status[job_idx]='finished'
                        self.update_my_namefile(name,status='ready for job')
                        #mergestatus=self.mergethisnode(name,move=1)
                        #self.logger.info(f'for node name:{name}, mergestatus:{mergestatus}')
                        #print(f'for node name:{name}, mergestatus:{mergestatus}')
                        readynamelist.append(name)
                    except:
                        self.logger.exception('')
            '''if i<100:
                sleep(1)
            else:
                sleep(10)'''
        

            

        #assert i==model_run_count, f"i={i}but model_run_count={model_run_count}"
        self.savemasterstatus(assignment_tracker,run_dict_status,list_of_run_dicts)
        self.archivemaster()
        print('all jobs finished')
        return
    

    
    def mergethisnode(self,name,old=0,move=0):
        nodesdir=os.path.join(self.savedirectory,name)
        if not move:
            for i in range(1):
                try:
                    print(f'trying merge{i}')
                    self.merge_and_condense_saved_models(merge_directory=nodesdir,save_directory=self.savedirectory,condense=0,verbose=0)
                    print(f'completed merge{i}')
                    break
                except:
                    if i==0:
                        try:
                            print(f'trying final merge')
                            self.merge_and_condense_saved_models(merge_directory=nodesdir,save_directory=self.savedirectory,condense=0,verbose=1)
                            print('completed final merge')
                        except:
                            print(f'merge this node failed for node named:{name}')
                            self.logger.exception(f'error in {__name__}')
                            return False
        if move:
            try:
                nodedir=os.path.join(self.savedirectory, name)
                model_save_pathlist=[os.path.join(nodedir,name_i) for name_i in os.listdir(nodedir) if re.search('model_save',name_i)]
                renamelist=[]
                helper=Helper()
                for model_save_path in model_save_pathlist:
                    try:
                        shutil.move(model_save_path,self.model_collection_directorylist[0])
                    except:  
                        try:
                            newpath=os.path.join(self.model_collection_directorylist[0],os.path.split(model_save_path)[-1])
                            newpath=helper.getname(newpath)
                            newstem=os.path.split(newpath)[-1]
                            newoldpath=os.path.join(os.path.split(model_save_path)[0],newstem)
                            os.rename(model_save_path,newoldpath)
                            
                        except:
                            self.logger.exception(f'model_save_path:{model_save_path}')
                        try:
                            shutil.move(newoldpath,newpath)
                        except:
                            self.logger.exception(f'newpath:{newpath}')
            except:
                self.logger.exception('move error in mergethisnode')
                            
        if old:    
            return True
        else:
            return False
                
                
        
    def discard_job_for_node(self,name):
        for i in range(2):
            try:
                os.remove(os.path.join(self.savedirectory,name,name+'_job'))
                break
            except:
                if i==1:
                    self.logger.exception(f'error discarding job for name:{name}')
        return
    
    def setupalljobs(self,rundictlist):
        rundictpathlist=[]
        for idx,rundict in enumerate(rundictlist):
                jobdict={}
                jobdict['optimizedict']=rundict['optimizedict']
                jobdict['datagen_dict']=rundict['datagen_dict']
                now=strftime("%Y%m%d-%H%M%S")
                jobdict['node_status']=[(now,'ready for node')]
            
            for i in range(2):
                try:
                    jobpath=os.path.join(self.jobdirectory,idx+'_job')
                    with open(jobpath,'wb') as f:
                        pickle.dump(jobdict,f)
                    rundictpathlist.append(jobpath)
                    # print(f'newjob has jobdict:{jobdict}')
                    break
                except:
                    if i=1:
                        self.logger.exception(f'error in {__name__}')
                    sleep(0.35)
                
        return rundictpathlist

    def setup_job_for_node(self,name,rundictpath):
            
        for _ in range(10):
            try:
                with open(os.path.join(self.savedirectory,name,name+'_job'),'wb') as f:
                    pickle.dump(rundictpath,f)
                print(f'job setup for node:{name}')
                # print(f'newjob has jobdict:{jobdict}')
                break
            except:
                self.logger.exception(f'error in {__name__}')
                sleep(0.35)
                
        return

    def namefile_statuscheck(self,name):
        nodesnamefilefilename=os.path.join(self.masterdirectory,name+'.name')
        for i in range(2):
            try:
                with open(nodesnamefilefilename,'rb') as savednamefile:
                    namefile=pickle.load(savednamefile)
                break
            except:
                #sleep(0.25)
                if i==1:
                    self.logger.exception(f'error in {__name__}')
                    namefile=[(None,None)]
        return namefile[-1]


    def s_before_now(self,then):
        now=strftime("%Y%m%d-%H%M%S")
        now_s=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")
        if then==None:
            return None
        else:
            return now_s-datetime.datetime.strptime(then,"%Y%m%d-%H%M%S")                   
                        
    
    def getspeciesfromjobfile(self,idx):
        jobfilepath=os.path.join(self.jobdirector,idx+'_job')
        with open(jobfilepath,'rb') as f:
            jobdict=pickle.load(f)
        try:
            species=jobdict['datagen_dict']['species']
        except:
            self.logger.exception(f'getspeciesfromjobfile could not find species for idx:{idx} at jobfilepath:{jobfilepath}')
            species=None
        return species
    
    def model_save_activitycheck(self,name,idx=None):
        nodedir=os.path.join(self.savedirectory,name) 
        if idx==None: 
            assert False,'not developed'
            '''node_model_save_list=[]
            for rootpath,subdirs,files in os.walk(nodedir):
                for filepath in files:
                    if filepath[-10:]=='model_save':
                        node_model_save_list.append(os.path.join(rootpath,filepath))'''

        else:
            species=self.getspeciesfromjobfile(idx)
        filename='species-'species+'_final_model_save'
        
        node_model_save=os.path.join(nodedir,filename)
        if not os.path.exists(node_model_save):
            filename='species-'species+'_model_save'
            node_model_save=os.path.join(nodedir,filename)
            if not os.path.exists(nod_model_save):
                self.logger.info(f'could not find model save file for name:{name},idx:{idx}')
                return None
        #print(node_model_save)
        for  in range(2):
            try:
                with open(node_model_save,'rb') as saved_model_save:
                    model_save=pickle.load(saved_model_save)
                lastsave=model_save[-1]['when_saved']
                
                now=strftime("%Y%m%d-%H%M%S")
                
                sincelastsave=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(lastsave,"%Y%m%d-%H%M%S")
                print(f'activitycheck for name:{name}, time:{sincelastsave},timetype:{type(sincelastsave)}')
                return sincelastsave
            except:
                if i==1:
                    self.logger.info(f'error in {__name__} could not find{filename}')
                    if not filename=='final_model_save':
                        self.model_save_activitycheck(name,filename='final_model_save')
        return None


    def getnamelist(self):
        
        for i in range(3):
            try:
                name_regex=r'\.'+re.escape('name')+r'$'#matches anything ending with .name
                allfiles=os.listdir(self.masterdirectory)
                namefilelist=[name for name in allfiles if re.search(name_regex,name)]
                namelist=[name[:-5] for name in namefilelist]
                return namelist
            except:
                if i==2:
                    print('getnamelist failed')
                    self.logger.exception(f'error in {__name__}')
                sleep(0.25)
            return []


    def runnode(self,myname):
        """for i in range(10):
            try:
                masterfile_exists=self.checkmaster()
                break
            except:
                if i==9:
                    self.logger.exception(f'error in {__name__}')
                    assert False,f"runnode named {myname} could not check master"
        if masterfile_exists==False:
            for i in range(120):
                sleep(log(float(2*i+10)))
                try:
                    masterfile_exists = self.checkmaster()
                    if masterfile_exists:break
                except:
                    if i == 119:
                        self.logger.exception(f'error in {__name__}')
                        assert False, f"runnode named {myname} could not check master" 
        """

        #mydir=os.path.join(self.savedirectory,myname)
        #my_job_file=os.path.join(mydir,myname+'_job')
        
        mydir=self.nodedirectory
        if not os.path.exists(mydir):
            os.mkdir(mydir)
        #self.Ndiff_list_of_masks_x=None
        #self.Ndiff_list_of_masks_y=None
        keepgoing=1
        while keepgoing:
            self.update_my_namefile(myname,status='ready for job')
            start_time=strftime("%Y%m%d-%H%M%S")
            i_have_opt_job=0
            i=0
            print(f'{myname} ischecking for jobs')
            while i_have_opt_job==0:
                my_opt_job=self.check_for_opt_job(myname,start_time,mydir)
                if type(my_opt_job) is dict:
                    break
                elif type(my_opt_job) is str and my_opt_job=='shutdown':
                    self.update_my_namefile(myname,status='shutting down')
                    keepgoing=0
                    return
                else:
                    sleep(2)

            
            my_optimizedict=my_opt_job['optimizedict']
            my_datagen_dict=my_opt_job['datagen_dict']


            self.update_node_job_status(myname,status='starting',mydir=mydir)
            try:
                #kernelcompare.KernelCompare(directory=mydir,myname=myname).run_model_as_node(
                #    my_optimizedict,my_datagen_dict,force_start_params=0)
                self.run_model_as_node(my_optimizedict,my_datagen_dict,force_start_params=0)

                try:
                    self.update_node_job_status(myname,status="finished",mydir=mydir)
                except:
                    print(f'could not update node status for {myname} to finished')
                    self.logger.exception(f'error in {__name__}')
                    #self.runnode(myname+'0')#relying on wrapper function restarting the node
            except:
                try:
                    self.update_node_job_status(myname,status='finished',mydir=mydir)
                except:
                    self.logger.exception(f'error in {__name__}')
                    #self.runnode(myname+'0')#relying on wrapper function restarting the node
                self.logger.exception(f'error in {__name__}')



            #self.runnode(myname)#relying on wrapper function restarting the node



    def check_for_opt_job(self,myname,start_time,mydir):
        assert type(myname) is str,f"myname should be type str not type:{type(myname)}"
        
        my_job_file=os.path.join(mydir,myname+'_job')
        waiting=0
        i=0
        while waiting==0:
            
            try:
                with open(my_job_file,'rb') as myjob_save:
                    myjob=pickle.load(myjob_save)
                if type(myjob) is str and myjob=='shutdown':
                    return myjob
                last_node_status=myjob['node_status'][-1][1]
                if last_node_status=='ready for node':
                    self.update_node_job_status(myname,status='accepted',mydir=mydir)
                    i=0
                    return myjob
                else:
                    #print('myjob status:',myjob['node_status'])
                    i+=1
                    sleep(.25*log(10*i+1))
            except:

                i+=1
                now=strftime("%Y%m%d-%H%M%S")
                s_since_start=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(start_time,"%Y%m%d-%H%M%S")
                if s_since_start>datetime.timedelta(hours=2):#allowing 2 hours instead of using self.oldnode_threshold
                    print(f'node:{myname} checking for job s_since_start="{s_since_start}')
                    self.update_my_namefile(myname,'ready for node')#signal that this node is still active
                    self.logger.exception(f'error in {__name__}')
                
                
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
        for i in range(1):
            try:
                '''with open(namefilename,'rb') as namefile:
                    listoftups=pickle.load(namefile)
                    last5tups=listoftups[-4:]
                    last5tups.append((time_status_tup))'''
                last5tups=[time_status_tup]
                with open(namefilename,'wb') as namefile:
                    pickle.dump(last5tups,namefile)
                break
            except:
                if i==0:
                    self.logger.exception(f'error in {__name__}')
                #sleep(.15)

        return


    def check_node_job_status(self,name,time=None):
        #returns the status of the job in the directory of a node
        if time==None or time=='no':
            time=0
        if time=='yes':
            time=1
        nodes_dir=os.path.join(self.savedirectory,name)
        nodes_job_filename=os.path.join(nodes_dir,name+'_job')
        for i in range(3):
            try:
                with open(nodes_job_filename,'rb') as saved_job_file:
                    nodesjob_dict=pickle.load(saved_job_file)
                #print(f'check_node_job_status found: nodes_jobdict["status"]:{nodesjob_dict["node_status"]}')
                if time==0:
                    return nodesjob_dict['node_status'][-1][1]
                if time==1:
                    #print(f"nodesjob_dict['node_status'][-1]:{nodesjob_dict['node_status'][-1]}")
                    return nodesjob_dict['node_status'][-1][0],nodesjob_dict['node_status'][-1][1]#time_status tup
            except FileNotFoundError:
                if i==2:

                    #self.logger.exception(f'error in {__name__}')
                    if time==0:
                        return "no file found"#if the file doesn't exist, then assign the job
                    if time==1:
                        return strftime("%Y%m%d-%H%M%S"), "no file found"
                sleep(1)
                    
            

    def update_node_job_status(self,myname,status=None,mydir=None):
        self.update_my_namefile(myname,status=status)
        
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
                    self.logger.exception(f'error in {__name__}')
                    job_save_dict={'node_status':[]}
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

    #import mycluster
   
    
    mycluster.run_cluster(myname='node0',local_run=0)





