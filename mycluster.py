import pickle
import os,sys,psutil
import re
from time import strftime,sleep
import datetime
import kernelcompare 
import traceback
import shutil
from numpy import log
from random import randint,seed,shuffle
import logging
from helpers import Helper

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
    status='ready'(this process should keep namelist open, so other nodes can't do the exact same thing at the
    same time and both think they got the right name....
    When the master is started, it checks the namelist for anynodes that have not posted for awhile (need to add this next bit)
    and that are not working on a job. The master can also check the nodes model_save file
    '''
    
    def __init__(self,source=None,myname=None,optdict_variation_list=None,datagen_variation_list=None,dosteps=1,local_run=None):
        seed(1)        
        self.oldnode_threshold=datetime.timedelta(minutes=60)
        self.dosteps=dosteps
        self.optdict_variation_list=optdict_variation_list
        self.datagen_variation_list=datagen_variation_list
        if source is None: source='monte'
        self.source=source
        self.savedirectory=self.setdirectory(local_run=local_run)
        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'mycluster_{myname}.log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=1)],
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.helper=Helper()
        if myname=='master':
            self.model_collection_directory=os.path.join(os.getcwd(),'model_collection')
            #self.model_collection_directory=self.helper.getname()#
            self.logger.info(f'self.model_collection_directory:{self.model_collection_directory}')
            if not os.path.exists(self.model_collection_directory): os.mkdir(self.model_collection_directory)

        kernelcompare.KernelCompare.__init__(self,directory=self.savedirectory,source=source,myname=myname)
        
        self.masterdirectory, self.masterfiledirectory=self.setmasterdir(self.savedirectory,myname)
        self.jobdirectory=os.path.join(self.savedirectory,'jobs')
        self.modelsavedirectory=os.path.join(self.savedirectory,'saves')
        if not os.path.exists(self.jobdirectory): os.mkdir(self.jobdirectory)
        if not os.path.exists(self.modelsavedirectory): os.mkdir(self.modelsavedirectory)
        self.masterfilefilename=os.path.join(self.masterfiledirectory, 'masterfile')
        
        if myname is None:
            myname='node'
        
        if not myname=='master':
            platform=sys.platform
            p=psutil.Process(os.getpid())
            if platform=='win32':
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(6)
            return self.initialize(myname)
        
            
        if local_run==None or local_run=='yes' or local_run=='Yes':
            local_run=1
        if local_run=='no' or local_run=='No':
            local_run=0

        #p#rint(f'datagen_variation_list:{datagen_variation_list}')
        self.initialize(myname)

    def generate_rundicts_from_variations(self,step=None):
        if self.optdict_variation_list is None:
            self.optdict_variation_list=self.getoptdictvariations(source=self.source)
        if self.datagen_variation_list is None:
            self.datagen_variation_list=self.getdatagenvariations(source=self.source)
        initial_datagen_dict=self.setdata(self.source)
        list_of_run_dicts=self.prep_model_list(
        optdict_variation_list=self.optdict_variation_list,datagen_variation_list=self.datagen_variation_list,datagen_dict=initial_datagen_dict)
        #list_of_run_dicts=list_of_run_dicts[-1::-1]#reverse the order of the list
        self.setupalljob_paths(list_of_run_dicts,step=step)
        #p#rint(f'list_of_run_dicts[0:2]:{list_of_run_dicts[0:2]},{list_of_run_dicts[-2:]}')
        return list_of_run_dicts
    
    
    def setupalljob_paths(self,rundictlist,step=None):
        if step is None:
            step=0
        for idx,rundict in enumerate(rundictlist):
            jobstepdir=os.path.join(self.jobdirectory,'step'+str(step))
            if not os.path.exists(jobstepdir): os.mkdir(jobstepdir)
            jobpath=os.path.join(jobstepdir,str(idx)+'_jobdict')
            try: species=rundict['datagen_dict']['species']
            except: 
                self.logger.exception('')
                species=''
            savestepdir=os.path.join(self.modelsavedirectory,'step'+str(step))
            if not os.path.exists(savestepdir): os.mkdir(savestepdir)
            savepath=os.path.join(savestepdir,'species-'+species+'_model_save_'+str(idx))
            rundict['jobpath']=jobpath
            rundict['savepath']=savepath

    
    def setmasterdir(self,savedirectory,myname):
        masterdir=os.path.join(savedirectory,'master')
        masterfiledir=os.path.join(os.getcwd(),'master')
        if not os.path.exists(masterdir): os.mkdir(masterdir)
        if myname=='master':
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

    def mastermaster(self,):
        if not self.dosteps:
            list_of_run_dicts=self.generate_rundicts_from_variations()
            return self.runmaster(list_of_run_dicts)
        model_run_stepdict_list=self.build_stepdict_list()
        for i,stepdict in enumerate(model_run_stepdict_list):
            self.logger.debug(f'i:{i}, stepdict:{stepdict}')
            #stepfolders=stepdict['stepfolders']
            try:
                self.logger.debug(f'i:{i},stepdict:{stepdict}')
                if 'variations' in stepdict:
                    list_of_run_dicts=self.generate_rundicts_from_variations()
                    runmasterresult=self.runmaster(list_of_run_dicts)
                    #self.logger.info(f'step#:{i} completed, runmasterresult:{runmasterresult}')
                else:
                    resultslist=[]

                    for functup in stepdict['functions']:
                        args=functup[1]
                        if args==[]:
                            args=[resultslist[-1]]
                        kwargs=functup[2]
                        result=functup[0](*args,**kwargs)
                        resultslist.append(result)
                    list_of_run_dicts=resultslist[-1]
                    self.logger.debug(f'step:{i} len(list_of_run_dicts):{len(list_of_run_dicts)}')
                    #self.rundict_advance_path(list_of_run_dicts,i,stepfolders)
                    runmasterresult=self.runmaster(list_of_run_dicts)
                self.logger.info(f'step#:{i} completed, runmasterresult:{runmasterresult}')
            except:
                self.logger.exception(f'i:{i},stepdict:{stepdict}')
                assert False,'halt'
     
            
        
    def initialize(self,myname):
        
        if myname=="master":
            try:
                self.mastermaster()
            except:
                self.logger.exception('mastermaster failed')
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
            if last_time_status_tup[1] in ['ready','failed','finished']:
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
        masterfile_archive_path=os.path.join(self.masterdirectory,'mastefile_archive')
        masterfile_archive_path=self.helper.getname(masterfile_archive_path)
        try:
            with open(masterfile_archive_path,'wb') as f:
                pickle.dump(masterfile,f)
        except:
            self.logger.exception('archivemaster has failed')
            print('archivemaster has failed')
            return
        try:
            os.remove(self.masterfilefilename)
            try:
                os.remove(self.masterfilefilename+'_backup')
            except:
                self.logger.exception(f'error removing mastefilebackup at:{self.masterfilefilename+"_backup"}')
                
            return
        except:
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
    def myunzzip2(self,tuplist):
        if tuplist:
            return zip(*tuplist)
        else:
            return [],[]
                
    def rebuild_namefiles(self, run_dict_status, assignment_tracker):
        try:
            
            namelist = self.getnamelist()  # get a new copy just in case
            namefile_tuplist = [self.namefile_statuscheck(name) for name in namelist]
            # print(f'namefile_tuplist:{namefile_tuplist}')
            s_since_update_list = [self.s_before_now(time) for time, status in namefile_tuplist]

            current_name_list,current_name_list_tuplist = self.myunzzip2([(name,namefile_tuplist[i]) for i, name in enumerate(namelist) if (not s_since_update_list[i]==None) and s_since_update_list[i] < self.oldnode_threshold])
            old_name_list1,old_name_position_list = self.myunzzip2([(name,i) for i, name in enumerate(namelist) if s_since_update_list[i]==None or 
                              not s_since_update_list[i] < self.oldnode_threshold])
            
            old_name_list = []
            for i,name_i in enumerate(old_name_list1):
                for j in range(10):
                    try:
                        idx=assignment_tracker[name_i]
                    except:
                        idx=None
                    try:
                        time_i = self.model_save_activitycheck(name_i)
                        
                        if not type(time_i) is datetime.timedelta:
                            old_name_list.append(name_i)
                            self.logger.info(f'1-rebuild_namefiles classifies name_i:{name_i} with time_i:{time_i} as old')
                        elif time_i < self.oldnode_threshold:
                            current_name_list.append(name_i)
                            current_name_list_tuplist.append(namefile_tuplist[old_name_position_list[i]])
                            self.update_my_namefile(name_i,status='working')
                        else: 
                            old_name_list.append(name_i)
                            self.logger.info(f'2-rebuild_namefiles classifies name_i:{name_i} with time_i:{time_i} as old')
                        break
                    except:
                        if j == 1:
                            self.logger.info(f'-----rebuild namefiles timeout for name_i:{name_i} with time_i:{time_i}', traceback.format_exc())
                            old_name_list.append(name_i)

            if len(old_name_list) > 0:
                print(f'old_name_list:{old_name_list}')
            for j, name in enumerate(old_name_list):
                
                try:
                    self.mergethisnode(name)
                
                except:
                    self.logger.exception('')
                    self.logger.debug(f'failed to merge node named:{name}')
                
            assigned_to_not_current_name_idx=[]
            #ignment_tracker',assignment_tracker)
            for name_i,idx in assignment_tracker.items():
                for name_j in old_name_list:
                    if name_i==name_j:
                        assigned_to_not_current_name_idx.append(idx)
                        break

            the_not_current_names=[name_i for name_i,idx in assignment_tracker.items() if not any([name_j==name_i for name_j in current_name_list])]
            for idx in assigned_to_not_current_name_idx:
                run_dict_status[idx]='ready'
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
                run_dict_status[idx]='ready'

        except:
            self.logger.exception('')
        #readynamelist=current_name_list
        return run_dict_status, assignment_tracker, current_name_list,current_name_list_tuplist
  


    def runmaster(self,list_of_run_dicts):
        do_startup=1
        if self.checkmaster(): 
            masterfile=self.getmaster()
            
            
            try: 
                assignment_tracker=masterfile['assignment_tracker']
                list_of_run_dicts=masterfile['list_of_run_dicts']
                run_dict_status=masterfile['run_dict_status']
                model_run_count=len(list_of_run_dicts)
                self.logger.info('master is restarting from saved masterfile')
                do_startup=0
            except:
                self.logger.exception('restarting master')
                do_startup=1
        
        if do_startup:
            assignment_tracker={}
            #list_of_run_dicts=self.generate_rundicts_from_variations()
            model_run_count=len(list_of_run_dicts)
            run_dict_status=['ready' for status in range(model_run_count)]
            print('model_run_count',model_run_count)

        i=0;loopcount=0
        keepgoing=1
        #readynamelist=[]
        #next_readynamelist=[]
        while keepgoing:
            
            self.savemasterstatus(assignment_tracker,run_dict_status,list_of_run_dicts)
            
            #loopmax=5*len(readynamelist)
            if True:#loopcount>loopmax or len(next_readynamelist)==0:
                loopcount=0
                

                run_dict_status, assignment_tracker,current_name_list,current_name_list_tuplist=self.rebuild_namefiles(run_dict_status, assignment_tracker)#get rid of the old names that are inactive
                #namelist=self.getnamelist()
                #readynamelist=self.getreadynames(namelist)
                
                #self.logger.debug('i:{i},loopcount:{loopcount}readynamelist:{readynamelist}')
                current_name_count=len(current_name_list)
                self.logger.debug(f'current_name_list_tuplist:{current_name_list_tuplist}, current_name_list:{current_name_list}')
                print(f'current_name_list:{current_name_list}')
                
                if all([status=='finished' for status in run_dict_status]):
                    self.logger.info('all statuses are finished')
                    break
                ready_dict_idx=[i for i in range(model_run_count) if run_dict_status[i]=='ready']
                
                
            else:
                readynamelist=next_readynamelist
            #next_readynamelist=[]
            #shuffle(readynamelist)
            readysleepcount=0
            for i in range(current_name_count):
                loopcount+=1
                #name=readynamelist.pop(0)
                
                #ready_dicts=[dict_i for i,dict_i in enumerate(list_of_run_dicts) if run_dict_status[i]=='ready']
                name=current_name_list[i]
                
                job_time,job_status=current_name_list_tuplist[i]
                try:
                    if assignment_tracker[name] and job_status=='ready':
                        self.logger.debug(f'assignment_tracker[{name}]:{assignment_tracker[name]} and job_status ready')
                        job_status='assigned' # this happens when the node has been assigned a job but has not picked it up yet
                except KeyError:
                    self.logger.debug(f'name:{name} not in assignment_tracker')
                    pass #this happens when there is no assignment_tracker for 
                    #     this node or the assignment tracker evaluates as false (e.g., None)
                except:
                    self.logger.exception(f'unexpected error for name:{name}, job_time:{job_time},job_status:{job_status}')
                if job_status == 'ready':
                    print(f'about to setup the job for node:{name}')
                    self.logger.debug(f'about to setup the job for node:{name}')
                    #p#rint('len(ready_dict_idx)',len(ready_dict_idx))
                    if len(ready_dict_idx) == 0:
                        print('run_dict_status',run_dict_status)
                        self.logger.debug(f'run_dict_status:{run_dict_status}')
                    if len(ready_dict_idx)>0:
                        #next_ready_dict_idx=ready_dict_idx.pop()
                        readysleepcount+=1
                        next_ready_dict_idx=ready_dict_idx.pop(randint(0,len(ready_dict_idx)-1))#-1 b/c random.randint(a,b) includes b
                        try:
                            run_dict_status[next_ready_dict_idx] = 'assigned'
                            #ready_dict_idx = [ii for ii in range(model_run_count) if run_dict_status[ii] == 'ready']
                            #if not name in nonewjob_namelist:
                            setup=self.setup_job_for_node(name,list_of_run_dicts[next_ready_dict_idx])
                            assignment_tracker[name] = next_ready_dict_idx
                            #p#rint('assignment_tracker', assignment_tracker)
                            i+=1
                        except:
                            self.logger.exception(f'setup_job_for_node named:{name}, i:{i} has failed')
                            run_dict_status[next_ready_dict_idx] = 'ready'
                            #ready_dict_idx = [ii for ii in range(model_run_count) if run_dict_status[ii] == 'ready']
                            print(f'setup_job_for_node named:{name}, i:{i} has failed')
                

                elif job_status=='failed':
                    try:
                        job_idx=assignment_tracker[name]
                    except:
                        job_idx=None
                    try:
                        self.discard_job_for_node(name)
                        self.logger.info(f'for name:{name}, about to delete assignment_tracker[name]:{assignment_tracker[name]} with job_idx:{job_idx}')
                        assignment_tracker[name]=None
                        if job_idx:
                            run_dict_status[job_idx]='ready'
                            ready_dict_idx.append(job_idx)
                        try:
                            self.logger.info(f'namefile for name:{name} is being updated: ready for job')
                            self.update_my_namefile(name,status='ready')
                            #next_readynamelist.append(name)    
                        except:
                            self.logger.exception(f'could not update_my_namefile: name:{name}')
                        #mergestatus=self.mergethisnode(name,move=1)

                    except:
                        self.logger.exception(f'error deleting assignment_tracker for key:{name} with job_status:{job_status}')

                    #ready_dict_idx=[i for i in range(model_run_count) if run_dict_status[i]=='ready']

                elif job_status=='finished':
                    print(f'node:{name} has finished')
                    self.logger.info(f'node:{name} has finished')
                    try:
                        job_idx=assignment_tracker[name]
                    except:
                        print(f'assignment_tracker failed for key:{name}, job_status:{job_status} ')
                        print(f'assignment_tracker:{assignment_tracker}')
                        self.logger.exception(f'assignment_tracker failed for key:{name}, job_status:{job_status} ')
                        self.logger.info(f'assignment_tracker:{assignment_tracker}')
                    try:
                        self.discard_job_for_node(name)
                        print(f'deleting assignment_tracker for key:{name} with job_status:{job_status}')
                        self.logger.info(f'deleting assignment_tracker for key:{name} with job_status:{job_status}')
                        assignment_tracker[name]=None
                        run_dict_status[job_idx]='finished'
                        self.update_my_namefile(name,status='ready')
                        #next_readynamelist.append(name)
                        #mergestatus=self.mergethisnode(name,move=1)
                        #self.logger.info(f'for node name:{name}, mergestatus:{mergestatus}')
                        #p#rint(f'for node name:{name}, mergestatus:{mergestatus}')
                        #readynamelist.append(name)
                    except:
                        self.logger.exception('')
                elif job_status=='starting':
                    pass
                    #next_readynamelist.append(name)   
                elif job_status=='assigned':
                    pass # for a node that hasn't picked up it's job yet and that's not old either
                else:
                    self.logger.critical(f'for name:{name} job_status not recognized:{job_status}')
            if readysleepcount==0:
                sleep(5)
        self.savemasterstatus(assignment_tracker,run_dict_status,list_of_run_dicts)
        self.archivemaster()
        return
    

    
    def mergethisnode(self,name):
        nodedir=os.path.join(self.savedirectory, name)
        '''if os.path.exists(nodedir):
            model_save_pathlist=[name_i for name_i in os.listdir(nodedir) if re.search('model_save',name_i)]
            renamelist=[]

            for model_save_path in model_save_pathlist:
                try:
                    newdir=self.model_collection_directory
                    newpath=os.path.join(newdir,model_save_path)
                    unique_newpath=self.helper.getname(newpath)
                    shutil.move(os.path.join(nodedir,model_save_path),unique_newpath)

                except:
                    self.logger.exception(f'move error in mergethisnode name:{name}, model_save_path:{model_save_path}')'''
        try:
            namefilesuffix=name + '.name'
            namefilepath=os.path.join(self.masterdirectory,namefilesuffix)
            destination_namefilepath=self.helper.getname(os.path.join(self.model_collection_directory,namefilesuffix))
            try:
                shutil.move(namefilepath,destination_namefilepath)
            except:
                self.logger.exception('')
                self.logger.debug(f'could not move namefilepath:{namefilepath} to destination_namefilepath:{destination_namefilepath}')
        except:
            self.logger.exception('')
            self.logger.debug(f'error for name:{name}')
        try:    
            nodedir=os.path.join(self.savedirectory,name)
            destination_nodedir=self.helper.getname(os.path.join(self.model_collection_directory,name))
            try:
                shutil.move(nodedir,destination_nodedir)
            except:
                self.logger.exception('')
                self.logger.debug(f'could not move nodedir:{nodedir} to destination_nodedir:{destination_nodedir}')
        except:
            self.logger.exception('')
            self.logger.debug(f'error for name:{name}')
                
        
    def discard_job_for_node(self,name):
        for i in range(5):
            try:
                os.remove(os.path.join(self.savedirectory,name,name+'_job'))
                break
            except:
                sleep(1)
                if i==4:
                    self.logger.exception(f'error discarding job for name:{name}')
                    
                    
        return
    
                                  
    def setup_job_for_node(self,name,rundict):
        nodedir=os.path.join(self.savedirectory,name)
        if not os.path.exists(nodedir):
            self.logger.warning(f'fornode:{name}, adding missing nodedir:{nodedir}')
            os.mkdir(nodedir)
        
        job_file_path=os.path.join(nodedir,name+'_job')
        try:
            with open(job_file_path,'rb') as f:
                myjobfile=pickle.load(f)
        except FileNotFoundError:
            myjobfile={'status':[]}
        except:
            self.logger.exception('')
            myjobfile={'status':[]}
        
        rundictpath=rundict['jobpath']
        myjobfile['jobpath']=rundictpath
        myjobfile['savepath']=rundict['savepath']
        now=strftime("%Y%m%d-%H%M%S")
        time_status_tup=(now,'ready')
        myjobfile['status'].append(time_status_tup)
        try:
            with open(job_file_path,'wb') as f:
                pickle.dump(myjobfile,f)
            print(f'job setup for node:{name}')
            self.logger.debug(f'job setup for node:{name}')
            # print(f'newjob has jobdict:{jobdict}')
        except:
            self.logger.exception(f'error in {__name__}')
            sleep(0.35)
            return 0
        if not os.path.exists(rundictpath):
            jobdict={}
            jobdict['optimizedict']=rundict['optimizedict']
            jobdict['datagen_dict']=rundict['datagen_dict']
            now=strftime("%Y%m%d-%H%M%S")
            jobdict['status']=[(now,'ready')]
            
            self.savepickle(jobdict,rundictpath)
            #self.logger.debug(f'setting up jobdict:{jobdict} at rundictpath:{rundictpath}')
            self.logger.debug(f'setting up jobdict at rundictpath:{rundictpath}')
        return 1

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
                        
    
    def getspeciesfrom_idxjobdict_file(self,idx):
        jobfilepath=os.path.join(self.jobdirector,str(idx)+'_jobdict')
        with open(jobfilepath,'rb') as f:
            jobdict=pickle.load(f)
        try:
            species=jobdict['datagen_dict']['species']
        except:
            self.logger.exception(f'getspeciesfrom_idxjobdict_file could not find species for idx:{idx} at jobfilepath:{jobfilepath}')
            species=None
        return species
    
    def model_save_activitycheck(self,name):
        try:
            nodedir=os.path.join(self.savedirectory,name) 
            node_model_save_pathlist=self.recursive_build_model_save_pathlist(nodedir) #inherited from kernelcompare
            #=os.path.join(nodedir,name+'_job')
            modtimelist=[]
            
            for path in node_model_save_pathlist:
                try:
                    modtime=os.path.getmtime(path)
                    modtimelist.append(modtime)
                except:
                    self.logger.exception(f'problem for name:{name},path:{path}')
            if modtimelist:
                last_model_save=modtimelist.index(max(modtimelist))
                now=strftime("%Y%m%d-%H%M%S")
                sincelastsave=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.fromtimestamp(modtime)
                return sincelastsave
        except:
            self.logger.exception(f'problem with name:{name}')
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
        self.update_my_namefile(myname,status='ready')
        while keepgoing:
            
            #self.update_my_namefile(myname,status='ready')
            start_time=strftime("%Y%m%d-%H%M%S")
            i_have_opt_job=0
            i=0
            print(f'{myname} ischecking for jobs')
            self.logger.debug(f'{myname} ischecking for jobs')
            while i_have_opt_job==0:
                jobfile_dict=self.check_for_opt_job(myname,start_time,mydir)
                if type(jobfile_dict) is dict:
                    break
                elif type(jobfile_dict) is str and my_opt_job=='shutdown':
                    self.update_my_namefile(myname,status='shutting down')
                    keepgoing=0
                    return
                else:
                    sleep(2)
            rundictpath=jobfile_dict['jobpath']
            my_opt_job=self.getpickle(rundictpath)
            my_optimizedict=my_opt_job['optimizedict']
            my_optimizedict['savepath']=jobfile_dict['savepath']
            my_optimizedict['jobpath']=jobfile_dict['jobpath']
            my_datagen_dict=my_opt_job['datagen_dict']


            self.update_node_job_status(myname,status='starting',mydir=mydir)
            try:
                #kernelcompare.KernelCompare(directory=mydir,myname=myname).run_model_as_node(
                #    my_optimizedict,my_datagen_dict,force_start_params=0)
                self.run_model_as_node(my_optimizedict,my_datagen_dict,force_start_params=1)
                self.logger.debug(f'myname:{myname} succeeded')
                success=1
                
            except:
                success=0
                self.logger.exception(f'myname:{myname} failed')
                try:
                    self.update_node_job_status(myname,status='failed',mydir=mydir)
                except:
                    self.logger.exception(f'error in {__name__}')
                    #self.runnode(myname+'0')#relying on wrapper function restarting the node
                self.logger.exception(f'error in {__name__}')
            if success:
                try:
                    self.update_node_job_status(myname,status="finished",mydir=mydir)
                except:
                    print(f'could not update node status for {myname} to finished')
                    self.logger.exception(f'could not update node status for {myname} to finished')
                    #self.runnode(myname+'0')#relying on wrapper function restarting the node


            #self.runnode(myname)#relying on wrapper function restarting the node



    def check_for_opt_job(self,myname,start_time,mydir):
        assert type(myname) is str,f"myname should be type str not type:{type(myname)}"
        
        my_job_file=os.path.join(mydir,myname+'_job')
        waiting=0
        i=0
        updated=0
        while waiting==0:
            
            try:
                with open(my_job_file,'rb') as myjob_save:
                    myjob=pickle.load(myjob_save)
                if type(myjob) is str and myjob=='shutdown':
                    return myjob
                last_status=myjob['status'][-1][1]
                if last_status=='ready':
                    self.update_node_job_status(myname,status='accepted',mydir=mydir)
                    i=0
                    return myjob
                else:
                    #p#rint('myjob status:',myjob['status'])
                    i+=1
                    sleep(.25*log(10*i+1))
            except:

                i+=1
                now=strftime("%Y%m%d-%H%M%S")
                s_since_start=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(start_time,"%Y%m%d-%H%M%S")
                if s_since_start>self.oldnode_threshold/2 and not updated:# :datetime.timedelta(hours=2):#allowing 2 hours instead of using self.oldnode_threshold
                    print(f'node:{myname} checking for job s_since_start="{s_since_start}')
                    self.update_my_namefile(myname,'ready')#signal that this node is still active
                    self.logger.info(f'halfway to time out in {myname}')
                    updated=1
                
                
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
                #p#rint(f'check_node_job_status found: nodes_jobdict["status"]:{nodesjob_dict["status"]}')
                if time==0:
                    return nodesjob_dict['status'][-1][1]
                if time==1:
                    #p#rint(f"nodesjob_dict['status'][-1]:{nodesjob_dict['status'][-1]}")
                    return nodesjob_dict['status'][-1][0],nodesjob_dict['status'][-1][1]#time_status tup
            except FileNotFoundError:
                if i==2:

                    #self.logger.exception(f'error in {__name__}')
                    if time==0:
                        return 'filenotfound' #if the file doesn't exist, then assign the job
                    if time==1:
                        return strftime("%Y%m%d-%H%M%S"), 'filenotfound'
                sleep(1)
                    
            

    def update_node_job_status(self,myname,status=None,mydir=None):
        self.update_my_namefile(myname,status=status)
        
        if mydir==None:
            mydir=os.path.join(self.savedirectory,myname)
        jobpath=os.path.join(mydir,myname+'_job')

        for i in range(10):
            try:
                with open(jobpath,'rb') as f:
                    job_save_dict=pickle.load(f)
                break
            except:
                sleep(.25)
                
                if i==9:
                    self.logger.exception(f'error in {__name__}')
                    job_save_dict={'status':[]}
        if type(status) is str:
            now=strftime("%Y%m%d-%H%M%S")
            job_save_dict['status'].append((now,status))
        for _ in range(10):
            try:
                with open(jobpath,'wb') as f:
                    pickle.dump(job_save_dict,f)
                break
            except:pass

        return

if __name__=="__main__":

    #import mycluster
   
    
    mycluster.run_cluster(myname='node0',local_run=0)





