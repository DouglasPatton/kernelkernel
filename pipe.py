import os
import re
import logging

class PipeLine(object):

    def __init__(self,):
        self.logger=logging.getLogger(__name__)
        self.pipestepdict=None
        self.mainstep_setupdict={
            'stepcount':5,
            'startstep':0,
            'bestshare_list':[128,32,8,1],
            'threshcutstep':3,
            'loss_threshold_list':None,
            'do_minimize_list':[0,0,0,1],
            'maxiter_list':[0,0,0,4],
            'maxbatchbatchcount_list':[2,4,8,16],
            'do_validate_list':[0,0,0,1],
            'sidestep':0,
            'overrides':[]
            }
        
        self.sidestep_override_runlist=[[('modeldict:loss_function',lf)] for lf in ['mae']]
        
    def buildsidestep_setupdictlist(self,sidestep_override_runlist,mainstep_setupdict,startstep=1):
        count=len(sidestep_override_runlist)
        sidestep_setupdictlist=[]
        for run_idx in range(count):
            sidestep_setupdict=mainstep_setupdict.copy()
            stepname='sidestep'
            for strtup in sidestep_override_runlist[run_idx]:
                stepname+='_'+re.split(':',strtup[0])[-1]+str(strtup[1])
            self.logger.debug(f'stepname:{stepname}')
            sidestep_setupdict['sidestep']=stepname
            sidestep_setupdict['startstep']=startstep
            sidestep_setupdict['overrides']=sidestep_override_runlist[run_idx]
            sidestep_setupdictlist.append(sidestep_setupdict)
        return sidestep_setupdictlist
                
    def build_pipeline(self,mainstep_setupdict=None,sidestep_override_runlist=None,order='breadth_first'):
        
        if mainstep_setupdict is None:
            mainstep_setupdict=self.mainstep_setupdict
            maxstepcount=mainstep_setupdict['stepcount']
        if sidestep_override_runlist is None:
            sidestep_override_runlist=self.sidestep_override_runlist
        sidestep_setupdictlist=self.buildsidestep_setupdictlist(sidestep_override_runlist,mainstep_setupdict)
        maxstepcount=max([maxstepcount,max([sidestep_setupdict['stepcount'] for sidestep_setupdict in sidestep_setupdictlist])])
        self.logger.debug(f'building mainstep:{mainstep_setupdict}')
        mainstep=[self.build_pipesteps(**mainstep_setupdict)]
        self.logger.debug('building sidesteplist')
        sidesteplist=[self.build_pipesteps(**sidestep_setupdict) for sidestep_setupdict in sidestep_setupdictlist]
        self.pipelinestepdict={
            'mainstep':mainstep,
            'sidesteplist':sidesteplist,
            'order':order
            }
        
        pipelinesteps=[]
        steptype_idx=['mainstep','sidesteplist']
        if order=='depth_first':
            
            for key in steptype_idx:
                stepdictlist=self.pipelinestepdict[key]
                for side_idx,pipestepdict in enumerate(stepdictlist):
                    for step_idx in pipestepdict['steps']:
                        nextpipestep=pipestepdict['stepdictlist'][step_idx]
                        pipelinesteps.append(nextpipestep)
                        if 'validatedictlist' in pipestepdict:
                            valpipestep=pipestepdict['validatedictlist'][step_idx]
                            if valpipestep:
                                pipelinesteps.append(valpipestep)
                        
        elif order=='breadth_first':
            for step_idx in range(maxstepcount):
                for key in steptype_idx:
                    
                    stepdictlist=self.pipelinestepdict[key]
                    for side_idx,pipestepdict in enumerate(stepdictlist):
                        if step_idx in pipestepdict['steps']:
                            step_pos=pipestepdict['steps'].index(step_idx)
                            self.logger.debug(f"step_idx:{step_idx}, pipestepdict['steps']:{pipestepdict['steps']}")
                            nextpipestep=pipestepdict['stepdictlist'][step_pos]
                            pipelinesteps.append(nextpipestep)
                            if 'validatedictlist' in pipestepdict:
                                valpipestep=pipestepdict['validatedictlist'][step_pos]
                                if valpipestep:
                                    pipelinesteps.append(valpipestep)
                        
        else: assert False, f'not expecting order:{order}'
                        
            
        #pipelinesteps=[self.pipestepdict[i0][i1][i2] for i0,i1,i2 in pipesteps_idxlist] #creating flat lists
        
        return pipelinesteps
        
    def build_pipesteps(self,stepcount=None,
            startstep=None,
            bestshare_list=None,
            threshcutstep=None,
            loss_threshold_list=None,
            do_minimize_list=None,
            maxiter_list=None,
            maxbatchbatchcount_list=None,
            do_validate_list=None,
            sidestep=None,
            overrides=None):
        '''
        even if step0 is skipped, include it in the step count
        '''
        
        print(stepcount)
        try:sidestep
        except:sidestep=0
        try:overrides
        except:overrides=[]
        try:
            self.logger.info(f'stepcount:{stepcount},threshcutstep:{threshcutstep}, startstep:{startstep},len(bestshare_list):{len(bestshare_list)}')
            pipestepdict={}
            stepdictlist=[]

            filterthreshold_list=[None for _ in range(stepcount-1)]
            if type(threshcutstep) is int:
                for i in range(threshcutstep-1:stepcount-1):
                    filterthreshold_list[i]='naiveloss'
                
            if loss_threshold_list is None:
                loss_threshold_list=[None for _ in range(stepcount-1)]
            '''if maxiter_list is None:
                maxiter_list=[1,2,4,4]
            if maxbatchbatchcount_list is None:
                maxbatchbatchcount_list=[2,2,2,4]
            if do_minimize_list is None:
                do_minimize_list=[0,1,1,1]
            if do_validate_list is None:    
                do_validate_list=[0]+do_minimize_list.copy()'''

            self.max_maxbatchbatchcount=max(maxbatchbatchcount_list) # this is 
            #     used for standardizing variables across steps 
            #     and later to divide training from validation data
            if not startstep: # could be None or 0
                optdict_variation_list=self.getoptdictvariations(source=self.source)
                datagen_variation_list=self.getdatagenvariations(source=self.source)
                step0={'variations':{'optdict_variation_list':optdict_variation_list, 'datagen_variation_list':datagen_variation_list}} #do not change keys b/c used as kwargs in qcluster
                stepdictlist.append(step0)
                startstep=1
                steplist=[0]
            else:steplist=[]
            for step in range(startstep,stepcount):
                steplist.append(step)
                if sidestep:
                    savefolder_preadvance=sidestep+str(step-1)
                else:
                    savefolder_preadvance=str(step-1)
                if sidestep and step>startstep: #sidesteps startstep searches mainstep path for models
                    prior_step=savefolder_preadvance#not applied at startstep
                else:
                    prior_step=step-1
                    
                step_idx=step-1 # for pulling from it's mainstep_setupdict or sidestep_setupdict
                filter_kwargs={'filterthreshold':filterthreshold_list[step_idx],
                               'bestshare':bestshare_list[step_idx]}
                startdir=os.path.join(self.modelsavedirectory,'step'+str(prior_step)) #step is incremented by rundict_advance_path
                savedir=os.path.join(self.modelsavedirectory,'step'+savefolder_preadvance)
                jobdir=os.path.join(self.jobdirectory,'step'+savefolder_preadvance)
                if not os.path.exists(jobdir):os.mkdir(jobdir)
                stepfolders={'savedir':savedir,'jobdir':jobdir}
                if not os.path.exists(startdir): os.mkdir(startdir)
                ppm_kwargs={'condense':1,'recondense':0,'recondense2':0}
                opt_job_kwargs={ #these are changes to be to opt_dict for next step
                    'loss_threshold':loss_threshold_list[step_idx],
                    'maxiter':maxiter_list[step_idx],
                    'do_minimize':do_minimize_list[step_idx],
                    'maxbatchbatchcount':maxbatchbatchcount_list[step_idx],
                    'overrides':overrides
                }
                advance_path_kwargs={'i':prior_step,'stepfolders':stepfolders}
                stepdict={'functions':[
                    (self.process_pisces_models,[startdir],ppm_kwargs),
                    (self.merge_dict_model_filter,[],filter_kwargs),
                    (self.opt_job_builder,[],opt_job_kwargs),
                    (self.rundict_advance_path,[],advance_path_kwargs)
                ]}
                stepdictlist.append(stepdict)
            pipestepdict['stepdictlist']=stepdictlist
            validatedictlist=self.makeValidateDictList(stepdictlist,do_validate_list)
            pipestepdict['validatedictlist']=validatedictlist
            pipestepdict['steps']=steplist
            return pipestepdict
        except:
            self.logger.exception('build_pipeline error')
            
            
    def makeValidateDictList(self,stepdictlist,do_validate_list):
        try:    
            steps=len(stepdictlist)
            step0=steps-len(do_validate_list) #1 if step0 is included
            validatedictlist=[None for _ in range(steps)]
            for step in range(len(do_validate_list)): 
                if do_validate_list[step]:
                    self.logger.debug(f'making validatedict for step:{step+1}')
                    stepdict=stepdictlist[step+step0]
                    valdict=self.convertStepToValDict(stepdict)
                    validatedictlist[step+step0]=valdict
            return validatedictlist
        except:
            self.logger.exception('makeValidateDictList error')
        
    
    def convertStepToValDict(self,stepdict):
        #adds to every step the kwarg: 'validate':1
        #valdict=stepdict.copy()#{key:val for key,val in stepdict.items()}#new dict not a pointer, no copy b/c queue in funcs
        valdict={}
        functiontup_list=stepdict['functions']
        newfunctuplist=[]
        for f_idx,functiontup in enumerate(functiontup_list):
            oldkwargs=functiontup[2]
            newkwargs={'validate':1, **oldkwargs}
            oldargs=functiontup[1]
            if f_idx==0:
                startpath=oldargs[0]
                newstartpath=self.incrementStringEndDigits(startpath)#incremented,b/c not starting with step before
                newargs=[newstartpath]
            else:
                newargs=oldargs
            newfunctuplist.append((functiontup[0],newargs,newkwargs))
        valdict['functions']=newfunctuplist
        self.logger.debug(f'valdict:{valdict} from stepdict:{stepdict}')
        return valdict
    
    def incrementStringEndDigits(self,oldstring,decrement=0):
        if type(oldstring) is int:
            if decrement:oldstring-=1
            else:oldstring+=1
            return str(oldstring)
        end_digits=''
        for char in oldstring[::-1]:
            if char.isdigit():
                end_digits+=char
            else: break
        digitcount=len(end_digits)
        
        if decrement:
            newstring=oldstring[:-digitcount]+str(int(end_digits)-1)
        else:
            newstring=oldstring[:-digitcount]+str(int(end_digits)+1)
            
        self.logger.debug(f'oldstring:{oldstring}, newstring:{newstring}')
        return newstring
    
    
    def processPipeStep(self,stepdict):
        resultslist=[]
        try:
            if 'variations' in stepdict:
                list_of_run_dicts=self.generate_rundicts_from_variations(**stepdict['variations'])
            else:
                for functup in stepdict['functions']:
                    args=functup[1]
                    if args==[]:
                        args=[resultslist[-1]]
                    kwargs=functup[2]
                    result=functup[0](*args,**kwargs)
                    resultslist.append(result)
                list_of_run_dicts=resultslist[-1]
            return list_of_run_dicts
            self.logger.debug(f'step:{i} len(list_of_run_dicts):{len(list_of_run_dicts)}')
        except:
            self.logger.exception(f'pipestep error stepdict:{stepdict}')
               
    def rundict_advance_path(self,list_of_rundicts,i=None,stepfolders=None,validate=0):
        self.logger.info(f'len(list_of_rundicts):{len(list_of_rundicts)},i:{i},stepfolders:{stepfolders}, validate:{validate}')
        savefolderpath=stepfolders['savedir']
        jobfolderpath=stepfolders['jobdir']
        if validate:
            valstring='_val'
        else:
            valstring=''
        newjobfolderpath=self.incrementStringEndDigits(jobfolderpath)+valstring
        if not os.path.exists(newjobfolderpath):os.mkdir(newjobfolderpath)
        newsavefolderpath=self.incrementStringEndDigits(savefolderpath)+valstring
        if not os.path.exists(newsavefolderpath):os.mkdir(newsavefolderpath)
        self.logger.debug(f'newjobfolderpath:{newjobfolderpath}, newsavefolderpath:{newsavefolderpath}')
        for rundict in list_of_rundicts:
            jobpath=rundict['jobpath']
            savepath=rundict['savepath']
            _,savepathstem=os.path.split(savepath)
            _,jobpathstem=os.path.split(jobpath)
            
            rundict['jobpath']=os.path.join(newjobfolderpath,jobpathstem)
            rundict['savepath']=os.path.join(newsavefolderpath,savepathstem)
        return list_of_rundicts      
        
