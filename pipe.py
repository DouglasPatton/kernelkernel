
class PipeLine:

    def __init__(self,):
            
    def build_pipeline(self,stepcount=5,threshcutstep=None,skipstep0=0,bestshare_list=[]):
        '''
        even if step0 is skipped, include it in the step count
        '''
        try:
            self.logger.info(f'stepcount:{stepcount},threshcutstep:{threshcutstep}, skipstep0:{skipstep0},len(bestshare_list):{len(bestshare_list)}')
            pipelinedict={}
            stepdictlist=[]

            if not bestshare_list:
                bestshare_list=[32,1,1,1]#[0.04]+[0.5 for _ in range(stepcount-2)]

            filterthreshold_list=[None for _ in range(stepcount-1)]
            if type(threshcutstep) is int:
                filterthreshold_list[threshcutstep-1]='naiveloss'

            loss_threshold_list=[None for _ in range(stepcount-1)]

            maxiter_list=[1,2,4,4]

            maxbatchbatchcount_list=[2,2,2,4]

            self.max_maxbatchbatchcount=max(maxbatchbatchcount_list) # this is 
            #     used for standardizing variables across steps 
            #     and later to divide training from validation data

            do_minimize_list=[0,1,1,1]#[1 for _ in range(stepcount-1)]

            do_validate_list=[0]+do_minimize_list.copy()

            if not skipstep0:
                optdict_variation_list=self.getoptdictvariations(source=self.source)
                datagen_variation_list=self.getdatagenvariations(source=self.source)
                step0={'variations':{'optdictvariations':optdict_variation_list, 'datagen_variation_list':datagen_variation_list}}
                stepdictlist.append(step0)
            for step in range(stepcount-1):
                filter_kwargs={'filterthreshold':filterthreshold_list[step],
                               'bestshare':bestshare_list[step]}
                startdir=os.path.join(self.modelsavedirectory,'step'+str(step)) #step is incremented by rundict_advance_path
                savedir=startdir
                jobdir=os.path.join(self.jobdirectory,'step'+str(step))
                if not os.path.exists(jobdir):os.mkdir(jobdir)
                stepfolders={'savedir':savedir,'jobdir':jobdir}
                if not os.path.exists(startdir): os.mkdir(startdir)
                ppm_kwargs={'condense':1,'recondense':0,'recondense2':0}
                opt_job_kwargs={ #these are changes to be to opt_dict for next step
                    'loss_threshold':loss_threshold_list[step],
                    'maxiter':maxiter_list[step],
                    'do_minimize':do_minimize_list[step],
                    'maxbatchbatchcount':maxbatchbatchcount_list[step]
                }
                advance_path_kwargs={'i':step,'stepfolders':stepfolders}
                stepdict={'functions':[
                    (self.process_pisces_models,[startdir],ppm_kwargs),
                    (self.merge_dict_model_filter,[],filter_kwargs),
                    (self.opt_job_builder,[],opt_job_kwargs),
                    (self.rundict_advance_path,[],advance_path_kwargs)
                ]}
                stepdictlist.append(stepdict)
            pipelinedict['stepdictlist']=stepdictlist
            validatedictlist=self.makeValidateDictList(stepdictlist,do_validate_list)
            pipelinedict['validatedictlist']=validatedictlist
            return pipelinedict
        except:
            self.logger.exception('build_pipeline error')
            
            
    def makeValidateDictList(self,stepdictlist,do_validate_list):
        try:    
            steps=len(stepdictlist)
            validatedictlist=[None for _ in range(steps)]
            for step in range(steps):
                if do_validate_list[step]:
                    self.logger.debug(f'making validatedict for step:{step}')
                    stepdict=stepdictlist[step]
                    valdict=self.convertStepToValDict(stepdict)
                    validatedictlist[step]=valdict
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
                newstartpath=self.incrementPathEndDigits(startpath)
                newargs=[newstartpath]
            else:
                newargs=oldargs
            newfunctuplist.append((functiontup[0],newargs,newkwargs))
        valdict['functions']=newfunctuplist
        self.logger.debug(f'valdict:{valdict} from stepdict:{stepdict}')
        return valdict
    
    def incrementPathEndDigits(self,path):
        end_digits=''
        for char in path[::-1]:
            if char.isdigit():
                end_digits+=char
            else: break
        digitcount=len(end_digits)
        newpath=path[:-digitcount]+str(int(end_digits)+1)
        self.logger.debug(f'path:{path}, newpath:{newpath}')
        return newpath
    
    
    def doPipeStep(self,stepdict):
        resultslist=[]
        try:
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
        next_i=str(i+1)
        savefolderpath=stepfolders['savedir']
        jobfolderpath=stepfolders['jobdir']
        charcount=len(str(i))+4 # 4 for 'step'
        if validate:
            next_i+='_val'
            #savefolderpath=self.incrementPathEndDigits(savefolderpath)
            #jobfolderpath=self.incrementPathEndDigits(jobfolderpath)
        newjobfolderpath=jobfolderpath[:-charcount]+'step'+next_i
        if not os.path.exists(newjobfolderpath):os.mkdir(newjobfolderpath)
        newsavefolderpath=savefolderpath[:-charcount]+'step'+next_i
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
        