import os
import numpy as np
from pisces_data_huc12 import PiscesDataTool
from copy import deepcopy

class KernelParams:
    
    def __init__(self,):
        self.n=8 #used to generate variations datagen-batch_n 
        self.batchcount_variation_list=[16]
        self.do_minimize=0
        self.maxiter=2
        
    def build_pipeline(self,stepcount=5,threshcutstep=None,skipstep0=0,bestshare_list=[]):
        '''
        even if step0 is skipped, include it in the step count
        '''
        try:
            self.logger.info(f'stepcount:{stepcount},threshcutstep:{threshcutstep}, skipstep0:{skipstep0},len(bestshare_list):{len(bestshare_list)}')
            pipelinedict={}
            stepdictlist=[]

            if not bestshare_list:
                bestshare_list=[16,4,1,1]#[0.04]+[0.5 for _ in range(stepcount-2)]

            filterthreshold_list=[None for _ in range(stepcount-1)]
            if type(threshcutstep) is int:
                filterthreshold_list[threshcutstep-1]='naiveloss'

            loss_threshold_list=[None for _ in range(stepcount-1)]

            maxiter_list=[1,1,4,4]

            maxbatchbatchcount_list=[2,4,4,8]

            self.max_maxbatchbatchcount=max(maxbatchbatchcount_list) # this is 
            #     used for standardizing variables across steps 
            #     and later to divide training from validation data

            do_minimize_list=[0,0,1,1]#[1 for _ in range(stepcount-1)]

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
        valdict=stepdict.copy()#{key:val for key,val in stepdict.items()}#new dict not a pointer, no copy b/c queue in funcs
        functiontup_list=valdict['functions']
        newfunctuplist=[]
        for functiontup in functiontup_list:
            oldkwargs=functiontup[2]
            newfunctuplist.append(*functiontup[0:2],{'validate':1, **oldkwargs})
        valdict['functions']=newfunctuplist
        self.logger.debug(f'valdict:{valdict} from stepdict:{stepdict}')
        return valdict
        
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
        savefolderpath=stepfolders['savedir']
        jobfolderpath=stepfolders['jobdir']
        charcount=len(str(i))+4 # 4 for 'step'
        if not validate:
            next_i=str(i+1)
        else:
            next_i=str(i)+'_val'
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
        
        
    def getoptdictvariations(self,source='monte'):
        max_bw_Ndiff=2
        
               
        ''''hyper_param_form_dict':{
                'Ndiff_exponent':'free',
                'x_bandscale':'non-neg',
                'Ndiff_depth_bw':'non-neg',
                'outer_x_bw':'non-neg',
                'outer_y_bw':'non-neg',
                'y_bandscale':'fixed'
                '''
        
        arraylist=[]
        for i in range(4):
            startarray=np.ones(4,dtype=np.float64)*0.1
            startarray[i]=0.99
            arraylist.append(startarray)
        x_bandscale_startingvalue_variations=('hyper_param_dict:x_bandscale',arraylist)
        #hyper_param_form_dict_variations=('modeldict:hyper_param_form_dict:x_bandscale',['fixed'])
        #Ndiff_exponentstartingvalue_variations=('hyper_param_dict:Ndiff_exponent',[factor1*np.array([1,1]) for factor1 in [.4,.5,.75,1]])
        Ndiff_exponentstartingvalue_variations=('hyper_param_dict:Ndiff_exponent',[np.array([1,1])])
       
        #Ndiff_exponentstartingvalue_variations=('hyper_param_dict:Ndiff_exponent',[np.array([0,0])])
        #Ndiff_depth_bwstartingvalue_variations=('hyper_param_dict:Ndiff_depth_bw',list(np.linspace(.3,.9,3)))
        Ndiff_depth_bwstartingvalue_variations=('hyper_param_dict:Ndiff_depth_bw',list(np.array([0.5])))
        #Ndiff_outer_x_bw_startingvalue_variations=('hyper_param_dict:outer_x_bw',[np.array([i]) for i in np.linspace(.3,.9,3)])
        Ndiff_outer_x_bw_startingvalue_variations=('hyper_param_dict:outer_x_bw',[np.array([.5])])
        #Ndiff_outer_y_bw_startingvalue_variations=('hyper_param_dict:outer_y_bw',[np.array([i]) for i in np.linspace(.3,.9,3)])
        Ndiff_outer_y_bw_startingvalue_variations=('hyper_param_dict:outer_y_bw',[np.array([.5])])
        
                                  
        NWnorm_variations=('modeldict:NWnorm',['none'])
        #NWnorm_variations=('modeldict:NWnorm',['across-except:batchnorm','none'])
        #binary_y_variations=('modeldict:binary_y',[0.5])
        binary_y_variations=('modeldict:binary_y',[(.5, .45, .55)]) # if binary_y is a tuple,
        #   then optimization chooses continuous phat and calculates alternative MSEs. 'avgavg' means
        #   calculate the avg phat for 0 and for 1 and avg those for the threshold.
        residual_treatment_variations=('modeldict:residual_treatment',['batchnorm_crossval'])
        loss_function_variations=('modeldict:loss_function',['mse'])
        #loss_function_variations=('modeldict:loss_function',['batch_crossval'])
        #cross_mse,cross_mse2
        #loss_function_variations=('modeldict:loss_function',['batch_crossval'])
        #Ndiff_type_variations = ('modeldict:Ndiff_type', ['product','recursive'])
        Ndiff_type_variations = ('modeldict:Ndiff_type', ['recursive'])
        max_bw_Ndiff_variations = ('modeldict:max_bw_Ndiff', [max_bw_Ndiff])
        Ndiff_start_variations = ('modeldict:Ndiff_start', [1])
        product_kern_norm_variations = ('modeldict:product_kern_norm', ['none'])
        #product_kern_norm_variations = ('modeldict:product_kern_norm', ['self'])
        #product_kern_norm_variations = ('modeldict:product_kern_norm', ['none','own_n'])
        #normalize_Ndiffwtsum_variations = ('modeldict:normalize_Ndiffwtsum', ['own_n','none'])
        normalize_Ndiffwtsum_variations = ('modeldict:normalize_Ndiffwtsum', ['own_n'])
        maxbatchbatchcount_variations=('modeldict:maxbatchbatchcount',[1])
        
        if source=='monte':
            standardization_variations=('modeldict:std_data',['all'])
            ykerngrid_form_variations=('modeldict:ykerngrid_form',[('even',4),('exp',4)])
            ykern_grid_variations=('modeldict:ykern_grid',[self.n+1,'no'])
            regression_model_variations=('modeldict:regression_model',['NW','NW-rbf2','NW-rbf'])
            
            
            optdict_variation_list = [
                                      Ndiff_outer_x_bw_startingvalue_variations,
                                      Ndiff_outer_y_bw_startingvalue_variations,
                                      ykerngrid_form_variations,
                                      NWnorm_variations,
                                      loss_function_variations,
                                      residual_treatment_variations,
                                      regression_model_variations, 
                                      product_kern_norm_variations,
                                      normalize_Ndiffwtsum_variations,
                                      Ndiff_type_variations,
                                      ykern_grid_variations,
                                      max_bw_Ndiff_variations,
                                      Ndiff_start_variations,
                                      standardization_variations,
                                     ]#hyper_param_form_dict_variations,
                                     
        if source=='pisces':
            #standardization_variations=('modeldict:std_data',[([],'float')])#a tuple containing lists of variables to standardize in y,x. 'float' means standardize all variables that are floats rather than string
            #standardization_variations=('modeldict:std_data',[([],'float'),([0],'float')])#[i] means standardize the ith variable. for y it can only be [0] or [] for no std
            standardization_variations=('modeldict:std_data',[([],'float')])#[i] means standardize the ith variable. for y it can only be [0] or [] for no std
            ykerngrid_form_variations=('modeldict:ykerngrid_form',[('binary',)])
            ykern_grid_variations=('modeldict:ykern_grid',[2])
            regression_model_variations=('modeldict:regression_model',['NW'])#add logistic when developed fully
            #regression_model_variations=('modeldict:regression_model',['NW'])#add logistic when developed fully
            #spatialtransform_variations=('modeldict:spatialtransform',[('ln1',),('norm1',)])#
            spatialtransform_variations=('modeldict:spatialtransform',['ln1'])#
            optdict_variation_list = [binary_y_variations,
                                      Ndiff_depth_bwstartingvalue_variations,
                                      Ndiff_exponentstartingvalue_variations,
                                      Ndiff_outer_x_bw_startingvalue_variations,
                                      Ndiff_outer_y_bw_startingvalue_variations,
                                      ykerngrid_form_variations,
                                      NWnorm_variations,
                                      loss_function_variations,
                                      regression_model_variations, 
                                      product_kern_norm_variations,
                                      normalize_Ndiffwtsum_variations,
                                      Ndiff_type_variations,
                                      ykern_grid_variations,
                                      max_bw_Ndiff_variations,
                                      Ndiff_start_variations,
                                      standardization_variations,
                                      spatialtransform_variations,
                                      maxbatchbatchcount_variations,x_bandscale_startingvalue_variations
                                     ]
            #hyper_param_form_dict_variations,
        return optdict_variation_list

    def getdatagenvariations(self,source='monte'):
        if source=='monte':
            #the default datagen_dict as of 11/25/2019
            #datagen_dict={'batch_n':32,'batchcount':10, 'param_count':param_count,'seed':1, 'ftype':'linear', 'evar':1, 'source':'monte'}
            batch_n_variations=('batch_n',[self.n])
            batchcount_variations=('batchcount',[4])
            ftype_variations=('ftype',['linear','quadratic'])
            param_count_variations=('param_count',[2,4])
            datagen_variation_list=[batch_n_variations,batchcount_variations,ftype_variations,param_count_variations]
        if source=='pisces':
            try:self.specieslist
            except:
                pdh12=PiscesDataTool()
                self.specieslist=pdh12.returnspecieslist()
            
                
            #species_variations=('species',self.specieslist)
            #species_variations=('species',[self.specieslist[i] for i in range(300,len(self.specieslist))])    
            species_variations=('species',[self.specieslist[i] for i in range(0,2)])
            #species_variations=('species',[self.specieslist[i] for i in range(0,10)])
            #species_variations=('species',[self.specieslist[i] for i in range(20,100,2)])
            # print('species_variations',species_variations)
            #species_variations=('species',[self.specieslist[i] for i in range(0,len(self.specieslist)-11,11)])
            batch_n_variations=('batch_n',[self.n])
            batchcount_variations=('batchcount',self.batchcount_variation_list)
            datagen_variation_list=[batch_n_variations,batchcount_variations,species_variations]
        return datagen_variation_list
    
    
        
    def rebuild_hyper_param_dict(self,old_opt_dict,replacement_fixedfreedict,verbose=None):
        new_opt_dict=old_opt_dict.copy()
        if verbose==None or verbose=='no':
            verbose=0
        if verbose=='yes':
            verbose=1
        vstring=''
        for key,val in new_opt_dict['hyper_param_dict'].items():
            #if not old_opt_dict['modeldict']['hyper_param_form_dict'][key]=='fixed':
            new_val=self.pull_value_from_fixed_or_free(key,replacement_fixedfreedict)
            vstring+=f"for {key} old val({val})replaced with new val({new_val})"
            new_opt_dict['hyper_param_dict'][key]=new_val
        if verbose==1:print(vstring)
        #Sself.logger.debug(vstring)
        return new_opt_dict
    
    
    def build_hyper_param_start_values(self,modeldict):
        depthfactor=0.3
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        Ndiff_start=modeldict['Ndiff_start']
        Ndiff_param_count=max_bw_Ndiff-(Ndiff_start-1)
        
        p=modeldict['param_count']
        if p is None:
            p=2
         
        assert not p==None, f"p is unexpectedly p:{p}"
        if modeldict['Ndiff_type']=='product':
                hyper_paramdict1={
                'Ndiff_exponent':-.1*np.ones([Ndiff_param_count,]),
                'x_bandscale':0.5*np.ones([p,]),
                'outer_x_bw':np.array([0.3,]),
                'outer_y_bw':np.array([0.3,]),
                'Ndiff_depth_bw':depthfactor*np.ones([Ndiff_param_count,]),
                'y_bandscale':1.0*np.ones([1,])
                    }

        if modeldict['Ndiff_type']=='recursive':
            hyper_paramdict1={
                'Ndiff_exponent':0.001*np.ones([Ndiff_param_count,]),
                'x_bandscale':0.5*np.ones([p,]),
                'outer_x_bw':np.array([0.3,]),
                'outer_y_bw':np.array([0.3,]),
                'Ndiff_depth_bw':depthfactor*np.ones([1,]),
                'y_bandscale':1.0*np.ones([1,])
                }
        return hyper_paramdict1
            
                               
    def setdata(self, source):#creates the initial datagen_dict
        
        if source is None or source=='monte':
            source='monte'
            datagen_dict={
                'source':'monte',
                'validate_batchcount':10,
                'batch_n':self.n,
                'batchcount':10,  
                'param_count':2,
                'seed':1, 
                'ftype':'linear', 
                'evar':1                
                                }
        elif source=='pisces':
            #floatselecttup=(2,3,5,6)
            floatselecttup=()
            floatselecttup=(3,5,6)
            spatialselecttup=(8,)
            param_count=len(floatselecttup)+len(spatialselecttup)
            datagen_dict={
                'source':'pisces',
                'batch_n':self.n,
                'batchcount':8, #for batch_crossval and batchnorm_crossval, this specifies the number of groups of batch_n observations to be used for cross-validation. 
                #'batchbatchcount' this has never been part of the dict. it is determined by maxbatchbatchcount in modeldict as well as the available batchbatches.
                'sample_replace':0, #if 1 (formerly 'no') , batches are created until all data is sampled, and sampling with replacement used to fill up the last batchbatch
                #if 0 then drop any observations that don't fit into a batchbatch 
                # need to implement this on training vs. validation subset of data
                'species':'all',
                #'species':'all', #could be 'all', int for the idx or a string with the species name. if 'all', then variations of datagen_dict will be created from pdh12.specieslist
                'missing':'drop_row', #drop the row(observation) if any data is missing
                'floatselecttup':floatselecttup,
                'spatialselecttup':spatialselecttup,
                'param_count':param_count,
                #'validate_batchcount':'none', # 'none'->same as batchcount
                #'validate_batchbatchcount':1 # 'remaining'-> all batchbatches left after training 
            }
        else: 
            assert False, f"error, source not recognized. source:{source}"
            
        self.datagen_dict=datagen_dict
        return datagen_dict   
    
      

    def build_optdict(self,opt_dict_override=None,param_count=None,species=None):
        if opt_dict_override==None:
            opt_dict_override={}
        max_bw_Ndiff=2
        Ndiff_start=1
        Ndiff_param_count=max_bw_Ndiff-(Ndiff_start-1)
        modeldict1={
            #'validate':0, # if int - i, validate upto i batchbatches.
            #     if string - 'remaining' , validate all remaining batchbatches 
            #     and store average of results as well as separate list of all results
            'pthreshold':0.5, # used for assigning phat to yhat=1 or 0
            'binary_y':None, # if not None, then specifies the threshold of p(y=1|x) for predicting 1, e.g., 0.5... 
            #     if tuple of floats 0 to 1, calculates mse/mae for (binary) yhat-y
            'std_data':'all',
            'loss_function':'mse',
            'residual_treatment':'batchnorm_crossval',
            'Ndiff_type':'product',
            'param_count':param_count,
            'Ndiff_start':Ndiff_start,
            'max_bw_Ndiff':max_bw_Ndiff,
            'normalize_Ndiffwtsum':'own_n',
            'NWnorm':'across',
            'xkern_grid':'no',
            'ykern_grid':33,
            'maxbatchbatchcount':1,
            'outer_kern':'gaussian',
            'Ndiff_bw_kern':'rbfkern',
            'outer_x_bw_form':'one_for_all',
            'regression_model':'NW',
            'product_kern_norm':'self',
            'hyper_param_form_dict':{
                'Ndiff_exponent':'fixed',
                'x_bandscale':'non-neg',
                'Ndiff_depth_bw':'non-neg',
                'outer_x_bw':'non-neg',
                'outer_y_bw':'non-neg',
                'y_bandscale':'fixed',#'fixed'
                }
            }
        
        if not species is None:
            modeldict1['species']=species
            modeldict1['spatialtransform']=('divide',1)
        #hyper_paramdict1=self.build_hyper_param_start_values(modeldict1)
        hyper_paramdict1={}
        
        #optimization settings for Nelder-Mead optimization algorithm
        optiondict_NM={
            'xatol':0.005,
            'fatol':.001,
            'adaptive':True,
            'maxiter':self.maxiter
            }
        optiondict_p={'maxiter':self.maxiter}
        optimizer_settings_dict1={
            'method':'Powell',#'BFGS',#''Nelder-Mead',
            'options':optiondict_p,
            'loss_threshold':None,#1,#'naiveloss',
            'help_start':0,
            'partial_match':0,
            'do_minimize':self.do_minimize # do_minimize=0 means just predict once for mse and don't optimize
            }
        
        optimizedict1={
            'opt_settings_dict':optimizer_settings_dict1,
            'hyper_param_dict':hyper_paramdict1,
            'modeldict':modeldict1
            } 
        
        
        newoptimizedict1=self.do_dict_override(optimizedict1,opt_dict_override,verbose=0)
        
        newhyper_paramdict1=self.build_hyper_param_start_values(newoptimizedict1['modeldict'])
        newoptimizedict1['hyper_param_dict']=newhyper_paramdict1
        try: 
            start_val_override_dict=opt_dict_override['hyper_param_dict']
            print(f'start_val_override_dict:{start_val_override_dict}')
            start_override_opt_dict={'hyper_param_dict':start_val_override_dict}
            newoptimizedict1=self.do_dict_override(newoptimizedict1,start_override_opt_dict,verbose=0)
        except:
            pass
        #    self.logger.exception(f'error in {__name__}')             
        #    print('------no start value overrides encountered------')
        #print(f'newoptimizedict1{newoptimizedict1}')
        return newoptimizedict1

