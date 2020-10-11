from helpers import Helper
import logging
import os
import re


class KCPisces():
    def __init__(self):
        self.all_species_model_merge_dict_path=os.path.join(self.kc_savedirectory,'all_species_model_merge.pickle')
        try:
            self.logger=logging.getLogger(__name__)
            self.logger.info('starting new KCPisces object')
        except:
            #print(traceback.format_exc())
            #if myname is None: _name=''
            #else: _name=f'-{myname}'
            logdir=os.path.join(directory,'log')
            if not os.path.exists(logdir): os.mkdir(logdir)
            handlername=os.path.join(logdir,__name__)
            logging.basicConfig(
                handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
                level=logging.WARNING,
                format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%Y-%m-%dT%H:%M:%S')
            self.logger = logging.getLogger(handlername)
            self.logger.info('starting new KCPisces log')
       
    def update_species_model_save_path_dict(self,species_model_save_path_dict):
        path=self.species_model_save_path_dict
        if os.path.exists(path):
            existing_species_model_save_path_dict=self.getpickle(path)
            new_species_model_save_path_dict=self.merge_list_of_listdicts([existing_species_model_save_path_dict,species_model_save_path_dict])
        else:
            self.logger.info('savedir:{savedir} does not exist, so species_model_save_path_dict is the first one to be saved')
            new_species_model_save_path_dict=species_model_save_path_dict
        self.savepickle(new_species_model_save_path_dict,path)
        return new_species_model_save_path_dict
    
        
    def merge_dict_model_filter(self,all_species_model_merge_dict,filterthreshold=None,bestshare=None,validate=0):
        '''
        kernelparamsbuild_stepdict_list creates calls for mycluster.mastermaster to run this in sequence so 
        do not change args,kwargs here without changing there
        '''
        self.logger.info(f'merge_dict_model_filter: len(all_species_model_merge_dict):{len(all_species_model_merge_dict)},filterthreshold:{filterthreshold},bestshare:{bestshare}')
        if all_species_model_merge_dict is None:
            all_species_model_merge_dict=self.getpickle(self.all_species_model_merge_dict_path)
        new_model_save_list=[]
        for spec in all_species_model_merge_dict:
            spec_filter_threshold=filterthreshold
            model_save_list=all_species_model_merge_dict[spec]
            self.logger.debug(f'model_save_filter starting spec:{spec} with len(model_save_list):{len(model_save_list)}')
            model_save_list=all_species_model_merge_dict[spec]
            #sorted_condensed_model_list=self.condense_saved_model_list(model_save_list, help_start=0, strict=1,verbose=0,endsort=1,threshold=filterthreshold)
            model_list_losslist=[model_save['loss'] for model_save in model_save_list]
            #self.logger.debug(f'model_list_losslist:{model_list_losslist}')
            #if spec_filter_threshold is None:
            #    spec_filter_threshold=1+max(model_list_losslist)
            if type(spec_filter_threshold) is str:
                if spec_filter_threshold=='naiveloss':
                    spec_filter_threshold=model_save_list[-1]['naiveloss']
            #self.logger.debug(f'spec_filter_threshold:{spec_filter_threshold}')
            if spec_filter_threshold:    
                sorted_model_list=[model_save_list[pos] for loss,pos in sorted(zip(model_list_losslist,list(range(len(model_list_losslist))))) if loss<spec_filter_threshold]
            else:
                sorted_model_list=[model_save_list[pos] for loss,pos in sorted(zip(model_list_losslist,list(range(len(model_list_losslist)))))]
            #self.logger.debug(f'model_save_filter after spec_filter_threshold, len(sorted_model_list):{len(sorted_model_list)}, sorted_model_list[0:2]:{sorted_model_list[0:2]}')
            #help_start applies do_partial_match and will eliminate models with higher nwtloss and only a partial match of parameters.
            if bestshare:
                fullcount=len(sorted_model_list)
                if bestshare<1:
                    bestcount=max([1,int(fullcount*bestshare)])
                elif type(bestshare) is int:
                    bestcount=bestshare
                else: assert False,f'expected decimal or int but got (bestshare,type(bestshare)):{(bestshare,type(bestshare))}'
                new_model_save_list.extend(sorted_model_list[0:bestcount])
            else:
                new_model_save_list.extend(sorted_model_list)
            #self.logger.debug(f'for spec:{spec} len(new_model_save_list):{len(new_model_save_list)}')    
        return new_model_save_list
        
        
    def opt_job_builder(self,model_save_list,maxbatchbatchcount=None,loss_threshold=None,maxiter=None,do_minimize=None, validate=0,overrides=None):
        '''
        
        kernelparamsbuild_stepdict_list creates calls for mycluster.mastermaster to run this in sequence so 
        do not change args,kwargs here without changing there
        '''
        try:
            model_rundict_list=[]
            for model_save in model_save_list:
                new_opt_dict={}
                if overrides:
                    for str_dict_tup in overrides:
                        override_dict=self.build_override_dict_from_str(str_dict_tup[0],str_dict_tup[1])
                        self.logger.debug(f'opt_job_builder override_dict:{override_dict}')
                        model_save=self.do_dict_override(model_save,override_dict)
                modeldict=model_save['modeldict']
                
                    
                opt_settings_dict=model_save['opt_settings_dict']
                expanded_datagen_dict=model_save['datagen_dict']
                if not maxbatchbatchcount is None:
                    modeldict['maxbatchbatchcount']=maxbatchbatchcount
                if not maxiter is None:
                    opt_settings_dict['options']['maxiter']=maxiter
                if not loss_threshold is None:
                    opt_settings_dict['loss_threshold']=loss_threshold
                if not do_minimize is None:
                    opt_settings_dict['do_minimize']=do_minimize
                    
                if validate: 
                        opt_settings_dict['do_minimize']=0
                        opt_settings_dict['loss_threshold']=None
                        modeldict['validate']=validate

                new_opt_dict['opt_settings_dict']=opt_settings_dict
                new_opt_dict['modeldict']=modeldict
                #new_opt_dict['datagen_dict']=expanded_datagen_dict
                defaultoptimizedict=self.build_optdict(param_count=None,species=None)
                optimizedict=self.do_dict_override(defaultoptimizedict,new_opt_dict)
                fof_paramdict=model_save['params']
                self.logger.debug(f'opt_job_builder has fof_paramdict:{fof_paramdict}')
                optimizedict=self.rebuild_hyper_param_dict(optimizedict,fof_paramdict,verbose=0)
                self.logger.debug(f'after rebuild hyper param dict, optimizedict:{optimizedict}')
                optmodel_run_dict={'optimizedict':optimizedict,'datagen_dict':expanded_datagen_dict}  
                optmodel_run_dict['savepath']=model_save['savepath']
                optmodel_run_dict['jobpath']=model_save['jobpath']
                
                model_rundict_list.append(optmodel_run_dict)
            self.logger.debug(f'len(model_rundict_list):{len(model_rundict_list)}')
            return model_rundict_list
        except:self.logger.exception('')

            

        
    def process_pisces_models(self,startpath,condense=0,recondense=0,recondense2=0,merge_with_existing=0,validate=0,overrides=None):
        '''
        kernelparamsbuild_stepdict_list creates calls for mycluster.mastermaster to run this in sequence so 
        do not change args,kwargs here without changing there
        '''
        #species_model_save_path_dict_list=[]
        loss_function=None #if not set by override, comes from model_save modeldict:loss_function
        if overrides:
            for override_tup in overrides:
                if re.search('loss_function',override_tup[0]):
                    loss_function=override_tup[1]
                    break
                
        if validate:
            #startpath=self.incrementstartpath(startpath) # moved to creation of valdict in kernelparams
            condense=1;recondense=0;recondense2=0
            self.logger.debug(f'validation step at startpath:{startpath}')
        try:
            self.logger.info(f'process_pisces_models startpath:{startpath}')
            species_model_save_path_dict=self.split_pisces_model_save_path_dict(startpath)
            if merge_with_existing:
                try:
                    all_species_model_merge_dict=self.getpickle(self.all_species_model_merge_dict_path)
                except FileNotFoundError:
                    all_species_model_merge_dict={}
                except:
                    self.logger.exception('')
                    assert False, 'unexpected exception'
                    
                assert False, 'not developed'
                savepath=self.all_species_model_merge_dict_path
            else:
                savepath=os.path.join(startpath,'all_species_model_merge.pickle')
                all_species_model_merge_dict={}
            try:
                if os.path.exists(savepath):
                    all_species_model_merge_dict=self.getpickle(savepath)
                    self.logger.warning(f'using existing all_species_model_merge_dict from {startpath}')
                    return all_species_model_merge_dict
            except:
                self.logger.exception(f'error when opening all_species_model_merge_dict at {startpath}')

            speciescount=len(species_model_save_path_dict)

            for i,species in enumerate(species_model_save_path_dict):

                pathlist=species_model_save_path_dict[species]
                pathcount=len(pathlist)
                self.logger.debug(f'starting merge; pathcount:{pathcount},species:{species},({i}/{speciescount})')
                mergedlist=self.merge_and_condense_saved_models(
                    species_name=species,
                    pathlist=pathlist,
                    condense=condense,#first condensing addresses iterations
                    recondense=recondense,returnlist=1,
                    loss_function=loss_function
                    )
                self.logger.debug(f'len(mergedlist):{len(mergedlist)}')
                if species not in all_species_model_merge_dict:
                    all_species_model_merge_dict[species]=[]
                all_species_model_merge_dict[species].extend(mergedlist)
                if recondense2:
                    all_species_model_merge_dict[species]=self.condense_saved_model_list(all_species_model_merge_dict[species], help_start=0, strict=1,verbose=0,loss_function=loss_function)
            self.logger.debug(f'len(all_species_model_merge_dict):{len(all_species_model_merge_dict)}')

            self.savepickle(all_species_model_merge_dict,savepath)
            return all_species_model_merge_dict
        except:
            self.logger.exception("")
            assert False, 'Halt'
    
    
    def print_pisces_all_species_model_merge_dict(self,shortlist=[],startpath=None):
        if startpath is None:
            all_species_model_merge_dict=self.getpickle(self.all_species_model_merge_dict_path)
        speciescount=len(all_species_model_merge_dict)
        for i,species in enumerate(all_species_model_merge_dict):
            model_list=all_species_model_merge_dict[species]
            
            self.logger.debug(f'printing {species}; ({i}/{speciescount}), with {len(model_list)} models.')
            self.print_model_save(directory=self.kc_savedirectory,model_save_list=model_list,shortlist=shortlist,species=species)
            
    
        
    def split_pisces_model_save_path_dict(self,startdir):
        self.logger.debug('starting to add species names')
        #self.addspecies_name_and_resave(startdir=startdir)
        self.logger.debug('finished adding species names')
        
        
        model_save_pathlist=self.recursive_build_model_save_pathlist(startdir)
        
        #if not species_model_save_path_dict:
        species_model_save_path_dict={}
        for path in model_save_pathlist:
            #regex_genus_species=r'species-[(a-z)]+\s[a-z]+_'
            #regex_genus_species=r'species-[(a-z)]+(\s[a-z]+)+_'# this one will match 3 word species like cyprinella venusta cercostigma
            regex_genus_species=r'species-[(a-z)]+(\s[\-\.a-z]+)+_'# this one will match species with - or . in name
            searchresult=re.search(regex_genus_species,path)
            if searchresult:
                species_genus_slicer=slice(searchresult.start()+8,searchresult.end()-1)
                spec_name=path[species_genus_slicer]
            
                if not spec_name in species_model_save_path_dict:
                    self.logger.debug(f'adding spec_name:{spec_name} to species_model_save_path_dict which has len:{len(species_model_save_path_dict)}')
                    species_model_save_path_dict[spec_name]=[]
                species_model_save_path_dict[spec_name].append(path)
            else:
                self.logger.debug(f'split_pisces_species_model_save ignoring path:{path}')
        return species_model_save_path_dict

    
        
        
    def getspecies_name_from_model_save(self,path):
        try:
            model_save_list=self.getpickle(path)
            for model_save in model_save_list:
                    model_save['datagen_dict']['species']
                    
        except:
            self.logger.exception(f'error when retrieving species name form model_save path:{path}')
            return []

    def addspecies_name_and_resave(self,startdir=None,filepath=None):
        if filepath is None:
            model_save_pathlist=self.recursive_build_model_save_pathlist(startdir)
        else:
            if type(filepath) is str:
                filepath=[filepath]
            model_save_pathlist=[filepath]
            
        #self.logger.debug(f'model_save_pathlist:{model_save_pathlist}')
        species_model_dictlist={}
        species_save_path_toformat=os.path.join(os.path.split(startdir)[0],'species-{}_model_save')
        count=0
        for path in model_save_pathlist:
            if not os.path.split(path)[1][:8]=='species-':
                count+=1
                self.logger.debug(f'species name addition #{count}')
                try:
                    model_save_list=self.getpickle(path)
                except:
                    model_save_list=[]
                    self.logger.exception(f'problem retrieving path:{path}')
                if model_save_list:
                    for model_save in model_save_list:
                        try:
                            spec_name=model_save['datagen_dict']['species']
                        except:
                            self.logger.exception('no species name found')
                            self.logger.debug(f'model_save:{model_save}')
                            spec_name=''
                        if spec_name:
                            if not spec_name in species_model_dictlist:
                                species_model_dictlist[spec_name]=[model_save]
                            else:
                                species_model_dictlist[spec_name].append(model_save)
        for spec in species_model_dictlist:
            newpath=self.helper.getname(species_save_path_toformat.format(spec))
            spec_model_list=species_model_dictlist[spec]
            self.savepickle(spec_model_list,newpath)
