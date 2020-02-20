from copy import deepcopy
from time import strftime
import numpy as np
import pickle
import os
import datagen as dg
import mykern as mk
import re
import logging, logging.config
import yaml
from kernelparams import KernelParams
#import datetime

class KernelOptModelTools(mk.kNdtool):
    def __init__(self,directory=None,myname=None):
        if directory==None:
            self.kc_savedirectory=os.getcwd
        else:
            self.kc_savedirectory=directory
        mk.kNdtool.__init__(self,savedir=self.kc_savedirectory,myname=myname)
        
        self.name=myname
        '''
        logging.basicConfig(level=logging.INFO)
        logdir=os.path.join(self.kc_savedirectory,'log')
        #if not os.path.exists(logdir): 
        try:
            os.mkdir(logdir)
        except:pass
        handlername=f'kernelcompare.log'
        handler=logging.FileHandler(os.path.join(logdir,handlername))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        '''
        '''
        with open(os.path.join(os.getcwd(),'logconfig.yaml'),'rt') as f:
            configfile=yaml.safe_load(f.read())
        logging.config.dictConfig(configfile)
        self.logger = logging.getLogger('kcLogger')
        '''
                
    def do_monte_opt(self,optimizedict,datagen_dict,force_start_params=None):
        optimizedict['datagen_dict']=datagen_dict
        
        if force_start_params==None or force_start_params=='no':
            force_start_params=0
        if force_start_params=='yes':
            force_start_params=1

        datagen_obj=dg.datagen(datagen_dict)
        maxbatchbatchcount=optimizedict['modeldict']['maxbatchbatchcount']
        if type(maxbatchbatchcount) is int:
            thisbatchbatchcount=maxbatchbatchcount*datagen_obj.batch_n
            
        if datagen_obj.species_n<datagen_obj.batch_n*datagen_obj.batchcount:
            print(f'skipping {datagen_obj.species} b/c species_n:{datagen_obj.species_n} < datagen_obj.batch_n*datagen_obj.batchcount:{datagen_obj.batch_n*datagen_obj.batchcount}')
            self.logger.info(f'skipping {datagen_obj.species} b/c species_n:{datagen_obj.species_n} < fullbatchbatch_n:{datagen_obj.fullbatchbatch_n}')
            return
            
        naive_mse=self.do_naivemse(datagen_obj)
        print('naive_mse:',naive_mse)
        datagen_dict_expanded=datagen_obj.datagen_dict_expanded
        optimizedict['datagen_dict']=datagen_dict_expanded
        print(f'datagen_dict_expanded:{datagen_dict_expanded} for directory,{self.kc_savedirectory}')
        
        if force_start_params==0:
            optimizedict=self.run_opt_complete_check(optimizedict,replace=1)
            
            
        start_msg=f'starting at {strftime("%Y%m%d-%H%M%S")}'
        
        mk.optimize_free_params(datagen_obj,optimizedict,savedir=self.kc_savedirectory,myname=self.name)
        return
        
                               
    
                               
                               
    def run_opt_complete_check(self,optimizedict_orig,replace=None):
        '''
        checks model_save and then final_model_save to see if the same modeldict has been run before (e.g.,
        same model featuers, same starting parameters, same datagen_dict).
        -by default or if replace set to 1 or 'yes', then the start parameters are replaced by the matching model from model_save 
            and final_model_save with the lowest 'nwt mse "
        -if replace set to 0 or 'no', then the best matching model is still announced, but replacement of start parameters doens't happen
            Only mse and parameters pulled by self.pull2dicts are compared/replaced. modeldict and datagen_dict
        '''
        optimizedict=optimizedict_orig.copy()
        if replace==None or replace=='yes':
            replace=1
        if replace=='no':
            replace=0
        best_dict_list=[]
        help_start=optimizedict['opt_settings_dict']['help_start']
        #p#rint(f'help_start:{help_start}')
        partial_match=optimizedict['opt_settings_dict']['partial_match']
         #search the parent directory first
        condensedfilename=os.path.join(self.kc_savedirectory,'..','condensed_model_save')
        same_modelxy_dict_list1=self.open_and_compare_optdict(
            condensedfilename,optimizedict,help_start=help_start,partial_match=partial_match)
        #print('here1')
        if len(same_modelxy_dict_list1)==0:
            condensedfilename=os.path.join(self.kc_savedirectory,'condensed_model_save')
            same_modelxy_dict_list2=self.open_and_compare_optdict(
                condensedfilename,optimizedict,help_start=help_start,partial_match=partial_match)
            #print('here2')
            if len(same_modelxy_dict_list2)==0:
                same_modelxy_dict_list3=self.open_and_compare_optdict(
                    'model_save',optimizedict,help_start=help_start,partial_match=partial_match)
                same_modelxy_dict_list=same_modelxy_dict_list3#even if it is empty
            else: same_modelxy_dict_list=same_modelxy_dict_list2
        else: same_modelxy_dict_list=same_modelxy_dict_list1
            
        
        
        if len(same_modelxy_dict_list)>0:
            #p#rint(f"from model_save, This dictionary, x,y combo has finished optimization before:{len(same_modelxy_dict_list)} times")
            #p#rint(f'first item in modelxy_dict_list:{same_modelxy_dict_list[0]}'')
            mse_list=[dict_i['mse'] for dict_i in same_modelxy_dict_list]
            try:
                n_list=[dict_i['datagen_dict']['train_n'] for dict_i in same_modelxy_dict_list]
                batchcount_list=[1 for _ in range(len(n_list))]
            except:
                n_list=[dict_i['datagen_dict']['batch_n'] for dict_i in same_modelxy_dict_list]
                batchcount_list=[dict_i['datagen_dict']['batchcount'] for dict_i in same_modelxy_dict_list]

            batchbatchcountlist=[];naivemselist=[]
            for dict_i in same_modelxy_dict_list:
                try:
                    i_batchbatch1=dict_i['datagen_dict']['batchbatchcount']
                    i_batchbatch2=dict_i['modeldict']['maxbatchbatchcount']
                    if i_batchbatch1<i_batchbatch2:
                        batchbatchcountlist.append(i_batchbatch1)
                    else: batchbatchcountlist.append(i_batchbatch2)
                except: batchbatchcountlist.append(1)
                try:
                    naivemselist.append(dict_i['naivemse'])
                except: naivemselist.append(1)
                           
            n_wt_mse_list=[self.do_nwt_mse(mse_list[i],n_list[i],batchcount_list[i],batchbatchcount=batchbatchcountlist[i],naivemse=naivemselist[i]) for i in range(len(mse_list))]
            lowest_n_wt_mse=min(n_wt_mse_list)
            
            best_dict_list.append(same_modelxy_dict_list[n_wt_mse_list.index(lowest_n_wt_mse)])
        #if len(same_modelxy_dict_list)==0:
            #print('--------------no matching models found----------')
        
        mse_list=[dict_i['mse'] for dict_i in best_dict_list]
        if len(mse_list)>0:
            mse_list=[dict_i['mse'] for dict_i in best_dict_list]
            try:
                n_list=[dict_i['datagen_dict']['train_n'] for dict_i in same_modelxy_dict_list]
                batchcount_list=[1 for _ in range(len(n_list))]
            except:
                n_list=[dict_i['datagen_dict']['batch_n'] for dict_i in same_modelxy_dict_list]
                batchcount_list=[dict_i['datagen_dict']['batchcount'] for dict_i in same_modelxy_dict_list]

            batchbatchcountlist=[];naivemselist=[]
            for dict_i in same_modelxy_dict_list:
                try:
                    i_batchbatch1=dict_i['datagen_dict']['batchbatchcount']
                    i_batchbatch2=dict_i['modeldict']['maxbatchbatchcount']
                    if i_batchbatch1<i_batchbatch2:
                        batchbatchcountlist.append(i_batchbatch1)
                    else: batchbatchcountlist.append(i_batchbatch2)
                except: batchbatchcountlist.append(1)
                try:
                    naivemselist.append(dict_i['naivemse'])
                except: naivemselist.append(1)
                           
            n_wt_mse_list=[self.do_nwt_mse(mse_list[i],n_list[i],batchcount_list[i],batchbatchcount=batchbatchcountlist[i],naivemse=naivemselist[i]) for i in range(len(mse_list))]
            lowest_n_wt_mse=min(n_wt_mse_list)
            best_dict=best_dict_list[n_wt_mse_list.index(lowest_n_wt_mse)]
            
            #try:print(f'optimization dict with lowest mse:{best_dict["mse"]}, n:{best_dict["ydata"].shape[0]}was last saved{best_dict["whensaved"]}')
            printstring=f"optimization dict with lowest mse:{best_dict['mse']}, n:{best_dict['datagen_dict']['batch_n']}was last saved{best_dict['when_saved']}"
            print(printstring)
            self.logger.info(printstring)

            #p#rint(f'best_dict:{best_dict}')
            if replace==1:
                self.logger.info("overriding start parameters with saved parameters")
                optimizedict=self.rebuild_hyper_param_dict(optimizedict,best_dict['params'],verbose=0)
            else:
                print('continuing without replacing parameters with their saved values')
        return(optimizedict)
    
    def rebuild_hyper_param_dict(self,old_opt_dict,replacement_fixedfreedict,verbose=None):
        new_opt_dict=old_opt_dict.copy()
        if verbose==None or verbose=='no':
            verbose=0
        if verbose=='yes':
            verbose=1
        vstring=''
        for key,val in new_opt_dict['hyper_param_dict'].items():
            if not old_opt_dict['modeldict']['hyper_param_form_dict'][key]=='fixed':
                new_val=self.pull_value_from_fixed_or_free(key,replacement_fixedfreedict,transform='no')
                vstring+=f"for {key} old val({val})replaced with new val({new_val})"
                new_opt_dict['hyper_param_dict'][key]=new_val
        if verbose==1:print(vstring)
        return new_opt_dict


    def recursive_merge(self,startdirectory,overwrite=0,verbose=0,condense=0):
        if verbose==1:
            print(f'startdirectory:{startdirectory}')
        if not os.path.exists(startdirectory):
            startdirectory=os.path.join(os.getcwd,startdirectory)
        if overwrite==0:
            save_directory=os.path.join(startdirectory,'mergedfiles')
            if not os.path.exists(save_directory):
                os.mkdir(save_directory)
        else:save_directory=startdirectory

        dirlist=[diri[0] for diri in os.walk(startdirectory)]
        dirlist=[startdirectory]+dirlist
        for diri in dirlist:
            if verbose==1:print('diri',diri)
            self.merge_and_condense_saved_models(merge_directory=diri, save_directory=save_directory, condense=condense,verbose=verbose)

    def recursive_add_dict(self,startdirectory,add_tuple_list,overwrite=0,verbose=0):
        if not type(add_tuple_list) is list:
            add_tuple_list=[add_tuple_list]
        if not os.path.exists(startdirectory):
            startdirectory=os.path.join(os.getcwd,startdirectory)
        if overwrite==1:
            save_directory=os.path.join(startdirectory,'add_dict_files')
            if not os.path.exists(save_directory):
                os.mkdir(save_directory)
        else:
            save_directory = startdirectory

        dirlist = [dir_i[0] for dir_i in os.walk(startdirectory)]
        dirlist = [startdirectory] + dirlist
        for dir_i in dirlist:
            filelist=os.listdir(dir_i)
            for file_i in filelist:
                if re.search('model_save',file_i):
                    self.add_dict(
                        os.path.join(startdirectory,dir_i,file_i),add_tuple_list,overwrite=overwrite,verbose=verbose)
        return
    
    
    def deletefromsavedict(self,filename,flatdictstring,overwrite=0,verbose=0):
        if overwrite == 0 or overwrite==None:
            writefilename = filename + '-overwrite_dict'
        else:
            writefilename = filename
        for j in range(10):
            try:
                with open(os.path.join(self.kc_savedirectory,filename),'rb') as modelsavefile:
                    modelsave_list=pickle.load(modelsavefile)
                break
            except:
                if j==9:
                    self.logger.exception(f'error in {__name__}')
                    return
                
        delete_dict=self.build_override_dict_from_str(flatdictstring,None)
        new_modelsave_list=[]
        for savedict in modelsave_list:
            new_modelsave_list.extend(self.do_dict_override(savedict,delete_dict,deletekey=1,verbose=verbose))
        for j in range(10):
            try:
                with open(os.path.join(self.kc_savedirectory,writefilename),'wb') as modelsavefile:
                    pickle.dump(new_modelsave_list,modelsavefile)
                return
            except:
                if j==9:
                    self.logger.exception(f'error in {__name__}')
                    return
        
        
    def overwrite_savedict(self,filename,flatdict_tup,verbose=0,overwrite=0,overwrite_condition=None):
        '''
        flatdict_tup might look like this: ('modeldict:max_bw_Ndiff',4)
            where the first item is a flat dict and the second items is the new value.
        overwrite_condition is used when some values shouldn't be overwritten
        '''
        if overwrite == 0 or overwrite==None:
            writefilename = filename + '-overwrite_dict'
        else:
            writefilename = filename
        for j in range(10):
            try:
                with open(os.path.join(self.kc_savedirectory,filename),'rb') as modelsavefile:
                    modelsave_list=pickle.load(modelsavefile)
                break
            except:
                if j==9:
                    self.logger.exception(f'error in {__name__}')
                    return
        override_dict=self.build_override_dict_from_str(flatdict_tup[0],flatdict_tup[1])
        if not type(overwrite_condition) is tuple:
            condition_dict=override_dict
        else: 
            condition_dict=self.build_override_dict_from_str(overwrite_condition[0],overwrite_condition[1])
            overwrite_condition=overwrite_condition[1]
        new_modelsave_list=[]
        for savedict in modelsave_list:
            if not overwrite_condition==None:
                current_value=self.pull_nested_key(savedict,condition_dict)
                if current_value==overwrite_condition:
                    new_modelsave_list.append(self.do_dict_override(savedict,override_dict,replace=1,verbose=verbose))
                else:
                    print(current_value,'not overriden b/c does not match condition:',overwrite_condition)
                    new_modelsave_list.append(savedict)
            else:
                new_modelsave_list.append(self.do_dict_override(savedict,override_dict,replace=1,verbose=verbose))
        for j in range(10):
            try:
                with open(os.path.join(self.kc_savedirectory,writefilename),'wb') as modelsavefile:
                    pickle.dump(new_modelsave_list,modelsavefile)
                return
            except:
                if j==9:
                    self.logger.exception(f'error in {__name__}')
                    return
            
    def pull_nested_key(self,maindict,nesteddict,recursive=0):
        if recursive==0:
            vstring=''
        if type(recursive) is str:
            vstring=recursive
            
        for key,val in nesteddict.items():
            if not key in maindict:
                keylist=[key for key,val in maindict.items()]
                vstring+=f'key:{key} not in list of keys:{keylist}'
                print(vstring)
                return None
            val2=maindict[key]
            if type(val2) is dict:
                if not type(val) is dict:
                    vstring+=f'for key:{key} maindict[key] is a dict but type(val):{type(val)}'
                    print(vstring)
                    return None
                return self.pull_nested_key(val2,val,recursive=vstring)
            else:
                return val2
            
            

    def add_dict(self,filename,newdict_tup_list,verbose=0,overwrite=0):
        '''
        newdict_tup_list should have same format as optdict and datagen_dict variations do:
        e.g., (modeldict:ykern_grid,'no') to replace ykern_grid, nested in modeldict
        '''
        if overwrite == 0:
            writefilename = filename + '-add_dict'
        else:
            writefilename = filename
        if not type(newdict_tup_list) is list:
            newdict_tup_list=[newdict_tup_list]

        for j in range(10):
            try:
                with open(os.path.join(self.kc_savedirectory,filename),'rb') as modelsavefile:
                    modelsave_list=pickle.load(modelsavefile)
                break
            except:
                if j==9:
                    self.logger.exception(f'error in {__name__}')
                    return

        for dict_to_add in newdict_tup_list:
            print('dict_to_add',dict_to_add)
            for i,rundict_i in enumerate(modelsave_list):
                override_dict_i = self.build_override_dict_from_str(dict_to_add[0], dict_to_add[1])
                modelsave_list[i]=self.do_dict_override(rundict_i, override_dict_i, replace=0, verbose=verbose)

        for j in range(10):
            try:
                with open(os.path.join(self.kc_savedirectory,writefilename),'wb') as modelsavefile:
                    pickle.dump(modelsave_list, modelsavefile)
                break
            except:
                if j==9:
                    self.logger.exception(f'error in {__name__}')
                    break
        return


    def open_condense_resave(self,filename1,verbose=None):
        if verbose==None or verbose=='no':
            verbose=0
        if verbose=='yes':
            verbose=1

        with open(filename1,'rb') as savedfile:
            saved_model_list1=pickle.load(savedfile)
        condensed_list=self.condense_saved_model_list(saved_model_list1, help_start=0, strict=1,verbose=verbose)
          
        with open(filename1,'wb') as writefile:
            pickle.dump(condensed_list,writefile)

    def print_model_save(self,filename=None,directory=None):
        import pandas as pd
        if directory==None:
            directory=os.getcwd()
        if filename==None:
            filename='model_save'
        pd.set_option('display.max_colwidth', -1)
        file_loc=os.path.join(directory,filename)
        for i in range(10):
            try:
                exists=os.path.exists(file_loc)
                if not exists:
                    print(f'file:{file_loc} has os.path.exists value:{exists}')
                    return
                with open(file_loc,'rb') as model_save:
                    model_save_list=pickle.load(model_save)
            except:
                if i==9:
                    self.logger.info(f'could not open{file_loc}')
                    self.logger.exception(f'error in {__name__}')
                    return
        if len(model_save_list)==0:
            self.logger.info(f'no models in model_save_list for printing')
            return
        try:
            model_save_list.sort(key=lambda savedicti: savedicti['mse']/savedicti['naivemse'])
            
        except:
            print(traceback.format_exc())
            model_save_list.sort(key=lambda savedicti: savedicti['mse'])
              #sorts by mse

        output_loc=os.path.join(directory,'output')
        if not os.path.exists(output_loc):
            os.mkdir(output_loc)

        filecount=len(os.listdir(output_loc))
        output_filename = os.path.join(output_loc,f'models{filecount}'+'.html')

        modeltablehtml=''
        #keylist = ['mse','params', 'modeldict', 'when_saved', 'datagen_dict']#xdata and ydata aren't in here
        
        self.logger.info(f'len(model_save_list:{len(model_save_list)}')
        for j,model in enumerate(model_save_list):
            keylistj=[key for key in model]
            simpledicti=self.myflatdict(model,keys=keylistj)
            this_model_html_string=pd.DataFrame(simpledicti).T.to_html()
            modeltablehtml=modeltablehtml+f'model:{j+1}<br>'+this_model_html_string+"<br>"
        for i in range(10):
            try:
                with open(output_filename,'w') as _htmlfile:
                    _htmlfile.write(modeltablehtml)
                return
            except:
                if i==9:
                    self.logger.critical(f'could not write modeltablehtml to location:{output_filename}')
                    self.logger.exception(f'error in {__name__}')
                    return

    '''def getmodelrunmse(self,modelrundict):
        return modelrundict['mse']'''
                
    def myflatdict(self, complexdict, keys=None):
        thistype = type(complexdict)
        if not thistype is dict:
            return {'val': complexdict}

        if keys == None and thistype is dict:
            keys = [key for key, val in complexdict.items()]
        flatdict = {}

        for key in keys:
            try:
                val = complexdict[key]
            except:
                val = 'no val found'
            newdict = self.myflatdict(val)
            for key2, val2 in newdict.items():

                flatdict[f'{key}:{key2}'] = [val2]

        return flatdict


    def merge_and_condense_saved_models(self,merge_directory=None,save_directory=None,condense=None,verbose=None):
        if not merge_directory==None:
            if not os.path.exists(merge_directory):
                print(f'could not find merge_directory:{merge_directory}')
                self.logger.error(f'could not find merge_directory:{merge_directory}')
                return
        else:
            merge_directory=self.kc_savedirectory
                
        if not save_directory==None:
            assert os.path.exists(save_directory),f"save_directory does not exist:{save_directory}"
        else:
            save_directory=self.kc_savedirectory
                #os.makedirs(save_directory)
        if condense==None or condense=='no':
            condense=0
        if condense=='yes':
            condense=1
        if verbose==None or verbose=='no':
            verbose=0
        if verbose=='yes':
            verbose=1
        model_save_filelist=[name_i for name_i in os.listdir(merge_directory) if re.search('model_save',name_i)]
        #p#rint('here',model_save_filelist)
        

        condensedfilename=os.path.join(save_directory,'condensed_model_save')
        try:
            with open(condensedfilename,'rb') as savedfile:
                saved_condensed_list=pickle.load(savedfile)
        except: 
            if verbose==1:
                self.logger.info("couldn't open condensed_model_save in save_directory, trying self.kc_savedirectory")

            condensedfilename=os.path.join(self.kc_savedirectory,'condensed_model_save')
            try:
                with open(condensedfilename,'rb') as savedfile:
                    saved_condensed_list=pickle.load(savedfile)
            except:
                saved_condensed_list=[]
                if verbose==1:
                    self.logger.info("---------no existing files named condensed_model_save could be found. "
                        "if it is in merge_directory, it will be picked up and merged anyways--------")
        



        if len(model_save_filelist)==0:
            self.logger.warning('0 models found in save_directory:{merge_directory} when merging')
            return
                         
        #if len(model_save_filelist)==1 and saved_condensed_list==[]:
        #    print('only 1 model_save file found, and saved_condensed_list is empty, so no merge completed')
        #    return
        
        list_of_saved_models=[]

        for file_i in model_save_filelist:
            file_i_name=os.path.join(merge_directory,file_i)
            for i in range(10):
                with open(file_i_name,'rb') as savedfile:
                    try: 
                        saved_model_list=pickle.load(savedfile)
                        if verbose==1:
                            print(f'file_i:{file_i_name} has {len(saved_model_list)} saved model(s)')
                        break
                    except:
                        if i==9:
                            print(f'warning!saved_model_list{file_i_name} could not pickle.load')
                            self.logger.exception(f'error in {__name__}')
                            saved_model_list=[]

            if condense==1:
                list_of_saved_models.extend(self.condense_saved_model_list(saved_model_list, help_start=0, strict=1,verbose=verbose))
            else:
                list_of_saved_models.extend(saved_model_list)
        if not saved_condensed_list==[]:
            if condense==1:
                saved_condensed_list=self.condense_saved_model_list(saved_condensed_list,help_start=0,strict=1,verbose=verbose)
            list_of_saved_models.extend(saved_condensed_list)
        if condense==1:
            list_of_saved_models=self.condense_saved_model_list(list_of_saved_models,help_start=0,strict=1,verbose=verbose)


        condensedfilename=os.path.join(save_directory,'condensed_model_save')
        with open(condensedfilename,'wb') as newfile:
            print(f'writing new_model_list length:{len(list_of_saved_models)} to newfile:{newfile}')
            pickle.dump(list_of_saved_models,newfile)

                                              
    def condense_saved_model_list(self,saved_model_list,help_start=1,strict=None,verbose=None):
        if saved_model_list==None:
            return []
        if verbose=='yes': verbose=1
        if verbose==None or verbose=='no':verbose=0
        assert type(verbose)==int, f'type(verbose) should be int but :{type(verbose)}'
        if strict=='yes':strict=1
        if strict=='no':strict=0
        modelcount=len(saved_model_list)
        keep_model=[1 for _ in range(modelcount)]
        for i,full_model_i in enumerate(saved_model_list):
            if verbose>0:
                print(f'{100*i/modelcount}%',end=',')
                
            if keep_model[i]==1:
                for j,full_model_j in enumerate(saved_model_list[i+1:]):
                    j=j+i+1
                    if keep_model[j]==1:
                        matchlist=self.do_partial_match([full_model_i],full_model_j,help_start=help_start,strict=strict)
                        #if full_model_i['modeldict']==full_model_j['modeldict']:
                        if len(matchlist)>0:
                            i_mse=full_model_i['mse']
                            j_mse=full_model_j['mse']
                            try:
                                i_naivemse=full_model_i['naivemse']
                                j_naivemse=full_model_j['naivemse']
                            except:
                                i_naivemse=1
                                j_naivemse=1
                            try:
                                i_n=full_model_i['datagen_dict']['train_n']
                                i_batchcount=1                               
                            except:
                                i_n=full_model_i['datagen_dict']['batch_n']
                                i_batchcount=full_model_i['datagen_dict']['batchcount']
                            try:
                                j_n=full_model_j['datagen_dict']['train_n']
                                j_batchcount=1                                  
                            except:
                                j_n=full_model_j['datagen_dict']['batch_n']
                                j_batchcount=full_model_j['datagen_dict']['batchcount']

                            try:
                                j_batchbatch1=full_model_j['datagen_dict']['batchbatchcount']
                                j_batchbatch2=full_model_j['modeldict']['maxbatchbatchcount']
                                if j_batchbatch1<j_batchbatch2:
                                    j_batchbatchcount=j_batchbatch1
                                else: j_batchbatchcount=j_batchbatch2
                            except: j_batchbatchcount=1
                            try:
                                i_batchbatch1=full_model_i['datagen_dict']['batchbatchcount']
                                i_batchbatch2=full_model_i['modeldict']['maxbatchbatchcount']
                                if i_batchbatch1<i_batchbatch2:
                                    i_batchbatchcount=i_batchbatch1
                                else: i_batchbatchcount=i_batchbatch2
                            except: i_batchbatchcount=1
                                
                                
                            iwt=self.do_nwt_mse(i_mse,i_n,i_batchcount,naivemse=i_naivemse,batchbatchcount=i_batchbatchcount)
                            jwt=self.do_nwt_mse(j_mse,j_n,j_batchcount,naivemse=j_naivemse,batchbatchcount=j_batchbatchcount)
                            if verbose>1:
                                print(f'i_mse:{i_mse},i_n:{i_n},iwt:{iwt},j_mse:{j_mse},j_n:{j_n},jwt:{jwt}')

                            if iwt<jwt:
                                if verbose>1:
                                    print('model j loses')
                                keep_model[j]=0
                            else:
                                if verbose>1:
                                    print('model i loses')
                                keep_model[i]=0
                                break
                    
        final_keep_list=[model for i,model in enumerate(saved_model_list) if keep_model[i]==1]
        if verbose>0:
            print(f'len(final_keep_list):{len(final_keep_list)}')
        return final_keep_list

    def do_nwt_mse(self,mse,n,batch_count=1,naivemse=1,batchbatchcount=1):
        batch_count=batch_count*batchbatchcount
        if not type(mse) is float:
            return 10**301
        
        else:
            #p#rint('type(mse)',type(mse))
            return np.log(mse/naivemse+1)/(np.log(n**2*batch_count)**1.5)
    
    def pull2dicts(self,optimizedict):
        return {'modeldict':optimizedict['modeldict'],'datagen_dict':optimizedict['datagen_dict']}
    
    def open_and_compare_optdict(self,saved_filename,optimizedict,help_start=None,partial_match=None):
        if help_start==None or help_start=='no': 
            help_start=0
        if help_start=='yes': 
            help_start=1
        if partial_match==None or partial_match=='no':
            partial_match=0
        if partial_match=='yes': 
            partial_match=1
         
        assert type(saved_filename) is str, f'saved_filename expected to be string but is type:{type(saved_filename)}'
        try:    
            with open(saved_filename,'rb') as saved_model_bytes:
                saved_dict_list=pickle.load(saved_model_bytes)
                #p#rint(f'from filename:{saved_filename}, last in saved_dict_list:{saved_dict_list[-1]["modeldict"]}')
                #p#rint(f'optimizedict["modeldict"]:{optimizedict["modeldict"]}')
        except:
            #self.logger.exception(f'error in {__name__}')
            self.logger.info(f'saved_filename is {saved_filename}, but does not seem to exist')
            return []
        #saved_dict_list=[model for model in saved_model]
        
        this_2dicts=self.pull2dicts(optimizedict)
        #p#rint(saved_filename)
        #p#rint(f'saved_dict_list has first item of:{type(saved_dict_list[0])}')
        doubledict_list=[self.pull2dicts(dict_i) for dict_i in saved_dict_list]
        print(f'in saved_filename:{saved_filename}, len(doubledict_list):{len(doubledict_list)},len(saved_dict_list):{len(saved_dict_list)}')
        doubledict_match_list_select=[self.are_dicts_equal(dict_i,this_2dicts) for dict_i in doubledict_list]#list of boolean
        doubledict_match_list=[saved_dict_list[i] for i,ismatch in enumerate(doubledict_match_list_select) if ismatch]
        
        
        if help_start==1 and len(doubledict_match_list)==0:
            print('--------------------------------help_start is triggered---------------------------')
            doubledict_match_list=self.do_partial_match(saved_dict_list,optimizedict,help_start=1, strict=1)
            #p#rint('len(doubledict_match_list)',len(doubledict_match_list))
            
            if len(doubledict_match_list)>0:
                return self.condense_saved_model_list(doubledict_match_list,help_start=0,strict=1)
        if len(doubledict_match_list)==0 and help_start==1 and partial_match==1:
            print('--------------help_start with partial match triggered----------------')
            tryagain= self.condense_saved_model_list(self.do_partial_match(saved_dict_list,optimizedict,help_start=1,strict=0))
            print('here3')
            if not tryagain==None:
                return tryagain
        else:
            print('here4')
            #same_doubledict_list=[saved_dict_list[i] for i,optdict_i in enumerate(saved_dict_list) if \
            #                      self.pull2dicts(optdict_i)==this_2dicts]
                                # optdict['modeldict']==optimizedict['modeldict'] and\
                                # optdict['datagen_dict']==optimizedict['datagen_dict'] ]
            same_modeldict_list=[saved_dict_list[i] for i,optdict_i in enumerate(saved_dict_list) if \
                                 self.are_dicts_equal(optdict_i['modeldict'],optimizedict['modeldict'])]
            #same_doubledict_list=[saved_dict_list[i] for i,is_same in enumerate(modeldict_compare_list) if is_same]
            #xcompare_list=[np.all(dict_i['xdata']==x) for dict_i in doubledict_match_list]
            #same_model_and_x_dict_list=[doubledict_match_list[i] for i,is_same in enumerate(xcompare_list) if is_same]
            #ycompare_list=[np.all(dict_i['ydata']==y) for dict_i in same_model_and_x_dict_list]
            #same_modelxy_dict_list=[same_model_and_x_dict_list[i] for i,is_same in enumerate(ycompare_list) if is_same]
            if len(doubledict_match_list)==0 and partial_match==1:
                print('found matching models but no matching datagen_dict')
                return same_modeldict_list


            return doubledict_match_list
    
    def do_partial_match(self,saved_optdict_list,afullmodel,help_start,strict=None):
        if strict==None or strict=='no':
            strict=0
        if strict=='yes': strict=1
                
        adoubledict=self.pull2dicts(afullmodel)
        saved_doubledict_list=[self.pull2dicts(dict_i) for dict_i in saved_optdict_list]
        #if type(adoubledict) is list: print(adoubledict)
        #lists=[adoubledict_i for adoubledict_i in saved_doubledict_list if type(adoubledict_i) is list]
        #if len(lists)>0:print(lists)
        same_model_datagen_compare=[self.are_dicts_equal(adoubledict,dict_i) for dict_i in saved_doubledict_list]
        
        matches=[item for i,item in enumerate(saved_optdict_list) if same_model_datagen_compare[i]]
        matchcount=len(matches)
        if strict==1:
            #keys=[key for key,val in afullmodel.items()]
            #p#rint(f'keys:{keys}')
            return matches
        
        if not matchcount<help_start:
            return matches
        print('-----partial match is looking for a partial match------')
        new_dict_list=[]
        #datagen_dict={'train_n':60,'n':200, 'param_count':2,'seed':1, 'ftype':'linear', 'evar':1}
        string_list=[('modeldict','ykern_grid'),('modeldict','maxbatchbatchcount'),('datagen_dict','batchbatchcount'),('datagen_dict','batchcount'),('datagen_dict','seed'),('datagen_dict','batch_n'),
                     ('modeldict','species_n'),('modeldict','spatialtransform'),('modeldict','species'),('modeldict','ykern_grid'),
                     ('modeldict','xkern_grid'),('datagen_dict','batchcount'),('datagen_dict','evar'),('modeldict','hyper_param_form_dict'),('modeldict','regression_model'),
                     ('modeldict','loss_function'),('modeldict','NWnorm'),('modeldict','ykerngrid_form'),('modeldict','logic_date')]
        for string_tup in string_list:
            try:
                sub_value=adoubledict[string_tup[0]][string_tup[1]]
                new_dict_list.append({string_tup[0]:{string_tup[1]:sub_value}})#make the list match amodeldict, so optimization settings aren't changed
            except: pass
        #new_dict_list.append(amodeldict['xkern_grid'])
        #new_dict_list.append(amodeldict['hyper_param_form_dict'])
        #new_dict_list.append(amodeldict['regression_model'])
        
        simple_doubledict_list=deepcopy(saved_doubledict_list)#added deepcopy abovedeepcopy(saved_doubledict_list)#initialize these as copies that will be progressively simplified
        #simple_adoubledict=deepcopy(adoubledict)
        for new_dict in new_dict_list:
            #p#rint(f'partial match trying {new_dict}')
            try: 
                simple_doubledict_list=[self.do_dict_override(dict_i,new_dict) for dict_i in simple_doubledict_list]
                matchlist_idx=[self.are_dicts_equal(adoubledict,dict_i) for dict_i in simple_doubledict_list]
                matchlist=[dict_i for i,dict_i in enumerate(saved_optdict_list) if matchlist_idx[i]]
                if len(matchlist)>0:
                    print(f'{len(matchlist)} partial matches found only after substituting {new_dict}')
                    break
            except: 
                self.logger.warning('dict override failed for dict_i:{dict_i}')
        if len(matchlist)==0:
            print(f'partial_match could not find any partial matches')
        return matchlist
    
    def are_dicts_equal(self,dict1,dict2):
        for key,val1 in dict1.items():
            if not key in dict2:
                return False
            
            val2=dict2[key]
            type1=type(val1);type2=type(val2)
            if not type1==type2:
                return False
            if type(val1) is dict:
                if not self.are_dicts_equal(val1,val2):
                    return False
            else:
                if type(val1) is np.ndarray:
                    if not np.array_equal(val1,val2):
                        return False
                else:
                    try: 
                        if not val1==val2:
                            return False
                    except: 
                        print('type(val1),type(val2)',type(val1),type(val2))
                        self.logger.exception(f'error in {__name__}')
                        assert False,""
                        
        for key,_ in dict2.items():
            if not key in dict1:
                return False
        return True
            
                
                  
    def do_dict_override(self,old_dict,new_dict,verbose=None,recursive=None,replace=None,deletekey=None):#key:values in old_dict replaced by any matching keys in new_dict, otherwise old_dict is left the same and returned.
        old_dict_copy=deepcopy(old_dict)
        if replace==None or replace=='yes':
            replace=1
        if replace=='no':replace=0#i.e., for adding a key:val that was missing
        if verbose==None or verbose=='no':
            verbose=0
        if verbose=='yes':
            verbose=1

        if deletekey=='yes':deletekey=1
        if deletekey==None:deletekey=='no'

        vstring=''
        if new_dict==None or new_dict=={}:
            if verbose==1:
                print(f'vstring:{vstring}, and done1')
            return old_dict_copy
        for key,val in new_dict.items():
            if verbose==1:
                vstring=vstring+f":key({key})"
            if type(val) is dict:
                if verbose==1:print(f'val is dict in {key}, recursive call')
                old_dict_copy[key],vstring2=self.do_dict_override(old_dict_copy[key],val,recursive=1,verbose=verbose,replace=replace)
                vstring=vstring+vstring2
                #p#rint('made it back from recursive call')
            elif type(val) is None and deletekey==1:
                try: 
                    old_dict_copy.pop(key)
                    if verbose==1:
                          vstring = vstring + f"deleted:{key}:{val}"
                except KeyError:
                    if verbose==1:
                          vstring = vstring + f"delete could not find key:{key}"
                        
            
            elif replace==0 and (key in old_dict_copy):
                if verbose == 1:
                    print(f":val({new_dict[key]}) does not replace val({old_dict_copy[key]}) because replace={replace}\n")
                    vstring = vstring + f":for key:{key}, val({val}) does not replace val({old_dict_copy[key]})\n"
                else:
                    pass#no replacement due to above condition
            else:
                try:
                    if key in old_dict_copy:
                        oldval=old_dict_copy[key]
                    else:
                        oldval=f'{key} not in old_dict'
                    old_dict_copy[key]=val
                    if verbose==1:
                        print(f":val({new_dict[key]}) replaces val({oldval})\n")
                        vstring=vstring+f":for key:{key}, val({val}) replaces val({oldval})\n"

                except:
                    print(f'Warning: old_dict has keys:{[key for key,value in old_dict_copy.items()]} and new_dict has key:value::{key}:{new_dict[key]}')
        if verbose==1:
                print(f'vstring:{vstring} and done2')            
        if recursive==1:
            return old_dict_copy, vstring

        else:
            #p#rint(f'old_dict_copy{old_dict_copy}')
            return old_dict_copy
    

        
class KernelCompare(KernelOptModelTools,KernelParams):
    def __init__(self,directory=None,myname=None, source=None):
        if source is None:
            source='monte'
        self.source=source
        self.name=myname
        if directory==None:
            self.kc_savedirectory=os.getcwd()
            merge_directory=self.kc_savedirectory
        else: 
            self.kc_savedirectory=directory
            merge_directory=os.path.join(self.kc_savedirectory,'..')

        KernelOptModelTools.__init__(self,directory=self.kc_savedirectory,myname=myname)
        KernelParams.__init__(self,)

    def prep_model_list(self, optdict_variation_list=None,datagen_variation_list=None,datagen_dict=None,verbose=0):
        if not type(datagen_dict) is dict:
            if datagen_dict is None:
                try: datagen_dict=self.datagen_dict
                except:
                    print('initializing monte carlo datagen_dict')
                    self.logger.warning('initializing monte carlo datagen_dict')
                    param_count=2
                    datagen_dict={'validate_batchcount':10,'batch_n':32,'batchcount':10, 'param_count':param_count,'seed':1, 'ftype':'linear', 'evar':1, 'source':'monte'}
            else:assert False,f'datagen_dict:{datagen_dict}'
        
        
        
        if datagen_variation_list is None:
            datagen_variation_list=[{}]#will default to parameters in datagen_dict above
        if datagen_dict['source']=='pisces':
            speciesvarfound=0
            for datagenvar in datagen_variation_list:
                if datagenvar[0]=='species':
                    speciesvarfound=1
            if speciesvarfound==0 and datagen_dict['species']=='all':
                print('adding all species variations')
                datagen_variation_list=self.addspeciesvariations(datagen_variation_list)
                
        '''assert type(datagen_variation_list)==list,f'datagen_variation_list type:{type(datagen_variation_list)} but expected a list'
        assert type(datagen_variation_list[0])==tuple,f'first item of datagen_variation_list type:{type(datagen_variation_list[0])} but expected a tuple'
                        
        assert type(optdict_variation_list)==list,f'optdict_variation_list type:{type(optdict_variation_list)} but expected a list'
        assert type(optdict_variation_list[0])==tuple,f'first item of optdict_variation_list type:{type(optdict_variation_list[0])} but expected a tuple'
        '''
        
        
        model_run_dict_list=[]
        #print(f'datagen_dict:{datagen_dict}, datagen_variation_list:{datagen_variation_list}')
        datagen_dict_list=self.build_dict_variations(datagen_dict,datagen_variation_list,verbose=1)
        print(f'len(datagen_dict_list):{len(datagen_dict_list)}')
        for alt_datagen_dict in datagen_dict_list:
            if 'species' in alt_datagen_dict:
                species=alt_datagen_dict['species']
            else: species=None
            initial_opt_dict=self.build_optdict(param_count=alt_datagen_dict['param_count'],species=species)
            optdict_list=self.build_dict_variations(initial_opt_dict,optdict_variation_list)    
            for optdict_i in optdict_list:
                #rebuild hyper_param_start_values since variations may change array length.
                #newhyper_paramdict=self.build_hyper_param_start_values(optdict_i['modeldict'])
                #optdict_i['hyper_param_dict']=newhyper_paramdict
                
                optmodel_run_dict={'optimizedict':optdict_i,'datagen_dict':alt_datagen_dict}    
                model_run_dict_list.append(optmodel_run_dict)
                #p#rint('model_run_dict_list:',model_run_dict_list)
        return model_run_dict_list
    
    def addspeciesvariations(self,datagen_variation_list):
        #remove existing species variations in case species:all is included in the list
        
        datagen_variation_list=[tup_i for tup_i in datagen_variation_list if tup_i[0]!='species']
        
         
        try:self.specieslist
        except:
            pdh12=dg.PiscesDataTool()
            self.specieslist=pdh12.returnspecieslist()
            
        
        species_variations=('species',self.specieslist)
        #p#rint(f'before-datagen_variation_list:{datagen_variation_list}')
        datagen_variation_list.append(species_variations)
        #p#rint(f'after-datagen_variation_list:{datagen_variation_list}')
        return datagen_variation_list
                         
    def run_model_as_node(self,optimizedict,datagen_dict,force_start_params=None):
        self.do_monte_opt(optimizedict,datagen_dict,force_start_params=force_start_params)
        return
        

    def build_dict_variations(self,initial_dict,variation_list,verbose=1):
        dict_combo_list=[]
        sub_list=[]
        dict_ik=deepcopy(initial_dict)

        #pull and replace first value from each variation
        for tup_i in variation_list:
            override_dict_ik=self.build_override_dict_from_str(tup_i[0],tup_i[1][0])
            #print('override_dict_ik',override_dict_ik)
            dict_ik=self.do_dict_override(dict_ik,override_dict_ik)
        dict_combo_list.append(dict_ik)#this is now the starting dictionary.
        remaining_variation_list=[(tup_i[0],tup_i[1][1:]) for tup_i in variation_list if len(tup_i[1])>1]
        for tup_i in remaining_variation_list:
            additions=[]
            for val in tup_i[1]:
                for dict_i in dict_combo_list:
                    override_dict=self.build_override_dict_from_str(tup_i[0],val)
                    #print(override_dict)
                    newdict=self.do_dict_override(dict_i,override_dict)
                    additions.append(newdict)
            dict_combo_list=dict_combo_list+additions
        '''if verbose==1:
            print(f'dict_combo_list has {len(dict_combo_list)} variations to run')'''
        return dict_combo_list    
            
    '''def build_dict_variations(self,initial_dict,variation_list):
        dict_combo_list=[]
        for i,tup_i in enumerate(variation_list):
            sub_list=[not_tup_i for j,not_tup_i in enumerate(variation_list) if not j==i]
            for k,val in enumerate(tup_i[1]):
                override_dict_ik=self.build_override_dict_from_str(tup_i[0],val)
                dict_ik=self.do_dict_override(initial_dict,override_dict_ik)
                #p#rint('dict_combo_list',dict_combo_list)
                dict_combo_list.append(dict_ik)
                if len(sub_list)>0:
                    new_items=self.build_dict_variations(dict_ik,sub_list)
                    dict_combo_list=dict_combo_list+new_items
                else: 
                    return dict_combo_list 
        return dict_combo_list'''

                         
    def build_override_dict_from_str(self,dict_string,val):
        colon_loc=[i for i,char in enumerate(dict_string) if char==':']
        #print(f'string_address: {dict_string}, colon_loc: {colon_loc}')
        outdict=self.recursive_string_dict_helper(dict_string,colon_loc,val)
        return outdict
          
                         
    def recursive_string_dict_helper(self,dict_string,colon_loc,val):
        if len(colon_loc)==0:
            return {dict_string:val}
        if len(colon_loc)>0:
            key=dict_string[0:colon_loc[0]]
            newcolon_loc=[i-(colon_loc[0]+1) for i in colon_loc[1:]]#+1 b/c we removed that many characters plus 1 since zero was included
            val=self.recursive_string_dict_helper(dict_string[colon_loc[0]+1:],newcolon_loc,val)
            outdict={key:val}
            return outdict
        
                         
    '''def build_quadratic_datagen_dict_override(self):
        datagen_dict_override={}
        datagen_dict_override['ftype']='quadratic'
        return datagen_dict_override'''
        
                         

                               
                               
if __name__=='__main__':
    import kernelcompare as kc
    kc_obj=kc.KernelCompare()
