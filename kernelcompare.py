from copy import deepcopy
from time import strftime
import numpy as np
import pickle
import os
import datagen as dg
import mykern as mk
import re
import traceback
#import datetime

class KernelOptModelTools:
    def __init__(self,directory=None):
        if directory==None:
            self.kc_savedirectory=os.getcwd
        else:
            self.kc_savedirectory=directory
        pass
        
    def do_monte_opt(self,optimizedict,datagen_dict,force_start_params=None):
        optimizedict['datagen_dict']=datagen_dict
        
        if force_start_params==None or force_start_params=='no':
            force_start_params=0
        if force_start_params=='yes':
            force_start_params=1


        data_dict=self.build_dataset_dict(datagen_dict)
        y=data_dict['train_y']
        x=data_dict['train_x']
        
        if force_start_params==0:
            optimizedict=self.run_opt_complete_check(optimizedict,y,x,replace=1)
        
        
        start_msg=f'starting at {strftime("%Y%m%d-%H%M%S")}'
        
        mk.optimize_free_params(y,x,optimizedict)
        return
        
    def run_opt_complete_check(self,optimizedict_orig,y,x,replace=None):
        '''
        checks model_save and then final_model_save to see if the same modeldict has been run before (e.g.,
        same model featuers, same starting parameters, same data).
        -by default or if replace set to 1 or 'yes', then the start parameters are replaced by the matching model from model_save 
            and final_model_save with the lowest mse
        -if replace set to 0 or 'no', then the best matching model is still announced, but replacement of start parameters doens't happen
        '''
        optimizedict=optimizedict_orig.copy()
        if replace==None or replace=='yes':
            replace=1
        if replace=='no':
            replace=0
        best_dict_list=[]
        help_start=optimizedict['opt_settings_dict']['help_start']
        #print(f'help_start:{help_start}')
        partial_match=optimizedict['opt_settings_dict']['partial_match']
        try:
            same_modelxy_dict_list=self.open_and_compare_optdict('condensed_model_save',optimizedict,y,x,help_start=help_start,partial_match=partial_match)
        except:
            print(traceback.format_exc())
            same_modelxy_dict_list=self.open_and_compare_optdict('model_save',optimizedict,y,x,help_start=help_start,partial_match=partial_match)
        if len(same_modelxy_dict_list)>0:
            #print(f"from model_save, This dictionary, x,y combo has finished optimization before:{len(same_modelxy_dict_list)} times")
            #print(f'first item in modelxy_dict_list:{same_modelxy_dict_list[0]}'')
            mse_list=[dict_i['mse'] for dict_i in same_modelxy_dict_list]
            train_n=[dict_i['ydata'].shape[0] for dict_i in same_modelxy_dict_list]
            n_wt_mse_list=[self.do_nwt_mse(mse_list[i],train_n[i]) for i in range(len(mse_list))]
            lowest_n_wt_mse=min(n_wt_mse_list)
            
            best_dict_list.append(same_modelxy_dict_list[n_wt_mse_list.index(lowest_n_wt_mse)])
        if len(same_modelxy_dict_list)==0:
            print('--------------no matching models found----------')
        
        mse_list=[dict_i['mse'] for dict_i in best_dict_list]
        if len(mse_list)>0:
            mse_list=[dict_i['mse'] for dict_i in best_dict_list]
            train_n=[dict_i['ydata'].shape[0] for dict_i in best_dict_list]
            n_wt_mse_list=[self.do_nwt_mse(mse_list[i],train_n[i]) for i in range(len(mse_list))]
            lowest_n_wt_mse=min(n_wt_mse_list)
            best_dict=best_dict_list[n_wt_mse_list.index(lowest_n_wt_mse)]
            
            try:print(f'optimization dict with lowest mse:{best_dict["mse"]}, n:{best_dict["ydata"].shape[0]}was last saved{best_dict["whensaved"]}')
            except:print(f'optimization dict with lowest mse:{best_dict["mse"]}, n:{best_dict["ydata"].shape[0]}was last saved{best_dict["when_saved"]}')
            print(f'best_dict:{best_dict}')
            if replace==1:
                print("overriding start parameters with saved parameters")
                self.rebuild_hyper_param_dict(optimizedict,best_dict['params'],verbose=0)
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
            new_val=mk.kNdtool.pull_value_from_fixed_or_free(self,key,replacement_fixedfreedict,transform='no')
            vstring+=f"for {key} old val({val})replaced with new val({new_val})"
            new_opt_dict['hyper_param_dict'][key]=new_val
        #print(f'rebuild hyper param dict vstring:{vstring}')
        return new_opt_dict
                  
    def open_condense_resave(self,filename1,verbose=None):#not calling this. delete?
        if verbose==None or verbose=='no':
            verbose=0
        if verbose=='yes':
            verbose=1

        try:
            with open(filename1,'rb') as savedfile:
                saved_model_list1=pickle.load(savedfile)
            condensed_list=self.condense_saved_model_list(saved_model_list1, help_start=0, strict=1,verbose=verbose)
        except:
            print(traceback.format_exc())
            print(f'filename:{filename1} not found')
            return
        try:
            with open(filename1,'wb') as writefile:
                pickle.dump(condensed_list,writefile)
        except:
                print(f'filewrite for filename:{filename1} failed')
                
            
    def merge_and_condense_saved_models(self,merge_directory=None,save_directory=None,condense=None,verbose=None):
        if not merge_directory==None:
            assert os.path.exists(merge_directory),"merge_directory does not exist"
        else:
            merge_directory=self.kc_savedirectory
                
        if not save_directory==None:
            assert os.path.exists(save_directory),"save_directory does not exist"
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
        print('here',model_save_filelist)
        
        os.chdir(save_directory)

        try:
            with open('condensed_model_save','rb') as savedfile:
                saved_condensed_list=pickle.load(savedfile)
        except: 
            if verbose==1:
                print("couldn't open condensed_model_save in save_directory, trying self.kc_savedirectory")
            os.chdir(self.kc_savedirectory)
            try:
                with open('condensed_model_save','rb') as savedfile:
                    saved_condensed_list=pickle.load(savedfile)
            except:
                saved_condensed_list=[]
                if verbose==1:
                    print("---------no existing files named condensed_model_save could be found. "
                        "if it is in merge_directory, it will be picked up and merged anyways--------")
        
        os.chdir(self.kc_savedirectory)


        if len(model_save_filelist)==0:
            print('no models found when merging')
            return
        #if len(model_save_filelist)==1 and saved_condensed_list==[]:
        #    print('only 1 model_save file found, and saved_condensed_list is empty, so no merge completed')
        #    return
        new_model_list=[]
        list_of_saved_lists=[]
        if len(model_save_filelist)>0:
            for file_i in model_save_filelist:
                with open(file_i,'rb') as savedfile:
                    try: 
                        saved_model_list=pickle.load(savedfile)
                        if verbose==1:
                            print(f'file_i:{file_i} has {len(file_i)} saved model(s)')
                    except:
                        if verbose==1:
                            print(f'warning!saved_model_list{file_i} could not pickle.load')
                        
                if condense==1:
                    list_of_saved_lists.append(self.condense_saved_model_list(saved_model_list, help_start=0, strict=1,verbose=verbose))
                else:
                    list_of_saved_lists.append(saved_model_list)
            if not saved_condensed_list==[]:
                list_of_saved_lists.append(saved_condensed_list)
            for i,list_i in enumerate(list_of_saved_lists):#each saved_list is from a different file, and they have been put in a list
                for list_j in list_of_saved_lists[i:]:#no plus 1, so compare list_i to itself, but not ones before it.
                    doubledict_list_i=[self.pull2dicts(dict_i) for dict_i in list_i]
                    doubledict_list_j=[self.pull2dicts(dict_j) for dict_j in list_j]
                    #atagen_dict_list_i=[dict_i['datagen_dict'] for dict_i in list_i]
                    #datagen_dict_list_j=[dict_i['datagen_dict'] for dict_i in list_j]
                    
                    
        #matching_doubledict_list[dict_1==dict_2 for dict_1 in doubledict_list_i for dict_2 in doubledict_list_j]
                    jlen=len(list_j)
                    ilen=len(list_i)
                    jbest=[1]*jlen

                    for ii in range(ilen):
                        ibest=1#start optimistic
                        for jj in range(jlen):
                            both_dicts_match=doubledict_list_i[ii]==doubledict_list_j[jj]# and datagen_dict_list_i[ii]==datagen_dict_list_j[jj]
                            if both_dicts_match:
                                if list_i[ii]['mse']>list_j[jj]['mse']:
                                    ibest=0
                                else:
                                    jbest[jj]=0
                        if ibest==1:#only true if dict_1[i] never lost, because it won or was unique
                            new_model_list.append(list_i[ii])
                    for k in range(jlen):
                        if jbest[k]==1:
                            new_model_list.append(list_j[k])
        
        os.chdir(save_directory)
        with open('condensed_model_save','wb') as newfile:
            #print(f'list_i:{new_model_list}')
            pickle.dump(new_model_list,newfile)
        os.chdir(self.kc_savedirectory)
                                              
    def condense_saved_model_list(self,saved_model_list,help_start=1,strict=None,verbose=None):
        if saved_model_list==None:
            return []
        if verbose==None or verbose=='yes': verbose=1
        if verbose=='no':verbose=0
        if strict=='yes':strict=1
        if strict=='no':strict=0
        keep_model=[1]*len(saved_model_list)
        for i,full_model_i in enumerate(saved_model_list):
            if keep_model[i]==1:
                for j,full_model_j in enumerate(saved_model_list[i+1:]):
                    j=j+i+1
                    matchlist=self.do_partial_match([full_model_i],full_model_j,help_start=help_start,strict=strict)
                    #if full_model_i['modeldict']==full_model_j['modeldict']:
                    if len(matchlist)>0:
                        i_mse=full_model_i['mse']
                        i_n=full_model_i['ydata'].shape[0]
                        j_mse=full_model_j['mse']
                        j_n=full_model_j['ydata'].shape[0]
                        iwt=self.do_nwt_mse(i_mse,i_n)
                        jwt=self.do_nwt_mse(j_mse,j_n)
                        if verbose==1:
                            print(f'i_mse:{i_mse},i_n:{i_n},iwt:{iwt},j_mse:{j_mse},j_n:{j_n},jwt:{jwt}')

                        if iwt<jwt:
                            if verbose==1:
                                print('model j loses')
                            keep_model[j]=0
                        else:
                            if verbose==1:
                                print('model i loses')
                            keep_model[i]=0
                    
        final_match_list=[model for i,model in enumerate(saved_model_list) if keep_model[i]==1]
        print(f'len(final_match_list):{len(final_match_list)}')
        return final_match_list

    def do_nwt_mse(self,mse,n):
        return np.log(mse+1)/(np.log(n)**3)
    
    def pull2dicts(self,optimizedict):
        return {'modeldict':optimizedict['modeldict'],'datagen_dict':optimizedict['datagen_dict']}
    
    def open_and_compare_optdict(self,saved_filename,optimizedict,y,x,help_start=None,partial_match=None):
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
                #print(f'from filename:{saved_filename}, last in saved_dict_list:{saved_dict_list[-1]["modeldict"]}')
                #print(f'optimizedict["modeldict"]:{optimizedict["modeldict"]}')
        except:
            print(f'saved_filename is {saved_filename}, but does not seem to exist')
            return []
        #saved_dict_list=[model for model in saved_model]
        
        this_2dicts=self.pull2dicts(optimizedict)
        #print(saved_filename)
        #print(f'saved_dict_list has first item of:{type(saved_dict_list[0])}')
        doubledict_list=[self.pull2dicts(dict_i) for dict_i in saved_dict_list]
        print(f'in saved_filename:{saved_filename}, len(doubledict_list):{len(doubledict_list)},len(saved_dict_list):{len(saved_dict_list)}')
        optdict_match_list_select=[dict_i==this_2dicts for dict_i in doubledict_list]#list of boolean
        optdict_match_list=[saved_dict_list[i] for i,ismatch in enumerate(optdict_match_list_select) if ismatch]
        
        
        if help_start==1 and len(optdict_match_list)>0:
            print('--------------------------------help_start is triggered---------------------------')
            optdict_match_list=self.do_partial_match(saved_dict_list,optimizedict,help_start=1, strict=1)
            #print(f'optdict_match_list2:{optdict_match_list}')
            return self.condense_saved_model_list(optdict_match_list,help_start=0,strict=1)
        elif len(optdict_match_list)==0 and partial_match==1:
            print('--------------here----------------')
            tryagain= self.condense_saved_model_list(self.do_partial_match(saved_dict_list,optimizedict,help_start=1,strict=0))
            if tryagain==None:return []
            else:return tryagain
        else:
            #same_doubledict_list=[saved_dict_list[i] for i,is_same in enumerate(modeldict_compare_list) if is_same]
            xcompare_list=[np.all(dict_i['xdata']==x) for dict_i in optdict_match_list]
            same_model_and_x_dict_list=[optdict_match_list[i] for i,is_same in enumerate(xcompare_list) if is_same]
            ycompare_list=[np.all(dict_i['ydata']==y) for dict_i in same_model_and_x_dict_list]
            same_modelxy_dict_list=[same_model_and_x_dict_list[i] for i,is_same in enumerate(ycompare_list) if is_same]
            if len(same_modelxy_dict_list)==0 and partial_match==1:
                if len(same_modelxy_dict_list)>0: 
                    print('found matching models and matching datagen_dict but no matching x,y too')
                    return same_modelxy_dict_list
                else: 
                    print('found matching models but not matching x,y')
                    return optdict_match_list
            print('')
            return same_modelxy_dict_list
    
    def do_partial_match(self,saved_optdict_list,afullmodel,help_start,strict=None):
        if strict==None or strict=='no':
            strict=0
        if strict=='yes': strict=1
        adoubledict=self.pull2dicts(afullmodel)
        saved_doubledict_list=[self.pull2dicts(dict_i) for dict_i in saved_optdict_list]
        same_model_datagen_compare=[adoubledict==dict_i for dict_i in saved_doubledict_list]
        
        matches=[item for i,item in enumerate(saved_optdict_list) if same_model_datagen_compare[i]]
        matchcount=len(matches)
        if strict==1:
            #keys=[key for key,val in afullmodel.items()]
            #print(f'keys:{keys}')
            return matches
        
        if matchcount>help_start*2:
            print(f'partial match found a full match.....matchcount:{matchcount}')
            return matches
        
        new_dict_list=[]
        #datagen_dict={'train_n':60,'n':200, 'param_count':2,'seed':1, 'ftype':'linear', 'evar':1}
        string_list=[('datagen_dict','seed'),('datagen_dict','n'),('modeldict','ykern_grid'),('modeldict','xkern_grid'),('datagen_dict','train_n'),('datagen_dict','evar'),('modeldict','hyper_param_form_dict'),('modeldict','regression_model')]
        for string_tup in string_list:
            new_dict_list.append({string_tup[0]:{string_tup[1]:adoubledict[string_tup[0]][string_tup[1]]}})#make the list match amodeldict, so optimization settings aren't changed
        #new_dict_list.append(amodeldict['xkern_grid'])
        #new_dict_list.append(amodeldict['hyper_param_form_dict'])
        #new_dict_list.append(amodeldict['regression_model'])
        
        simple_doubledict_list=deepcopy(saved_doubledict_list)#initialize these as copies that will be progressively simplified
        #simple_adoubledict=deepcopy(adoubledict)
        for new_dict in new_dict_list:
            #print(f'partial match trying {new_dict}')
            simple_doubledict_list=[self.do_dict_override(dict_i,new_dict) for dict_i in simple_doubledict_list]
            
            matchlist_idx=[adoubledict==dict_i for dict_i in simple_doubledict_list]
            matchlist=[dict_i for i,dict_i in enumerate(saved_optdict_list) if matchlist_idx[i]]
            if len(matchlist)>0:
                print(f'{len(matchlist)} partial matches found only after substituting {new_dict}')
                return matchlist
            
                  
    def do_dict_override(self,old_dict,new_dict,verbose=None,recursive=None):#key:values in old_dict replaced by any matching keys in new_dict, otherwise old_dict is left the same and returned.
        old_dict_copy=old_dict.copy()
        if verbose==None or verbose=='no':
            verbose=0
        if verbose=='yes':
            verbose=1
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
                old_dict_copy[key],vstring2=self.do_dict_override(old_dict_copy[key],val,recursive=1)
                vstring=vstring+vstring2
                #print('made it back from recursive call')
            else:
                try:
                    old_dict_copy[key]=val
                    if verbose==1:
                        print(f":val({new_dict[key]}) replaces val({old_dict_copy[key]})\n")
                        vstring=vstring+f":val({new_dict[key]}) or ({val}) replaces val({old_dict_copy[key]})\n"
                        
                except:
                    print(f'Warning: old_dict has keys:{[key for key,value in old_dict_copy.items()]} and new_dict has key:value::{key}:{new_dict[key]}')
        if verbose==1:
                print(f'vstring:{vstring} and done2')            
        if recursive==1:
            return old_dict_copy, vstring
        else: return old_dict_copy
    
    def build_hyper_param_start_values(self,modeldict):
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        Ndiff_start=modeldict['Ndiff_start']
        Ndiff_param_count=max_bw_Ndiff-(Ndiff_start-1)
        p=modeldict['param_count']
        assert not p==None, f"p is unexpectedly p:{p}"
        if modeldict['Ndiff_type']=='product':
                hyper_paramdict1={
                'Ndiff_exponent':.3*np.ones([Ndiff_param_count,]),
                'x_bandscale':1*np.ones([p,]),
                'outer_x_bw':np.array([2.7,]),
                'outer_y_bw':np.array([2.2,]),
                'Ndiff_depth_bw':.5*np.ones([Ndiff_param_count,]),
                'y_bandscale':1.5*np.ones([1,])
                    }

        if modeldict['Ndiff_type']=='recursive':
            hyper_paramdict1={
                'Ndiff_exponent':0*np.ones([Ndiff_param_count,]),
                'x_bandscale':1*np.ones([self.p,]),#
                'outer_x_bw':np.array([0.3,]),
                'outer_y_bw':np.array([0.3,]),
                'Ndiff_depth_bw':np.array([0.5]),
                'y_bandscale':0.2*np.ones([1,])
                }
        return hyper_paramdict1
            
        
    def build_dataset_dict(self,datagen_dict):
        data_dict={}
        param_count=datagen_dict['param_count']
        data_dict['p']=param_count
        seed=datagen_dict['seed']
        ftype=datagen_dict['ftype']
        evar=datagen_dict['evar']
        train_n=datagen_dict['train_n']
        n=datagen_dict['n']
        data_dict['train_n']=train_n
        
        self.dg_data=dg.datagen(data_shape=(n,param_count),seed=seed,ftype=ftype,evar=evar)
        data_dict['train_x']=self.dg_data.x[0:train_n,1:param_count+1]#drop constant from x and interaction/quadratic terms
        data_dict['train_y']=self.dg_data.y[0:train_n]
        
        val_n=n-train_n;assert not val_n<0,f'val_n expected non-neg, but val_n:{val_n}'
        data_dict['val_x']=self.dg_data.x[train_n:,1:param_count+1]#drop constant from x and 
        data_dict['val_y']=self.dg_data.y[train_n:]
        return data_dict
    
                         
    def build_optdict(self,opt_dict_override=None,param_count=None):
        if opt_dict_override==None:
            opt_dict_override={}
        max_bw_Ndiff=2
        try:train_n=self.train_n
        except:train_n=60
        Ndiff_start=1
        Ndiff_param_count=max_bw_Ndiff-(Ndiff_start-1)
        modeldict1={
            'Ndiff_type':'product',
            'param_count':param_count,
            'Ndiff_start':Ndiff_start,
            'max_bw_Ndiff':max_bw_Ndiff,
            'normalize_Ndiffwtsum':'own_n',
            'xkern_grid':'no',
            'ykern_grid':train_n+1,
            'outer_kern':'gaussian',
            'Ndiff_bw_kern':'rbfkern',
            'outer_x_bw_form':'one_for_all',
            'regression_model':'NW',
            'product_kern_norm':'self',
            'hyper_param_form_dict':{
                'Ndiff_exponent':'free',
                'x_bandscale':'non-neg',
                'Ndiff_depth_bw':'non-neg',
                'outer_x_bw':'non-neg',
                'outer_y_bw':'non-neg',
                'y_bandscale':'non-neg'
                }
            }
        #hyper_paramdict1=self.build_hyper_param_start_values(modeldict1)
        hyper_paramdict1={}
        
        #optimization settings for Nelder-Mead optimization algorithm
        optiondict_NM={
            'xatol':0.01,
            'fatol':0.1,
            'adaptive':True
            }
        optimizer_settings_dict1={
            'method':'Nelder-Mead',
            'options':optiondict_NM,
            'help_start':1,
            'partial_match':1
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
        #    print(traceback.format_exc())             
        #    print('------no start value overrides encountered------')
        #print(f'newoptimizedict1{newoptimizedict1}')
        return newoptimizedict1

        
        
class KernelCompare(KernelOptModelTools):
    def __init__(self,directory=None):
        if directory==None:
            self.kc_savedirectory=os.getcwd()
            merge_directory=self.kc_savedirectory
        else: 
            self.kc_savedirectory=directory
            merge_directory=".."
        os.chdir(self.kc_savedirectory)
        KernelOptModelTools.__init__(self,directory=self.kc_savedirectory)
        self.merge_and_condense_saved_models(merge_directory=merge_directory,save_directory=self.kc_savedirectory,condense=1,verbose=0)
                      #this should gather all directories from parent directory if directory is specified in object_init__()
                      #or if not, everything happens in the current working directory, which is good for testing without running
                      #through mycluster.
        
    def prep_model_list(self, optdict_variation_list=None,datagen_variation_list=None):
        param_count=2
        datagen_dict={'train_n':60,'n':200, 'param_count':param_count,'seed':1, 'ftype':'linear', 'evar':1}
        if datagen_variation_list==None:
            datagen_variation_list=[{}]#will default to parameters in datagen_dict below
        assert type(datagen_variation_list)==list,f'datagen_variation_list type:{type(datagen_variation_list)} but expected a list'
        assert type(datagen_variation_list[0])==tuple,f'first item of datagen_variation_list type:{type(datagen_variation_list[0])} but expected a tuple'
                        
        assert type(optdict_variation_list)==list,f'optdict_variation_list type:{type(optdict_variation_list)} but expected a list'
        assert type(optdict_variation_list[0])==tuple,f'first item of optdict_variation_list type:{type(optdict_variation_list[0])} but expected a tuple'
                         
        #initial_opt_dict=self.build_optdict(param_count=datagen_dict['param_count'])
        #if optdict_variation_list==None:
        #    optdict_list=[initial_opt_dict]
        
        model_run_dict_list=[]
        datagen_dict_list=self.build_dict_variations(datagen_dict,datagen_variation_list)
        print(f'len(datagen_dict_list):{len(datagen_dict_list)}')
        for alt_datagen_dict in datagen_dict_list:
            initial_opt_dict=self.build_optdict(param_count=alt_datagen_dict['param_count'])
            print('here1')
            optdict_list=self.build_dict_variations(initial_opt_dict,optdict_variation_list)    
            for optdict_i in optdict_list:
                optmodel_run_dict={'optimizedict':optdict_i,'datagen_dict':alt_datagen_dict}    
                model_run_dict_list.append(optmodel_run_dict)
                #print('model_run_dict_list:',model_run_dict_list)
        return model_run_dict_list
    
                         
    def run_model_as_node(self,optimizedict,datagen_dict,force_start_params=None):
        self.do_monte_opt(optimizedict,datagen_dict,force_start_params=force_start_params)
        return
        
                         
    def build_dict_variations(self,initial_dict,variation_list):
        dict_combo_list=[]
        for i,tup_i in enumerate(variation_list):
            sub_list=[not_tup_i for j,not_tup_i in enumerate(variation_list) if not j==i]
            for k,val in enumerate(tup_i[1]):
                override_dict_ik=self.build_override_dict_from_str(tup_i[0],val)
                dict_ik=self.do_dict_override(initial_dict,override_dict_ik)
                #print('dict_combo_list',dict_combo_list)
                dict_combo_list.append(dict_ik)
                if len(sub_list)>0:
                    new_items=self.build_dict_variations(dict_ik,sub_list)
                    dict_combo_list=dict_combo_list+new_items
                else: 
                    return dict_combo_list
        return dict_combo_list

                         
    def build_override_dict_from_str(self,string_address,val):
        colon_loc=[i for i,char in enumerate(string_address) if char==':']
        return self.recursive_string_dict_helper(string_address,colon_loc,val)
          
                         
    def recursive_string_dict_helper(self,dict_string,colon_loc,val):
        if len(colon_loc)==0:
            return {dict_string:val}
        if len(colon_loc)>0:
            return {dict_string[0:colon_loc[0]]:self.recursive_string_dict_helper(dict_string[colon_loc[0]+1:],colon_loc[1:],val)}
        
                         
    def build_quadratic_datagen_dict_override(self):
        datagen_dict_override={}
        datagen_dict_override['ftype']='quadratic'
        return datagen_dict_override
        
                         
    def test_build_opt_dict_override(self):
        trainsize=self.train_n
        opt_dict_override={}
        modeldict={}
        hyper_param_form_dict={}
        hyper_param_dict={}
        opt_settings_dict={}
        options={}

        modeldict['Ndiff_type']='recursive'
        modeldict['max_bw_Ndiff']=3
        modeldict['Ndiff_start']=1
        modeldict['ykern_grid']=trainsize+1
        #modeldict['hyper_param_form_dict']={'y_bandscale':'fixed'}
        #hyper_param_dict['y_bandscale']=np.array([1])
        #opt_dict_override['hyper_param_dict']=hyper_param_dict
        opt_dict_override['modeldict']=modeldict

        options['fatol']=0.5
        options['xatol']=.05
        opt_settings_dict['options']=options
        #opt_settings_dict['help_start']='no'
        #opt_settings_dict['partial_match']='no'
        opt_dict_override['opt_settings_dict']=opt_settings_dict
        return opt_dict_override
