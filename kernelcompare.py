from time import strftime
import numpy as np
import pickle
import os
import data_gen as dg
import mykern as mk
import re

#import datetime


        

class KernelOptModelTool:
    def __init__(self,datagen_dict_override=None):
        try: 
            self.dg_data
        except:
            default_datagen_dict={'train_n':40,'n':200, 'param_count':2,'seed':1, 'ftype':'linear', 'evar':1}
            self.datagen_dict=self.do_dict_override(default_datagen_dict,datagen_dict_override)
            self.build_dataset()#create x,y 
        
    def do_monte_opt(self,opt_dict_override=None):
        
       
        self.build_dict(opt_dict_override)#
        #self.data_and_modeldict={'data':self.dg_data,'model':self.optimizedict}
        
        y=self.train_y
        x=self.train_x
        self.open_condense_resave('model_save')
        self.open_condense_resave('final_model_save')
        optimizedict=self.run_opt_complete_check(y,x,self.optimizedict,replace=1)
        self.optimizedict=optimizedict.copy()
        self.minimize_obj=self.run_optimization(self.train_y,self.train_x,self.optimizedict)
        
        
    def run_opt_complete_check(self,y,x,optimizedict_orig,replace=None):
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
        
        same_modelxy_dict_list=self.open_and_compare_optdict('condensed_model_save',optimizedict,y,x,help_start=help_start,partial_match=partial_match)
        if len(same_modelxy_dict_list)>0:
            #print(f"from model_save, This dictionary, x,y combo has finished optimization before:{len(same_modelxy_dict_list)} times")
            #print(f'first item in modelxy_dict_list:{same_modelxy_dict_list[0]}'')
            mse_list=[dict_i['mse'] for dict_i in same_modelxy_dict_list]
            train_n=[dict_i['ydata'].shape[0] for dict_i in same_modelxy_dict_list]
            n_wt_mse_list=[self.do_nwt_mse(mse_list[i],train_n[i]) for i in range(len(mse_list))]
            lowest_n_wt_mse=min(n_wt_mse_list)
            
            best_dict_list.append(same_modelxy_dict_list[n_wt_mse_list.index(lowest_n_wt_mse)])
        '''  
        same_modelxy_dict_list=self.open_and_compare_optdict('final_model_save',optimizedict,y,x,help_start=help_start,partial_match=partial_match)
        if len(same_modelxy_dict_list)>0:
            #print(f"from final_model_save, This dictionary, x,y combo has finished optimization before:{len(same_modelxy_dict_list)} times")
            mse_list=[dict_i['mse'] for dict_i in same_modelxy_dict_list]
            train_n=[dict_i['ydata'].shape[0] for dict_i in same_modelxy_dict_list]
            n_wt_mse_list=[self.do_nwt_mse(mse_list[i],train_n[i]) for i in range(len(mse_list))]
            lowest_n_wt_mse=min(n_wt_mse_list)
            best_dict_list.append(same_modelxy_dict_list[n_wt_mse_list.index(lowest_n_wt_mse)])
        '''
        
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
                #print("overriding start parameters with saved parameters")
                self.rebuild_hyper_param_dict(optimizedict,best_dict['params'],verbose=0)
            else:
                print('continuing without replacing parameters with their saved values')
        
        
        #if len(mse_list)==0 and 
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
                  
    def open_condense_resave(self,filename1,verbose=None):
        if verbose==None or verbose=='no':
            verbose=0
        if verbose=='yes':
            verbose=1
        try:
            with open(filename1,'rb') as savedfile:
                saved_model_list1=pickle.load(savedfile)
            condensed_list=self.condense_saved_model_list(saved_model_list1, help_start=0, strict=1,verbose=verbose)
        except:
            print(f'filename:{filename1} not found')
            return
        try:
            with open(filename1,'wb') as writefile:
                pickle.dump(condensed_list,writefile)
        except:
                print(f'filewrite for filename:{filename1} failed')
                

        
            
    def merge_and_condense_saved_models(self,merge_directory=None,save_directory=None,condense=None,verbose=None):
        if not merge_directory==None:
            if not os.path.exists(merge_directory):
                os.makedirs(merge_directory)
        if not save_directory==None:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
        if condense==None or condense=='no':
            condense=0
        if condense=='yes':
            condense=1
                
        model_save_filelist=[name_i for name_i in os.listdir(merge_directory) if re.search('model_save',name_i)]
        print('here',model_save_filelist)
        
        try:
            with open('condensed_model_save','rb') as savedfile:
                saved_model_list1=pickle.load(savedfile)
            condensed_list1=self.condense_saved_model_list(saved_model_list1, help_start=0, strict=1,verbose=0)
            modeldict_list1=[dict_i['modeldict'] for dict_i in condensed_list1]
        except:
            condensed_list1=[]
            modeldict_list1=[]
                    
        
        if len(model_save_filelist)==0:
            print('no models found when merging')
            return
        if len(model_save_filelist)==1:
            print('only 1 model_save file found, no merge completed')
            return
        if len(model_save_filelist)>1:
            for file_i in model_save_filelist:
                with open(file_i,'rb') as savedfile2:
                    try: saved_model_list2=pickle.load(savedfile2)
                    except:saved_model_list2=[]
                print(f'file_i:{file_i} has {len(file_i)} saved model(s)')
                condensed_list2=self.condense_saved_model_list(saved_model_list2, help_start=0, strict=1,verbose=0)
                new_model_list=[]
                modeldict_list2=[dict_i['modeldict'] for dict_i in condensed_list2]
        #matching_modeldict_list[dict_1==dict_2 for dict_1 in modeldict_list1 for dict_2 in modeldict_list2]
        
                jbest=[1]*len(condensed_list2)
                
                for i,dict_1 in enumerate(modeldict_list1):
                    ibest=1#start optimistic
                    for j,dict_2 in enumerate(modeldict_list2):
                        if dict_1==dict_2:
                            if condensed_list1[i]['mse']>condensed_list2[j]['mse']:
                                ibest=0
                            else:
                                jbest[j]=0
                    if ibest==1:#only true if dict_1[i] never lost, because it won or was unique
                        new_model_list.append(condensed_list1[i])
                for i,dict_2 in enumerate(condensed_list2):
                    if jbest[i]==1:
                        new_model_list.append(condensed_list2[i])
                condensed_list1=new_model_list
        with open('condensed_model_save','wb') as newfile:
            print(f'condensed_list1:{condensed_list1}')
            pickle.dump(condensed_list1,newfile)
                                      
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
        thismodeldict=optimizedict['modeldict']
        #print(saved_filename)
        #print(f'saved_dict_list has first item of:{type(saved_dict_list[0])}')
        modeldict_list=[dict_i['modeldict'] for dict_i in saved_dict_list]
        optdict_match_list_select=[dict_i==thismodeldict for dict_i in modeldict_list]#list of boolean
        optdict_match_list=[saved_dict_list[i] for i,ismatch in enumerate(optdict_match_list_select) if ismatch]
        #print(f'optdict_match_list1:{optdict_match_list}')
        testlist1=[dict_i['modeldict'] for dict_i in saved_dict_list]
        testlist2=[thismodeldict==dict_i for dict_i in testlist1]
        finallist=[dict_i for i,dict_i in enumerate(saved_dict_list) if testlist2[i]]
        #print(f'do lists match? {[optdict_match_list[i]==finallist[i] for i in range(len(optdict_match_list))]}')
        #print(f'finallist length:{len(finallist)},optdict_match_list length:{len(optdict_match_list)}')
        
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
            #same_modeldict_list=[saved_dict_list[i] for i,is_same in enumerate(modeldict_compare_list) if is_same]
            xcompare_list=[np.all(dict_i['xdata']==x) for dict_i in optdict_match_list]
            same_model_and_x_dict_list=[optdict_match_list[i] for i,is_same in enumerate(xcompare_list) if is_same]
            ycompare_list=[np.all(dict_i['ydata']==y) for dict_i in same_model_and_x_dict_list]
            same_modelxy_dict_list=[same_model_and_x_dict_list[i] for i,is_same in enumerate(ycompare_list) if is_same]
            if len(same_modelxy_dict_list)==0 and partial_match==1:
                if len(same_model_and_x_dict_list)>0: 
                    print('found matching models and matching x but no matching y too')
                    return same_model_and_x_dict_list
                else: 
                    print('found matching models but not matching x or matching y')
                    return modeldict_compare_list
            print('')
            return same_modelxy_dict_list
    
    def do_partial_match(self,saved_optdict_list,afullmodel,help_start,strict=None):
        if strict==None or strict=='no':
            strict=0
        if strict=='yes': strict=1
        amodeldict=afullmodel['modeldict']
        saved_modeldict_list=[dict_i['modeldict'] for dict_i in saved_optdict_list]
        same_modeldict_compare=[amodeldict==dict_i for dict_i in saved_modeldict_list]
        
        matches=[item for i,item in enumerate(saved_optdict_list) if same_modeldict_compare[i]]
        matchcount=len(matches)
        if strict==1:
            #keys=[key for key,val in afullmodel.items()]
            #print(f'keys:{keys}')
            try:n,k=afullmodel['xdata'].shape
            except:n,k=self.train_x.shape
            same_modeldict_compare=[dict_i for dict_i in matches if dict_i['xdata'].shape==(n,k)]
            return same_modeldict_compare
        if help_start==0:
            return matches
        if matchcount>help_start*1:
            print(f'partial match found a full match.....matchcount:{matchcount}')
            return matches
        #print(simplifying1)
        new_dict_list=[]
        string_list=['ykern_grid','xkern_grid','hyper_param_form_dict','regression_model']
        for string in string_list:
            new_dict_list.append({string:amodeldict[string]})#make the list match amodeldict, so optimization settings aren't changed
        #new_dict_list.append(amodeldict['xkern_grid'])
        #new_dict_list.append(amodeldict['hyper_param_form_dict'])
        #new_dict_list.append(amodeldict['regression_model'])
        
        simple_modeldict_list=saved_modeldict_list.copy()#initialize these as copies that will be progressively simplified
        simple_amodeldict=amodeldict.copy()
        for new_dict in new_dict_list:
            print(f'partial match trying {new_dict}')
            simple_modeldict_list=[self.do_dict_override(dict_i,new_dict) for dict_i in simple_modeldict_list]
            simple_amodeldict=self.do_dict_override(simple_amodeldict,new_dict)
            matchlist_idx=[simple_amodeldict==dict_i for dict_i in simple_modeldict_list]
            matchlist=[dict_i for i,dict_i in enumerate(saved_optdict_list) if matchlist_idx[i]]
            if len(matchlist)>0:
                print(f'{len(matchlist)} partial matches found after substituting {new_dict}')
                return matchlist
            
        
        
            
            
    
    def run_optimization(self,y,x,optimizedict):
        start_msg=f'starting at {strftime("%Y%m%d-%H%M%S")}'
        print(start_msg)
        self.optimize_obj=mk.optimize_free_params(y,x,optimizedict)
                  
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
                print(f'val is dict in {key}, recursive call')
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
    
    def build_start_values(self,modeldict):
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        Ndiff_start=modeldict['Ndiff_start']
        Ndiff_param_count=max_bw_Ndiff-(Ndiff_start-1)
        
        if modeldict['Ndiff_type']=='product':
                hyper_paramdict1={
                'Ndiff_exponent':.3*np.ones([Ndiff_param_count,]),
                'x_bandscale':1*np.ones([self.p,]),
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
            
        
        
    def build_dataset(self):
        param_count=self.datagen_dict['param_count']
        seed=self.datagen_dict['seed']
        ftype=self.datagen_dict['ftype']
        evar=self.datagen_dict['evar']
        train_n=self.datagen_dict['train_n']
        n=self.datagen_dict['n']
        
        self.dg_data=dg.data_gen(data_shape=(n,param_count),seed=seed,ftype=ftype,evar=evar)
        self.train_x=self.dg_data.x[0:train_n,1:param_count+1]#drop constant from x and interaction/quadratic terms
        self.train_y=self.dg_data.y[0:train_n]
        #print(self.train_y)
        val_n=n-train_n;assert not val_n<0,f'val_n expected non-neg, but val_n:{val_n}'
        self.val_x=self.dg_data.x[train_n:,1:param_count+1]#drop constant from x and 
        self.val_y=self.dg_data.y[train_n:]
    
    def build_dict(self,opt_dict_override):
        
        max_bw_Ndiff=2
        self.p=self.datagen_dict['param_count']
        Ndiff_start=1
        Ndiff_param_count=max_bw_Ndiff-(Ndiff_start-1)
        modeldict1={
            'Ndiff_type':'product',
            'Ndiff_start':Ndiff_start,
            'max_bw_Ndiff':max_bw_Ndiff,
            'normalize_Ndiffwtsum':'own_n',
            'xkern_grid':'no',
            'ykern_grid':50,
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
        #hyper_paramdict1=self.build_start_values(modeldict1)
        hyper_paramdict1={}
        
        #optimization settings for Nelder-Mead optimization algorithm
        optiondict_NM={
            'xatol':0.001,
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
        
        newhyper_paramdict1=self.build_start_values(newoptimizedict1['modeldict'])
        newoptimizedict1['hyper_param_dict']=newhyper_paramdict1
        try: 
            start_val_override_dict=opt_dict_override['hyper_param_dict']
            print(f'start_val_override_dict:{start_val_override_dict}')
            start_override_opt_dict={'hyper_param_dict':start_val_override_dict}
            newoptimizedict1=self.do_dict_override(newoptimizedict1,start_override_opt_dict,verbose=0)
        except:
            print('------no start value overrides encountered------')
        print(f'newoptimizedict1{newoptimizedict1}')
        self.optimizedict=newoptimizedict1

        
        
class Kernelcompare(KernelOptModelTool):
    def __init__(self):
        datagen_dict_override=self.build_datagen_dict_override()
        KernelOptModelTool.__init__(self,datagen_dict_override=datagen_dict_override)
        opt_dict_override=self.build_opt_dict_override()
        self.merge_and_condense_saved_models(merge_directory=None,save_directory=None,condense=1,verbose=1)
        self.do_monte_opt(opt_dict_override=opt_dict_override)
        
        
    def build_datagen_dict_override(self):
        self.trainsize=45
        datagen_dict_override={}
        datagen_dict_override['train_n']=self.trainsize
        datagen_dict_override['ftype']='quadratic'
        return datagen_dict_override
    
    
    def build_opt_dict_override(self):
        opt_dict_override={}
        modeldict={}
        hyper_param_form_dict={}
        hyper_param_dict={}
        opt_settings_dict={}
        options={}

        modeldict['Ndiff_type']='recursive'
        modeldict['max_bw_Ndiff']=3
        modeldict['Ndiff_start']=2
        modeldict['ykern_grid']=self.trainsize+1
        #modeldict['hyper_param_form_dict']={'y_bandscale':'fixed'}
        #hyper_param_dict['y_bandscale']=np.array([1])
        #opt_dict_override['hyper_param_dict']=hyper_param_dict
        opt_dict_override['modeldict']=modeldict

        options['fatol']=0.05
        options['xatol']=.005
        opt_settings_dict['options']=options
        #opt_settings_dict['help_start']='no'
        #opt_settings_dict['partial_match']='no'
        opt_dict_override['opt_settings_dict']=opt_settings_dict
        return opt_dict_override
