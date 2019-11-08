from time import strftime
import numpy as np
import pickle
import data_gen as dg
import mykern as mk
#import datetime

class DoKernelOpt(object):
    def __init__(self,datagen_dict_override=None,opt_dict_override=None):
        default_datagen_dict={'train_n':40,'n':200, 'param_count':2,'seed':1, 'ftype':'linear', 'evar':1}
        self.datagen_dict=self.do_dict_override(default_datagen_dict,datagen_dict_override)
        self.build_dataset()#create x,y
        self.build_dict(opt_dict_override)#
        #self.data_and_modeldict={'data':self.dg_data,'model':self.optimizedict}
        
        y=self.train_y
        x=self.train_x
        
        self.optimizedict=self.run_opt_complete_check(y,x,self.optimizedict,replace=1)

        self.run_optimization(self.train_y,self.train_x,self.optimizedict)
        
        
    def run_opt_complete_check(self,y,x,optimizedict,replace=None):
        '''
        checks model_save and then final_model_save to see if the same modeldict has been run before (e.g.,
        same model featuers, same starting parameters, same data).
        -by default or if replace set to 1 or 'yes', then the start parameters are replaced by the matching model from model_save 
            and final_model_save with the lowest mse
        -if replace set to 0 or 'no', then the best matching model is still announced, but replacement of start parameters doens't happen
        '''
        if replace==None or replace=='yes':
            replace=1
        if replace=='no':
            replace=0
        best_dict_list=[]
        help_start=optimizedict['opt_settings_dict']['help_start']
        partial_match=partial_match=optimizedict['opt_settings_dict']['partial_match']
        same_modelxy_dict_list=self.open_and_compare_optdict('model_save',optimizedict,y,x,help_start=help_start,partial_match=partial_match)
        if len(same_modelxy_dict_list)>0:
            print(f"from model_save, This dictionary, x,y combo has finished optimization before:{len(same_modelxy_dict_list)} times")
            mse_list=[dict_i['mse'] for dict_i in same_modelxy_dict_list]
            train_n=[dict_i['ydata'].shape[0] for dict_i in same_modelxy_dict_list]
            n_wt_mse_list=[mse_list[i]*train_n[i]**-2 for i in range(len(mse_list))]
            lowest_n_wt_mse=min(n_wt_mse_list)
            print('boom')
            best_dict_list.append(same_modelxy_dict_list[n_wt_mse_list.index(lowest_n_wt_mse)])
            
        same_modelxy_dict_list=self.open_and_compare_optdict('final_model_save',optimizedict,y,x,help_start=help_start,partial_match=0)
        if len(same_modelxy_dict_list)>0:
            print(f"from final_model_save, This dictionary, x,y combo has finished optimization before:{len(same_modelxy_dict_list)} times")
            mse_list=[dict_i['mse'] for dict_i in same_modelxy_dict_list]
            train_n=[dict_i['ydata'].shape[0] for dict_i in same_modelxy_dict_list]
            n_wt_mse_list=[mse_list[i]*train_n[i]**-2 for i in range(len(mse_list))]
            lowest_n_wt_mse=min(n_wt_mse_list)
            best_dict_list.append(same_modelxy_dict_list[n_wt_mse_list.index(lowest_n_wt_mse)])
        
        
        mse_list=[dict_i['mse'] for dict_i in best_dict_list]
        if len(mse_list)>0:
            n_wt_mse_list=[mse_list[i]*train_n[i]**-2 for i in range(len(mse_list))]
            train_n=[dict_i['ydata'].shape[0] for dict_i in same_modelxy_dict_list]
            lowest_n_wt_mse=min(n_wt_mse_list)
            best_dict=best_dict_list[n_wt_mse_list.index(lowest_n_wt_mse)]
            print(f'optimiziation dict with lowest mse:{best_dict["mse"]}was last saved{best_dict["when_saved"]}')
            if replace==1:
                print("overriding start parameters with saved parameters")
                self.rebuild_hyper_param_dict(optimizedict,best_dict['params'],verbose=1)
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
        print(f'rebuild hyper param dict vstring:{vstring}')
        return new_opt_dict
    
    def merge_and_condense_saved_models(self,filename1,filename2,condense=None,verbose=None):
        if condense==None or condense=='no':
            condense=0
        if condense=='yes':
            condense=1
        
        with open(filename1,'rb') as savedfile:
            saved_model_list1=pickle.load(savedfile)
        with open(filename2,'rb') as savedfile:
            saved_model_list2=pickle.load(savedfile)
        condensed_list1=self.condense_saved_model_list(saved_model_list1)
        condensed_list2=self.condense_saved_model_list(saved_model_list2)
        new_model_list=[]
        modeldict_list1=[dict_i['modeldict'] for dict_i in condensed_list1]
        modeldict_list2=[dict_i['modeldict'] for dict_i in condensed_list2]
        #matching_modeldict_list[dict_1==dict_2 for dict_1 in modeldict_list1 for dict_2 in modeldict_list2]
        
        jbest=[1]*len(condensed_list2)
        for dict_1 in modeldict_list1:
            ibest=1#start optimistic
            for j,dict_2 in enumerate(modeldict_list2):
                if dict_1==dict_2:
                    if condensed_list1[i]>condensed_list2[j]:
                        ibest=0
                    else:
                        jbest[j]=0
            if ibest==1:#only true if dict_1[i] never lost, because it won or was unique
                new_model_list.append(condensed_list1[i])
        for i,dict_2 in enumerate(condensed_list2):
            if jbest[i]==1:
                new_model_list.append(condensed_list2[i])
        with open(condensed+filename1[:-1]),'wb' as newfile:
            pickle.dump(new_model_list,newfile)
                                      
    def condense_saved_model_list(self,saved_model_list):
        keep_model=[1]*len(saved_model_list)
        for i,modeli in enumerate(saved_model_list):
            for j,modelj in enumerate(saved_model_list[i+1:]):
                if modeli['modeldict']==modelj['modeldict']:
                    if modeli['mse']<modelj['mse']:
                        keep_model[j]=0
                    else:
                        keep_model[i]=0
        return [model for i,model in enumerate(saved_model_list) if keep_model[i]==1]
    
    def open_and_compare_optdict(self,saved_filename,optimizedict,y,x,help_start=None,partial_match=None):
        if help_start==None: help_start=0
        if help_start=='yes': help_start=1
        if partial_match==None: partial_match=0
        if partial_match=='yes': partial_match=1
         
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
        optdict_match_list=[dict_i for dict_i in saved_dict_list if dict_i['modeldict']==thismodeldict]#list of boolean
        testlist1=[dict_i['modeldict'] for dict_i in saved_dict_list]
        testlist2=[thismodeldict==dict_i for dict_i in testlist1]
        finallist=[dict_i for dict_i in thismodeldict if testlist2]
        print(f'finallist length:{len(finallist)},optdict_match_list length:{len(optdict_match_list)}')
        
        if help_start==1 and len(optdict_match_list)>0:
            print('------help_start is triggered------')
            return self.condense_saved_model_list(optdict_match_list)
        elif len(optdict_match_list)==0 and partial_match==1:
            return self.do_partial_match(saved_dict_list,thismodeldict)
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
            return same_modelxy_dict_list
    
    def do_partial_match(self,saved_optdict_list,thismodeldict):
        saved_modeldict_list=[dict_i['modeldict'] for dict_i in saved_optdict_list]
        same_modeldict_compare=[thismodeldict==dict_i for dict_i in saved_modeldict_list]
        matchcount=len([i for i in saved_optdict_list if same_modeldict_compare])
        if matchcount>0:
            print(f'partial match found a full match.....matchcount:{matchcount}')
            return [i for i in saved_optdict_list if same_modeldict_compare]
        print(simplifying1)
        new_dict_list=[]
        new_dict_list.append(this_modeldict['ykern_grid'])#make the list match this_modeldict, so optimization settings aren't changed
        new_dict_list.append(this_modeldict['xkern_grid'])
        new_dict_list.append(this_modeldict['hyper_param_form_dict'])
        new_dict_list.append(this_modeldict['regression_model'])
        
        simple_modeldict_list=saved_modeldict_list['modeldict'].copy()#initialize these as copies that will be progressively simplified
        simple_thismodeldict=thismodeldict.copy()
        for new_dict in new_dict_list:
            print(f'partial match trying {new_dict}')
            simple_modeldict_list=[self.do_dict_override(dict_i,new_dict) for dict_i in simple_modeldict_list]
            simple_thismodeldict=self.do_dict_override(simple_thismodeldict,new_dict)
            matchlist_idx=[simple_thismodeldict==dict_i for dict_i in simple_modeldict_list]
            matchlist=[dict_i for i,dict_i in enumerate(saved_optdict_list) if matchlist_idx[i]]
            if len(matchlist)>0:
                print(f'{len(matchlist)} partial matches found after substituting {new_dict}')
                return matchlist
            
        
        
            
            
    
    def run_optimization(self,y,x,optimizedict):
        start_msg=f'starting at {strftime("%Y%m%d-%H%M%S")}'
        print(start_msg)
        mk.optimize_free_params(y,x,optimizedict)
                  
    def do_dict_override(self,old_dict,new_dict,verbose=None):#key:values in old_dict replaced by any matching keys in new_dict, otherwise old_dict is left the same and returned.
        if verbose==None or verbose=='no':
            verbose=0
        if verbose=='yes':
            verbose=1
        vstring=''
        if new_dict==None or new_dict=={}:
            if verbose==1:
                print(f'vstring:{vstring}')
            return old_dict
        for key,val in new_dict.items():
            if verbose==1:
                vstring=vstring+f":key({key})"
            if type(val) is dict:
                old_dict[key]=self.do_dict_override(old_dict[key],new_dict[key])
            else:
                try:
                    old_dict[key]=new_dict[key]
                    if verbose==1:
                        vstring=vstring+f":val({new_dict[key]}) replaces val({old_dict[key]})\n"
                        
                except:
                    print(f'old_dict has keys:{[key for key,value in old_dict.items()]} and new_dict has key:value::{key}:{new_dict[key]}')
        if verbose==1:
                print(f'vstring:{vstring}')            
        return old_dict
    
    def build_start_values(self,modeldict):
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        Ndiff_start=modeldict['Ndiff_start']
        Ndiff_param_count=max_bw_Ndiff-(Ndiff_start-1)
        
        if modeldict['Ndiff_type']=='product':
                hyper_paramdict1={
                'Ndiff_exponent':.3*np.ones([Ndiff_param_count,]),
                'x_bandscale':-0.3*np.ones([self.p,]),
                'outer_x_bw':np.array([2.7,]),
                'outer_y_bw':np.array([2.2,]),
                'Ndiff_depth_bw':.5*np.ones([Ndiff_param_count,]),
                'y_bandscale':1.5*np.ones([1,])
                    }

        if modeldict['Ndiff_type']=='recursive':
            hyper_paramdict1={
                'Ndiff_exponent':.3*np.ones([Ndiff_param_count,]),
                'x_bandscale':-0.03*np.ones([self.p,]),
                'outer_x_bw':np.array([1,]),
                'outer_y_bw':np.array([3,]),
                'Ndiff_depth_bw':np.array([0.5]),
                'y_bandscale':2*np.ones([1,])
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
            'help_start':'yes',
            'partial_match':'yes'
            }
        
        optimizedict1={
            'opt_settings_dict':optimizer_settings_dict1,
            'hyper_param_dict':hyper_paramdict1,
            'modeldict':modeldict1
            } 
        
        optimizedict1=self.do_dict_override(optimizedict1,opt_dict_override)
        hyper_paramdict1=self.build_start_values(optimizedict1['modeldict'])
        optimizedict1['hyper_param_dict']=hyper_paramdict1
        
        self.optimizedict=optimizedict1
   
