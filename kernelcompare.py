from kc_helpers import KCHelper
from kc_pisces import KCPisces
from helpers import Helper
from pipe import PipeLine
from kernelparams import KernelParams
import datagen as dg
import mykern_nomask as mk


import traceback
from copy import deepcopy
from time import strftime
import numpy as np
import pickle
import os
import re
import logging,traceback
import yaml


class KernelOptModelTools(mk.optimize_free_params,KCHelper,KCPisces,PipeLine):
    def __init__(self,directory=None,myname=None):
        self.max_maxbatchbatchcount=None
        if directory==None:
            self.kc_savedirectory=os.getcwd
        else:
            self.kc_savedirectory=directory
        self.species_model_save_path_dict=os.path.join(self.kc_savedirectory,'species_model_save_path_dict.pickle')
        
        #mk.kNdtool.__init__(self,savedir=self.kc_savedirectory,myname=myname)
        self.Ndiff_list_of_masks_x=None
        self.Ndiff_list_of_masks_y=None
        
        try:
            self.logger=logging.getLogger(__name__)
            self.logger.info('starting new KernelOptModelTools object')
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
            self.logger.info('starting new KernelOptModelTools log')
            print(traceback.format_exc())
        self.helper=Helper()
        KCHelper.__init__(self)
        KCPisces.__init__(self)
        
        
        mk.optimize_free_params.__init__(self,kcsavedir=self.kc_savedirectory,myname=myname)
        self.pname=myname
       
                
    def do_monte_opt(self,optimizedict,datagen_dict,force_start_params=None):
        '''if self.max_maxbatchbatchcount:
            if not max_maxbatchbatchcount in datagen_dict:
                datagen_dict['max_maxbatchbatchcount']=self.max_maxbatchbatchcount'''
        optimizedict['datagen_dict']=datagen_dict
        
        if force_start_params==None or force_start_params=='no':
            force_start_params=0
        if force_start_params=='yes':
            force_start_params=1

        
        maxbatchbatchcount=optimizedict['modeldict']['maxbatchbatchcount']
        
        datagen_obj=dg.datagen(datagen_dict)
        
        if datagen_obj.species_n<datagen_obj.batch_n*datagen_obj.batchcount:
            print(f'skipping {datagen_obj.species} b/c species_n:{datagen_obj.species_n} < datagen_obj.batch_n*datagen_obj.batchcount:{datagen_obj.batch_n*datagen_obj.batchcount}')
            self.logger.info(f'skipping {datagen_obj.species} b/c species_n:{datagen_obj.species_n} < fullbatchbatch_n:{datagen_obj.fullbatchbatch_n}')
            return
            
        # naiveloss=self.do_naiveloss(datagen_obj)
        # print('naiveloss:',naiveloss)
        
        datagen_dict_expanded=datagen_obj.datagen_dict_expanded
        optimizedict['datagen_dict']=datagen_dict_expanded
        #print(f'datagen_dict_expanded:{datagen_dict_expanded} for directory,{self.kc_savedirectory}')
        
        if not force_start_params:
            optimizedict=self.run_opt_complete_check(optimizedict,replace=1)
            
            
        start_msg=f'starting at {strftime("%Y%m%d-%H%M%S")}'
        try: nodedir=self.nodedirectory
        except: nodedir=self.kc_savedirectory
        self.run_opt(datagen_obj,optimizedict,nodedir)
        
        #(datagen_obj,optimizedict,self.nodedirectory,myname=self.pname,xmask=self.Ndiff_list_of_masks_x,ymask=self.Ndiff_list_of_masks_y)
        return
        
                               
    
                               
                               
    def run_opt_complete_check(self,optimizedict_orig,replace=None):
        '''
        checks model_save and then final_model_save to see if the same modeldict has been run before (e.g.,
        same model featuers, same starting parameters, same datagen_dict).
        -by default or if replace set to 1 or 'yes', then the start parameters are replaced by the matching model from model_save 
            and final_model_save with the lowest 'nwt loss "
        -if replace set to 0 or 'no', then the best matching model is still announced, but replacement of start parameters doens't happen
            Only loss and parameters pulled by self.pull2dicts are compared/replaced. modeldict and datagen_dict
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
        merged_path=os.path.join(self.kc_savedirectory,'..','condensed_model_save')
        same_modelxy_dict_list1=self.open_and_compare_optdict(
            merged_path,optimizedict,help_start=help_start,partial_match=partial_match)
        if len(same_modelxy_dict_list1)==0:
            merged_path=os.path.join(self.kc_savedirectory,'condensed_model_save')
            same_modelxy_dict_list2=self.open_and_compare_optdict(
                merged_path,optimizedict,help_start=help_start,partial_match=partial_match)
            if len(same_modelxy_dict_list2)==0:
                same_modelxy_dict_list3=self.open_and_compare_optdict(
                    'model_save',optimizedict,help_start=help_start,partial_match=partial_match)
                same_modelxy_dict_list=same_modelxy_dict_list3#even if it is empty
            else: same_modelxy_dict_list=same_modelxy_dict_list2
        else: same_modelxy_dict_list=same_modelxy_dict_list1
            
        
        
        if len(same_modelxy_dict_list)>0:
            #p#rint(f"from model_save, This dictionary, x,y combo has finished optimization before:{len(same_modelxy_dict_list)} times")
            #p#rint(f'first item in modelxy_dict_list:{same_modelxy_dict_list[0]}'')
            loss_list=[dict_i['loss'] for dict_i in same_modelxy_dict_list]
            try:
                n_list=[dict_i['datagen_dict']['train_n'] for dict_i in same_modelxy_dict_list]
                batchcount_list=[1 for _ in range(len(n_list))]
            except:
                n_list=[dict_i['datagen_dict']['batch_n'] for dict_i in same_modelxy_dict_list]
                batchcount_list=[dict_i['datagen_dict']['batchcount'] for dict_i in same_modelxy_dict_list]

            batchbatchcountlist=[];naivelosslist=[]
            for dict_i in same_modelxy_dict_list:
                try:
                    i_batchbatch1=dict_i['datagen_dict']['batchbatchcount']
                    i_batchbatch2=dict_i['modeldict']['maxbatchbatchcount']
                    if i_batchbatch1<i_batchbatch2:
                        batchbatchcountlist.append(i_batchbatch1)
                    else: batchbatchcountlist.append(i_batchbatch2)
                except: batchbatchcountlist.append(1)
                try:
                    naivelosslist.append(dict_i['naiveloss'])
                except: naivelosslist.append(1)
                           
            n_wt_loss_list=[self.do_nwt_loss(loss_list[i],n_list[i],batchcount_list[i],batchbatchcount=batchbatchcountlist[i],naiveloss=naivelosslist[i]) for i in range(len(loss_list))]
            lowest_n_wt_loss=min(n_wt_loss_list)
            
            best_dict_list.append(same_modelxy_dict_list[n_wt_loss_list.index(lowest_n_wt_loss)])
        #if len(same_modelxy_dict_list)==0:
            #print('--------------no matching models found----------')
        
        loss_list=[dict_i['loss'] for dict_i in best_dict_list]
        if len(loss_list)>0:
            loss_list=[dict_i['loss'] for dict_i in best_dict_list]
            try:
                n_list=[dict_i['datagen_dict']['train_n'] for dict_i in same_modelxy_dict_list]
                batchcount_list=[1 for _ in range(len(n_list))]
            except:
                n_list=[dict_i['datagen_dict']['batch_n'] for dict_i in same_modelxy_dict_list]
                batchcount_list=[dict_i['datagen_dict']['batchcount'] for dict_i in same_modelxy_dict_list]

            batchbatchcountlist=[];naivelosslist=[]
            for dict_i in same_modelxy_dict_list:
                try:
                    i_batchbatch1=dict_i['datagen_dict']['batchbatchcount']
                    i_batchbatch2=dict_i['modeldict']['maxbatchbatchcount']
                    if i_batchbatch1<i_batchbatch2:
                        batchbatchcountlist.append(i_batchbatch1)
                    else: batchbatchcountlist.append(i_batchbatch2)
                except: batchbatchcountlist.append(1)
                try:
                    naivelosslist.append(dict_i['naiveloss'])
                except: naivelosslist.append(1)
                           
            n_wt_loss_list=[self.do_nwt_loss(loss_list[i],n_list[i],batchcount_list[i],batchbatchcount=batchbatchcountlist[i],naiveloss=naivelosslist[i]) for i in range(len(loss_list))]
            lowest_n_wt_loss=min(n_wt_loss_list)
            best_dict=best_dict_list[n_wt_loss_list.index(lowest_n_wt_loss)]
            
            #try:print(f'optimization dict with lowest loss:{best_dict["loss"]}, n:{best_dict["ydata"].shape[0]}was last saved{best_dict["whensaved"]}')
            printstring=f"optimization dict with lowest loss:{best_dict['loss']}, n:{best_dict['datagen_dict']['batch_n']}was last saved{best_dict['when_saved']}"
            print(printstring)
            self.logger.info(printstring)

            #p#rint(f'best_dict:{best_dict}')
            if replace==1:
                self.logger.info("overriding start parameters with saved parameters")
                optimizedict=self.rebuild_hyper_param_dict(optimizedict,best_dict['params'],verbose=0)
            else:
                print('continuing without replacing parameters with their saved values')
        return(optimizedict)
    
    


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
    
   
    def merge_list_of_listdicts(self,listoflistdicts):#a listdict is a dict with each val an unordered list
        merged_listdict={}
        for listdict in listoflistdicts:
            for key in listdict:
                if not key in merged_listdict:
                    merged_listdict[key]=listdict[key]
                else:
                    merged_listdict[key].append(listdict[key])
        return merged_listdict


    def recursive_build_model_save_pathlist(self,startdirectory):
        try:
            model_save_pathlist=[]
            for rootpath,subdirs,files in os.walk(startdirectory):
                for newroot in subdirs:
                    model_save_pathlist.extend(self.recursive_build_model_save_pathlist(os.path.join(rootpath,newroot)))
                for file in files:
                    if re.search('model_save',file):
                        model_save_pathlist.append(os.path.join(rootpath,file))
            self.logger.debug(f'len(model_save_pathlist):{len(model_save_pathlist)}')
            return model_save_pathlist
        except:
            self.logger.exception('')
            assert False, 'halt'

    def merge_and_condense_saved_models(self,merge_directory=None,pathlist=None,species_name='',
                                        save_directory=None,condense=None,recondense=None,verbose=None,recursive=None,returnlist=0):
        try:
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
                save_directory=merge_directory
                    #os.makedirs(save_directory)
            if condense==None or condense=='no':
                condense=0
            if condense=='yes':
                condense=1
            if verbose==None or verbose=='no':
                verbose=0
            if verbose=='yes':
                verbose=1
            if pathlist is None:

                if recursive:
                    model_save_filelist=self.recursive_build_model_save_pathlist(merge_directory)
                    modelfile_count=len(model_save_filelist)
                    print(f'len(model_save_filelist):{len(model_save_filelist)}')
                    print(f'model_save_filelist[0:5]:{model_save_filelist[0:5]}')
                else:
                    pathlist=os.listdir(merge_directory)

                    model_save_filelist=[os.path.join(merge_directory,name_i) for name_i in pathlist if re.search('model_save',name_i)]
                    modelfile_count=len(model_save_filelist)
            else:
                model_save_filelist=pathlist
                modelfile_count=len(model_save_filelist)



            if len(model_save_filelist)==0:
                self.logger.warning(f'0 models found in save_directory:{merge_directory} when merging')



            list_of_saved_models=[]
            for file_i in model_save_filelist:
                #file_i_name=os.path.join(merge_directory,file_i)
                try:
                    with open(file_i,'rb') as savedfile:
                        try: 
                            saved_model_list=pickle.load(savedfile)
                            self.logger.debug(f'for file_i:{file_i}, len(saved_model_list):{len(saved_model_list)}')
                            #print(f'file_i:{file_i} has {len(saved_model_list)} saved model(s)')
                        except:
                            print(f'warning!saved_model_list{file_i} could not pickle.load')
                            self.logger.exception(f'error in {__name__}')
                            saved_model_list=None

                except:
                    saved_model_list=None
                    self.logger.exception('')

                if saved_model_list:    
                    if condense:
                        condense_saved_model_list=self.condense_saved_model_list(saved_model_list, help_start=0, strict=1,verbose=verbose,endsort=1)
                        self.logger.debug(f'for file_i:{file_i}, len(condense_saved_model_list):{len(condense_saved_model_list)}')
                        list_of_saved_models.extend(condense_saved_model_list[:condense])
                    else:
                        list_of_saved_models.extend(saved_model_list)

            if recondense:
                list_of_saved_models=self.condense_saved_model_list(list_of_saved_models,help_start=0,strict=1,verbose=verbose)

            if returnlist:
                return list_of_saved_models

            if os.path.exists(merged_path):
                merged_path=self.helper.getname(merged_path)

            with open(merged_path,'wb') as newfile:
                print(f'writing new_model_list length:{len(list_of_saved_models)} to newfile:{newfile}')
                pickle.dump(list_of_saved_models,newfile)
            return merged_path
        except:self.logger.exception('')

    def get_nwt_loss(self,saved_model_list):
        nwt_list=[]
        for full_model_i in saved_model_list:
            loss_function=full_model_i['modeldict']['loss_function']
            loss_i=full_model_i['lossdict'][loss_function]
            try:
                i_naiveloss=full_model_i['naiveloss']
            except:
                i_naiveloss=1
            i_n=full_model_i['datagen_dict']['batch_n']
            i_batchcount=full_model_i['datagen_dict']['batchcount']
            try:
                i_batchbatch1=full_model_i['datagen_dict']['batchbatchcount']
                i_batchbatch2=full_model_i['modeldict']['maxbatchbatchcount']
                if i_batchbatch1<i_batchbatch2:
                    i_batchbatchcount=i_batchbatch1
                else: i_batchbatchcount=i_batchbatch2
            except: i_batchbatchcount=1
            nwt_list.append(self.do_nwt_loss(loss_i,i_n,i_batchcount,naiveloss=i_naiveloss,batchbatchcount=i_batchbatchcount))
        return nwt_list
    
    def condense_saved_model_list(self,saved_model_list,help_start=1,strict=None,verbose=None,endsort=0,threshold=None):
        try:
            if saved_model_list==None:
                return []
            if verbose=='yes': verbose=1
            if verbose==None or verbose=='no':verbose=0
            assert type(verbose)==int, f'type(verbose) should be int but :{type(verbose)}'
            if strict=='yes':strict=1
            if strict=='no':strict=0

            nwt_list=self.get_nwt_loss(saved_model_list)


            modelcount=len(saved_model_list)        
            keep_model=[1 for _ in range(modelcount)]
            for i in range(modelcount):
                if verbose>0:
                    print(f'{100*i/modelcount}%',end=',')

                if keep_model[i]:
                    for j in range(modelcount-(i+1)):
                    #for j,full_model_j in enumerate(saved_model_list[i+1:]):
                        j=j+i+1
                        if keep_model[j]:
                            matchlist=self.do_partial_match([saved_model_list[i]],saved_model_list[j],help_start=help_start,strict=strict)
                            #if full_model_i['modeldict']==full_model_j['modeldict']:
                            if len(matchlist)>0:
                                iwt=nwt_list[i]
                                jwt=nwt_list[j]

                                if iwt<jwt:
                                    if verbose>1:
                                        print('model j loses')
                                    keep_model[j]=0
                                else:
                                    if verbose>1:
                                        print('model i loses')
                                    keep_model[i]=0
                                    break

            final_keep_list=[model for i,model in enumerate(saved_model_list) if keep_model[i]]
            self.logger.debug(f'len(final_keep_list):{len(final_keep_list)}')
            if verbose>0:
                print(f'len(final_keep_list):{len(final_keep_list)}')
            if endsort:
                final_nwt_list=[nwt_list[i] for i in range(modelcount) if keep_model[i]]
                final_keep_list,final_nwt_list=zip(*[(model,nwt_loss) for nwt_loss,model in sorted(zip(final_nwt_list,final_keep_list))])
            if threshold:
                if type(threshold) is str and threshold=='naiveloss':
                    final_keep_list=[model for model in final_keep_list if model['loss']<model['naiveloss']]
                if type(threshold) in  [float,int]:
                    final_keep_list=[model for model in final_keep_list if model['loss']<threshold]
                #condensed_model_list.sort(key=lambda savedicti: savedicti['loss'])
            self.logger.debug(f'finally, len(final_keep_list):{len(final_keep_list)}')
            return final_keep_list
        except:self.logger.exception('')

    def do_nwt_loss(self,loss,n,batch_count=1,naiveloss=1,batchbatchcount=1):
        batch_count=batch_count*batchbatchcount
        if not type(loss) is float:
            return 10**301
        
        else:
            #p#rint('type(loss)',type(loss))
            return np.log(loss/naiveloss+1)/(np.log(n**2*batch_count)**1.5)
    
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
            if not tryagain==None:
                return tryagain
        else:
            same_modeldict_list=[saved_dict_list[i] for i,optdict_i in enumerate(saved_dict_list) if \
                                 self.are_dicts_equal(optdict_i['modeldict'],optimizedict['modeldict'])]
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
        string_list=[('modeldict','hyper_param_form_dict'),('modeldict','ykern_grid'),('modeldict','maxbatchbatchcount'),('datagen_dict','batchbatchcount'),('datagen_dict','batchcount'),('datagen_dict','seed'),('datagen_dict','batch_n'),
                     ('modeldict','spatialtransform'),('modeldict','ykern_grid'),
                     ('modeldict','xkern_grid'),('datagen_dict','batchcount')]
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
            
                
                  
    def do_dict_override(self,old_dict,new_dict,verbose=None,recursive=None,replace=None,deletekey=None,justfind=None):#key:values in old_dict replaced by any matching keys in new_dict, otherwise old_dict is left the same and returned.
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
                    if justfind:
                        return False
            
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
                        if justfind:return False
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
            if justfind:return True
            #p#rint(f'old_dict_copy{old_dict_copy}')
            return old_dict_copy
    

        
class KernelCompare(KernelOptModelTools,KernelParams):
    def __init__(self,directory=None,myname=None, source=None):
        if directory is None:
            directory=os.getcwd()
        
        if myname:
            try:
                self.logger=logging.getLogger(__name__)
                self.logger.info('starting new KernelCompare object')
            except:
                print(traceback.format_exc())
        else:     
            _name=''
            logdir=os.path.join(directory,'log')
            if not os.path.exists(logdir): os.mkdir(logdir)
            handlername=os.path.join(logdir,'KernelCompare'+_name+'.log')
            logging.basicConfig(
                handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
                level=logging.WARNING,
                format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%Y-%m-%dT%H:%M:%S')
            self.logger = logging.getLogger(handlername)
            self.logger.info('starting new KernelCompare log')

            

        
        if source is None:
            source='pisces'
        self.source=source
        self.pname=myname
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
                    self.logger.warning('initializing monte carlo datagen_dict',exc_info=True)
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
        model_run_dict_list=self.restructure_small_n_species(model_run_dict_list)
        self.rebuild_Ndiff_depth_bw(model_run_dict_list)
        return model_run_dict_list
    
    def rebuild_Ndiff_depth_bw(self,model_run_dict_list):
        
        for model_run_dict in model_run_dict_list:
            optdict=model_run_dict['optimizedict']
            modeldict=optdict['modeldict'] 
            Ndiff_depth_bw=optdict['hyper_param_dict']['Ndiff_depth_bw']
            if not type(Ndiff_depth_bw) is np.ndarray:
                Ndiff_type=modeldict['Ndiff_type']
                if Ndiff_type=='recursive':
                    depth_p=1
                elif Ndiff_type=='product':
                    depth_p=modeldict['max_bw_Ndiff']
                else:
                    assert False,f'not expecting Ndiff_type:{Ndiff_type}'
                optdict['hyper_param_dict']['Ndiff_depth_bw']=Ndiff_depth_bw*np.ones([depth_p,],dtype=np.float64)
        #return model_run_dict_list
                
                
            
    
    
    def restructure_small_n_species(self,model_run_dict_list,validate=1):
        '''
        checks each model in the model_run_dict_list to make sure the associated species has enough observations. 
        if not, batchcount reduced as low as 2 and if species still don't have enough observations, they are dropped.
        '''
        newmodelrundictlist=[]
        species_n_dict={}
        for model_run_dict in model_run_dict_list:
            toosmall=0
            datagen_dict=model_run_dict['datagen_dict']
            spec=datagen_dict['species']
            
            optdict=model_run_dict['optimizedict']
            modeldict=optdict['modeldict']         
            
            batchcount=datagen_dict['batchcount']
            batch_n=datagen_dict['batch_n']
            if self.max_maxbatchbatchcount:
                if not 'max_maxbatchbatchcount' in datagen_dict:
                    datagen_dict['max_maxbatchbatchcount']=self.max_maxbatchbatchcount
            
            
                    
            min_n=batchcount*batch_n*(1+validate) # in order to validate, we must have at least 2 batchbatches.    
                
                
            
            '''try: 
                dg_validate_batchcount=datagen_dict['validate_batchcount']
                validate=1
            else: 
                dg_validate_batchcount=None
                validate=0
            if dg_validate_batchcount in ['none',None]:
                validate_batchcount=batchcount
            elif type(dg_validate_batchcount) is int:
                validate_batchcount=dg_validate_batchcount
            else: validate_batchcount=0
                
            
            try: 
                dg_validate_batchbatchcount=datagen_dict['validate_batchbatchcount']
            except:
                dg_validate_batchbatchcount=1
            if type(dg_validate_batchbatchcount) is int:
                validate_batchbatchcount=dg_validate_batchbatchcount
            elif dg_validate_batchbatchcount=='remaining'
            '''
                
                
            if spec in species_n_dict:
                spec_n=species_n_dict[spec]
            else:
                datagen_obj=dg.datagen(datagen_dict) # create a new object each time, in part to reseed suffle
                spec_n=datagen_obj.species_n
                datagen_dict=datagen_obj.datagen_dict_expanded
                species_n_dict[spec]=spec_n
            if spec_n<min_n:
                newbatchcount=spec_n//(batch_n*(1+validate)) # ensuring at least 2 batchbatches
                self.logger.info(f'for species:{spec}, newbatccount:{newbatchcount}')
                if newbatchcount>1:
                    datagen_dict['batchcount']=newbatchcount
                else:
                    toosmall=1
            if not toosmall:
                model_run_dict['datagen_dict']=datagen_dict
                newmodelrundictlist.append(model_run_dict)
            else:
                self.logger.info(f'for species:{spec_n} is too small!')
        return newmodelrundictlist
    
    
    
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
        #force_start_params=1 #in mycluster instead
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
