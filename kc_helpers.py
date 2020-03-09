
class KCHelper():
    def __init__(self):
        pass
    
    
    
    def getpickle(self,path):
        with open(path,'rb') as f:
            result=pickle.load(f)
        return result
    
    def savepickle(self,thing,path):
        with open(path,'wb') as f:
            pickle.dump(thing,f)
        return
    
    
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
    
    def remove_model_fromsavedict(self,filename,flatdictstring,overwrite=0,verbose=0):
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
                
        match_dict=self.build_override_dict_from_str(flatdictstring,None)
        new_modelsave_list=[]
        for savedict in modelsave_list:
            if not self.do_dict_override(savedict,match_dict,verbose=verbose,justfind=1):
                # returns true or false if justfind=1
                new_modelsave_list.append(savedict)
        for j in range(10):
            try:
                with open(os.path.join(self.kc_savedirectory,writefilename),'wb') as modelsavefile:
                    pickle.dump(new_modelsave_list,modelsavefile)
                return
            except:
                if j==9:
                    self.logger.exception(f'error in {__name__}')
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
    
    
