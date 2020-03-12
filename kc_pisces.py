from kc_helpers import KCHelper
from helpers import Helper
class KCPisces():
    def __init__(self):
        pass
       
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
        
        
    def sort_by_species():
        pass

    def process_pisces_models(self,startpath,condense=0,recondense=0,recondense2=0):
        
        #species_model_save_path_dict_list=[]
        species_model_save_path_dict=self.split_pisces_model_save_path_dict(startpath)
        #species_model_save_path_dict_list.append(species_model_save_path_dict)
        #species_model_save_path_dict=self.merge_list_of_listdicts(species_model_save_path_dict_list)
        #full_species_model_save_path_dict=self.update_species_model_save_path_dict(species_model_save_path_dict)
        try:
            all_species_model_merge_dict=self.getpickle(self.all_species_model_merge_dict_path)
        except FileNotFoundError:
            all_species_model_merge_dict={}
        except:
            self.logger.exception('')
            assert False, 'unexpected exception'
            
        speciescount=len(species_model_save_path_dict)
        
        for i,species in enumerate(species_model_save_path_dict):
            
            pathlist=species_model_save_path_dict[species]
            pathcount=len(pathlist)
            self.logger.debug(f'starting merge; pathcount:{pathcount},species:{species},({i}/{speciescount})')
            mergedlist=self.merge_and_condense_saved_models(
                species_name=species,
                pathlist=pathlist,
                condense=condense,
                recondense=recondense,returnlist=1
                )
            if species not in all_species_model_merge_dict:
                all_species_model_merge_dict[species]=[]
            all_species_model_merge_dict[species].append(mergedlist)
            if recondense2:
                all_species_model_merge_dict[species]=self.condense_saved_model_list(all_species_model_merge_dict[species], help_start=0, strict=1,verbose=verbose)
        self.savepickle(all_species_model_merge_dict,self.all_species_model_merge_dict_path)
            
    
    
    def print_pisces_all_species_model_merge_dict(self,shortlist=[]):
        all_species_model_merge_dict=self.getpickle(self.all_species_model_merge_dict_path)
        speciescount=len(all_species_model_merge_dict)
        for i,species in enumerate(all_species_model_merge_dict):
            model_list=all_species_model_merge_dict[species]
            
            self.logger.debug(f'printing {species}; ({i}/{speciescount}), with {len(model_list)} models.')
            self.print_model_save(model_save_list=model_list,shortlist=shortlist,species=species)
            
    
        
    def split_pisces_model_save_path_dict(self,startdir):
        self.logger.debug('starting to add species names')
        self.addspecies_name_and_resave(startdir=startdir)
        self.logger.debug('finished adding species names')
        
        
        model_save_pathlist=self.recursive_build_model_save_pathlist(startdir)
        
        #if not species_model_save_path_dict:
        species_model_save_path_dict={}
        for path in model_save_pathlist:
            #regex_genus_species=r'species-[(a-z)]+\s[a-z]+_'
            regex_genus_species=r'species-[(a-z)]+(\s[a-z]+)+_'# this one will match 3 word species like cyprinella venusta cercostigma
            regex_genus_species=r'species-[(a-z)]+(\s[\-\.a-z]+)+_'# this one will match species with - or . in name
            searchresult=re.search(regex_genus_species,path)
            if searchresult:
                species_genus_slicer=slice(searchresult.start()+8,searchresult.end()-1)
                spec_name=path[species_genus_slicer]
            
                if not spec_name in species_model_save_path_dict:
                    self.logger.debug(f'adding spec_name:{spec_name} to species_model_save_path_dict which has len:{len(species_model_save_path_dict)}')
                    species_model_save_path_dict[spec_name]=[path]
                species_model_save_path_dict[spec_name].append(path)
            else:
                self.logger.debug(f'split_pisces_speices_model_save ignoring path:{path}')
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
