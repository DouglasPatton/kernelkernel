import traceback,logging
from copy import deepcopy
import os

class Helper:
    def __init__(self,):
        self.logger=logging.getLogger(__name__)
        self.logger.debug('Helper object started')

    def do_dict_override(self,old_dict,new_dict,verbose=None,recursive=None,inplace=None):
        '''
        key:values in old_dict replaced by any matching keys in new_dict, otherwise old_dict is left the same and returned.
        Original version with more complex options found in kernelkernel/pisces kernelcompare.py
        This option supports substitution of dictionaries from new_dict to old_dict when
            old_dict does not have a dict as its original value
        '''
        if inplace is None or inplace==0:
            old_dict_copy=deepcopy(old_dict)
        else:
            old_dict_copy=old_dict
        
        vstring=''
        if new_dict==None or new_dict=={}:
            if verbose==1:
                print(f'vstring:{vstring}, and done1')
            if recursive==1:
                return old_dict_copy, vstring
            else: 
                return old_dict_copy
        if 'geometry' in new_dict:#clear out the example geometry:huc12
            old_dict_copy['geometry']=self.do_dict_override(old_dict_copy['geometry'],{'hucID':None})
        
        for key,val in new_dict.items():
            if verbose==1:
                vstring=vstring+f":key({key})"
            
            
            if not type(old_dict_copy) is dict: 
                oldval=f'{key} not in old_dict'
                old_dict_copy=val
                
            elif val=={} and key in old_dict_copy:
                old_dict_copy[key]=val
                if recursive==1:
                    return old_dict_copy, vstring
                else: 
                    return old_dict_copy
                
            elif type(val) is dict and key in old_dict_copy:
                if not type(old_dict_copy[key]) is dict:
                    old_dict_copy[key]=val
                if verbose==1:print(f'val is dict in {key}, recursive call')
                old_dict_copy[key],vstring2=self.do_dict_override(old_dict_copy[key],val,recursive=1,verbose=verbose)
                vstring=vstring+vstring2
            
            else:#
                if key in old_dict_copy:
                    oldval=f'{old_dict_copy[key]}'
                else:oldval=f'key:{key} not found in old_dict'
                old_dict_copy[key]=val
                if verbose==1:
                    #p#rint(f":val({new_dict[key]}) replaces val({oldval})\n")
                    vstring=vstring+f":for key:{key}, val({val}) replaces val({oldval})\n"

                '''except:
                    print(traceback.format_exc())
                    print(f'Warning: old_dict has keys:{[key for key,value in old_dict_copy.items()]} and new_dict has key:value::{key}:{new_dict[key]}')'''
        if verbose==1:
                print(f'vstring:{vstring} and done2')            
        if recursive==1:
            return old_dict_copy, vstring

        else:
            if verbose==1:
                print(f'final:old_dict_copy{old_dict_copy}')
            return old_dict_copy
        
    def getname(self,filename):
        exists=1
        while exists==1:
            if os.path.exists(filename):
                countstr=''
                dotpos=[]#dot position for identifying extensions and 
                splitlist=filename.split('.')
                #for i,char in enumerate(filename):
                #    if char=='.':
                #        dotpos.append(i)
                #        break
                try: 
                    #lastdot=dotpos.pop(-1)
                    prefix=''.join(splitlist[:-1])
                    suffix=splitlist[-1]
                    prefix=filename[:lastdot]
                    suffix=filename[lastdot:]
                except:
                    prefix=filename
                    suffix=''
                _pos=[]
                for i,char in enumerate(prefix):
                    if char=='_':
                        _pos.append(i)
                
                '''try:
                    last_=_pos[-1]
                    firstprefix=prefix[:last_]
                    lastprefix=prefix[last_:]'''
                if len(_pos)>0:    
                    countstr=prefix[_pos[-1]+1:]#slice from the after the last underscore to the end
                    count=1
                    if not countstr.isdigit():
                        countstr='_0'
                        count=0
                else:
                    count=0
                    countstr='_0'
                if count==1:
                    prefix=prefix[:-len(countstr)]+str(int(countstr)+1)
                else:
                    prefix=prefix+countstr
                filename=prefix+suffix
            else:
                exists=0
        return filename