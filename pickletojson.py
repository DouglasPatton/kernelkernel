import os
import json,pickle
import traceback
import numpy as np
import traceback

class PickleToJson:
    def __init__(self,startdir=None,savedir=None):
        if savedir is None:
            print('starting')
            savedir=os.path.join(os.getcwd(),'..','json')
        if startdir is None:
            startdir=os.getcwd()
        self.startdir=startdir
        self.savedir=savedir
        print(startdir)
        print(savedir)
        #print(*os.walk(startdir))
        
    def recursive_dotpickle_finder(startdir):
        
        for rootpath,subdirs,files in os.walk(self.startdir):
            for newroot in subdirs:
                model_save_pathlist.extend(self.recursive_dotpickle_finder(os.path.join(rootpath,newroot)))
            for file in files:
                if file[-7:]=='.pickle':
                    model_save_pathlist.append(os.path.join(rootpath,file))
        
    
    def makejson(self):
        
        
        
        dirpath,dirnames,_filenames=zip(*os.walk(self.startdir))
            
            
        if not os.path.exists(self.savedir):os.mkdir(self.savedir)
        #print('dirnames',dirnames[0])   
        #print('_filenames',_filenames[0])
        #print('dirpath',dirpath[0])

        print('file count:',len(_filenames[0]))
        for _filename in _filenames[0]:
            #if _filename[0]!='.':
            try:
                path=os.path.join(dirpath[0],_filename)
                savepath=os.path.join(self.savedir,_filename+'.json')
                if _filename[-4:]!='.npy' and _filename[-3:]!='.py':
                    try:
                        with open(path,'rb') as f:
                            rawdata=pickle.load(f)
                        the_data=self.prep_json(rawdata)    
                        #print('here')
                        print(the_data)
                        try:
                            with open(savepath,'w') as f:
                                json.dump(the_data,f)
                        except:
                            print(f"JSON couldn't dump {path}")
                            print(traceback.format_exc())
                            pass
                    except:
                        print(f"PICKLE couldn't load {path}")
                        print(traceback.format_exc())
                if _filename[-4:]=='.npy':
                    try:
                        data_array=np.load(path)
                        datalist=data_array.tolist()
                        try:
                            with open(savepath,'w') as f:
                                json.dump(datalist,f)
                        except:
                            print(f"JSON couldn't dump {path}")
                            #print(traceback.format_exc())
                            pass
                    except:
                        print(f"numpy couldn't load {path}")
                        pass
            except:
                print(traceback.format_exc())
        return
          
    def prep_json(self,rawdata):
        
        try:
            if type(rawdata) is np.ndarray:
                return ('numpyarray:',list(rawdata))
            elif type(rawdata) is list:
                for i,item in enumerate(rawdata):
                    rawdata[i]=self.prep_json(item)
                return rawdata
            elif type(rawdata) is tuple:
                rawdata=list(rawdata)
                for i,item in enumerate(rawdata):
                    rawdata[i]=self.prep_json(item)
                return tuple(rawdata)
            elif type(rawdata) is dict:
                for key,val in rawdata.items():
                    rawdata[key]=self.prep_json(val)
                return rawdata
            else: return rawdata
        except:
            print(traceback.format_exc())
            
            
                




    
    
if __name__=="__main__":
    thispath=os.path.realpath(__file__)
    gitsdir=r'C:\Users\DPatton\gits'.split('\\')
    lastdir=thispath.split('\\')[-1]
    #savedir=os.path.join(*gitsdir,'lastdir')
    PickleToJson().makejson()
