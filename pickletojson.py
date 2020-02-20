import os
import json,pickle
import traceback
import numpy as np


class PickleToJson:
    def __init__(self,startdir=None,savedir=None):
        if savedir is None:
            print('starting')
            savedir=os.path.join(os.getcwd(),'json')
        if startdir is None:
            startdir=os.getcwd()
        _dirpath,_dirnames,_filenames=os.walk(startdir)
        
        
        if not os.path.exists(savedir):os.mkdir(savedir)
           
        for directory in enumerate(dirnames): # depth first traversal of filetree....
            nextsavedir=os.path.join(savedir,directory)
            nextstartdir=os.path.join(startdir,directory)
            PickleToJson(startdir=nextstartdir,savedir=nextsavedir)
        
        for filename in filenames:
            path=os.path.join(dirpath,filename)
            if file[-4:]!='.npy':
                try:
                    with open(path,'rb') as f:
                        filedata=pickle.load(f)
                    try:
                        with open(os.path.join(self.jsondir,file[:-7]+'.json'),'w') as f:
                            json.dump(filedata,f)
                    except:
                        print(f"JSON couldn't dump {path}")
                        print(traceback.format_exc())
                except:
                    print(f"PICKLE couldn't load {path}")
            if file[-4:]=='.npy':
                try:
                    data_array=np.load(os.path.join(directory,file))
                    datalist=data_array.tolist()
                    try:
                        with open(os.path.join(self.jsondir,file[:-4]+'.json'),'w') as f:
                            json.dump(datalist,f)
                    except:
                        print(f"JSON couldn't dump {path}")
                        print(traceback.format_exc())
                except:
                    print(f"numpy couldn't load {path}")
        
        
           
def main(startdir=None,savedir=None):
    PickleToJson()
    
if __name__=="__main__":
    thispath=os.path.realpath(__file__)
    gitsdir=r'C:\Users\DPatton\gits'.split('\\')
    lastdir=thispath.split('\\')[-1]
    savedir=os.path.join(*gitsdir,'lastdir')
    main(startdir=thispath,savedir=gitsdir)
