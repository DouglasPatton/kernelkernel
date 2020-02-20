import os
import json,pickle
import traceback
import numpy as np


class PickleToJson:
    def __init__(self,startdir=None,savedir=None):
        if savedir is None:
            print('starting')
            savedir=os.path.join(os.getcwd(),'..','json')
        if startdir is None:
            startdir=os.getcwd()
        print(startdir)
        print(savedir)
        #print(*os.walk(startdir))
        dirpath,dirnames,filenames=zip(*os.walk(startdir))
        
        
        if not os.path.exists(savedir):os.mkdir(savedir)
        #print('dirnames',dirnames[0])   
        #print('filenames',filenames[0])
        #print('dirpath',dirpath[0])
       
        
        for filename in filenames[0]:
            if filename[0]!='.':
                path=os.path.join(dirpath[0],filename)
                savepath=os.path.join(savedir,filename+'.json')
                if filename[-4:]!='.npy':
                    try:
                        with open(path,'rb') as f:
                            filedata=pickle.load(f)
                        try:
                            with open(savepath,'w') as f:
                                json.dump(filedata,f)
                        except:
                            print(f"JSON couldn't dump {path}")
                            print(traceback.format_exc())
                            pass
                    except:
                        print(f"PICKLE couldn't load {path}")
                        pass
                if filename[-4:]=='.npy':
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

        for directory in dirnames[0]:  # depth first traversal of filetree....
            if directory[0]!='.':
                #print(directory)
                nextsavedir=os.path.join(savedir,directory)
                if not os.path.exists(nextsavedir):
                    os.mkdir(nextsavedir)
                
                nextstartdir=os.path.join(startdir,directory)
                PickleToJson(startdir=nextstartdir,savedir=nextsavedir)

def main(startdir=None,savedir=None):
    PickleToJson()
    
if __name__=="__main__":
    thispath=os.path.realpath(__file__)
    gitsdir=r'C:\Users\DPatton\gits'.split('\\')
    lastdir=thispath.split('\\')[-1]
    savedir=os.path.join(*gitsdir,'lastdir')
    main(startdir=thispath,savedir=gitsdir)
