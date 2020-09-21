import os
import re
from zipfile import ZipFile
import csv
import json
import io
import logging
from geogtools import GeogTool as GT
from copy import deepcopy


class SCDataTool:
    def __init__(self,singlefile=1,zipdir=None,sc_data_dir=None,groupby_huc8=1):
        self.cwd=os.getcwd()
        self.logdir=os.path.join(self.cwd,'log'); 
        if not os.path.exists(self.logdir):os.mkdir(self.logdir)
        handlername=os.path.join(self.logdir,'streamcat_unzip_tool.log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=1)],
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.logger.info('streamcat_unzip_tool starting new logger')
        
        
        if zipdir is None:
            zipdir=self.cwd
        self.zipdir=zipdir
        if sc_data_dir is None:
            sc_data_dir=os.path.join(os.getcwd(),'sc_data')
        if not os.path.exists(sc_data_dir): os.mkdir(sc_data_dir)
        print('SCDataTool, sc_data_dir', sc_data_dir)
        self.sc_data_dir=sc_data_dir
        self.groupby_huc8=groupby_huc8
        if self.groupby_huc8:
            try: self.gt
            except: self.gt=GT()
            try:self.gt.huc12comiddict
            except:self.gt.gethuc12comiddict()
            try: self.gt.huchuc
            except:self.gt.build_huchuc()
            try:self.gt.rvrs_huc12comiddict
            except: self.gt.reverse_huc12comid()
            
            
            
            self.huc2_huc8_comid_dict={}
        else:
            self.huc2_comid_dict={}
            
    
    def build(self):
        filelist=[os.path.join(self.zipdir,zipfile) for zipfile in os.listdir(self.zipdir)]
        dotzipfilelist=[file for file in filelist if file[-4:]=='.zip']
        #print(len(dotzipfilelist))
        datadict={}
        huc2_pathdict={}
        for _file in dotzipfilelist:
            huc2=self.getHuc2(_file)
            try:
                huc2_pathdict[huc2].append(_file)
            except KeyError:
                huc2_pathdict[huc2]=[_file]
        for huc2,filelist in huc2_pathdict.items():
            skiphuc2=self.checkhuc2(huc2)
                #add file checker here
            if not skiphuc2:
                huc2owndict={};meta_huc2owndict={}
                print(f'starting HUC{huc2} with {len(filelist)} files...',end='')
                for i,_file in enumerate(filelist):
                    print(i+1,end=',')
                    #print(f'opening {_file}')
                    unzip_byte_list=[]
                    with ZipFile(_file) as zf:
                        for name in zf.namelist():
                            unzip_byte_list.append(zf.read(name))
                    for unzip_byte in unzip_byte_list:  
                        string_data=io.TextIOWrapper(io.BytesIO(unzip_byte),  newline='')
                        odict=csv.DictReader(string_data)

                        for row in odict:
                            #print(row)
                            comid=int(row['COMID']) # need to rebuild with string
                            #if not comid in self.comid_dict:
                            #    self.comid_dict[comid]={}
                            
                            if self.groupby_huc8:
                                try:
                                    huc12=self.gt.rvrs_huc12comiddict[comid]
                                    skiprow=0
                                except:
                                    self.logger.exception(f'could not locate comid:{comid} in huc2-:{huc2}, _file:{_file}')
                                    skiprow=1
                                if not skiprow:
                                    huc8=huc12[0:8]
                                    try:
                                        huc8owndict=huc2owndict[huc8]
                                        meta_huc8owndict=meta_huc2owndict[huc8]
                                        try:
                                            comid_data=huc8owndict[comid]
                                        except:
                                            comid_data={}; meta_data={}
                                    except:
                                        huc8owndict={}
                                        meta_huc8owndict={}
                                        comid_data={};meta_data={}
                                    for var,val in row.items():
                                        if var.lower() not in ['catareasqkm','wsareasqkm','catpctfull','wspctfull','comid']:
                                            comid_data[var]=val
                                        else: 
                                            if not var.lower()=='comid':meta_data[var]=val
                                    huc8owndict[comid]=comid_data
                                    meta_huc8owndict[comid]=meta_data

                                    huc2owndict[huc8]=huc8owndict
                                    meta_huc2owndict[huc8]=meta_huc8owndict
                            else:
                                try:
                                    comid_data=huc2owndict[comid]
                                except KeyError:
                                    comid_data={};meta_data={}
                                except:
                                    assert False, 'halt, unexpected error'
                                for var,val in row.items():
                                    if var.lower() not in ['catareasqkm','wsareasqkm','catpctfull','wspctfull','comid']:
                                        comid_data[var]=val
                                    else: 
                                        if not var.lower()=='comid':meta_data[var]=val

                                huc2owndict[comid]=comid_data
                                meta_huc2owndict[comid]=meta_data
                print('...',end='')
                self.savehuc2owndict(huc2owndict,meta_huc2owndict)
                print(f'. huc{huc2} complete.')
            else:
                print(f'huc{huc2} streamcat already exists, skipping.')
    
    def makeSCHucSavePath(self,huc):
        path=os.path.join(self.sc_data_dir,f'streamcat_huc{len(huc)}-{huc}_dict.json')
        return path
    
    
    def checkhuc2(self,huc2):
        huc8list=self.gt.huchuc['huc2_huc8dict'][huc2]
        pathlist=[]
        for huc8 in huc8list:
            path=self.makeSCHucSavePath(huc8)
            if os.path.exists(path):
                pathlist.append(path)
            else:
                self.logger.warning(f'file does not exist at path:{path}')
                return False
        for path in pathlist:
            try:self.openjson(path)
            except:
                self.logger.exception(f'could not open json at path:{path}')
                return False
        return True
            
            
    def savehuc2owndict(self,huc2owndict,meta_huc2owndict):
        if self.groupby_huc8:
            for huc8,comid_data in huc2owndict.items():
                path=self.makeSCHucSavePath(huc8)
                meta_path=path[:-5]+'_meta.json'
                self.savejson(comid_data,path)
                meta_data=meta_huc2owndict[huc8]
                self.savejson(meta_data,meta_path)
            
        else:
            self.savejson(huc2owndict,path)
            meta_path=path[:-5]+'_meta.json'
            self.savejson(meta_huc2owndict,meta_path)
        return
    
    
    def savejson(self,thing,path,retries=0):
        
        while os.path.exists(path):
            path=path[:-5]+'_1.json'
        with open(path,'w') as f:
            json.dump(thing,f)
        try: 
            result=self.openjson(path)
            self.logger.info(f'verified save for path:{path} with type(result):{type(result)}')
        except:
            self.logger.exception(f'error saving to path:{path}')
            if retries<4:
                return self.savejson(thing,path,retries=retries+1)
            else:
                self.logger.error(f'after {retries+1} attempts, could not save to path:{path}')
        
        return
    
    def openjson(self,path):
        with open(path,'r') as f:
            thing=json.load(f)
        return thing
            
    def verifyjsonresults(self,filelist=None):
        if filelist is None:filelist=os.listdir(self.sc_data_dir)
        jsonlist=[os.path.join(self.sc_data_dir,_file) for _file in filelist if _file[-5:]=='.json']
        for jsonfile in jsonlist:
            try:
                self.openjson(jsonfile)
                self.logger.info(f'sucessfully opened {jsonfile}')
            except:
                self.logger.exception(f'failed jsonfile: {jsonfile}')
                print(f'failed file: {jsonfile}')

                
                
        
            
    def getHuc2(self,path):
        srch=re.search('Region',path)
        return path[srch.end():srch.end()+2]
        
        
if __name__=="__main__":
    tool=SCDataTool(zipdir='/home/dp/Public/streamcat',sc_data_dir='sc_data')
    tool.build()

        
        