import traceback
#import geopandas as gpd
import requests
import json,pickle
import os
import multiprocessing as mp
import numpy as np
import html
import logging, logging.handlers
import re
from copy import deepcopy
from random import shuffle

class GeogTool:
    def __init__(self,sc_data_dir=None):
        '''try:
            self.comiddatadir
            
        except:
            self.comiddatadir=os.path.join(os.getcwd(),'data','comid_data')
            if not os.path.exists(self.comiddatadir):os.makedirs(self.comiddatadir)'''
        self.cwd=os.getcwd()
        try:
            self.logger
            self.logger = logging.getLogger(__name__)
            self.logger.info('GeogTool starting')
        except:
            
            self.logdir=os.path.join(self.cwd,'log'); 
            if not os.path.exists(self.logdir):os.mkdir(self.logdir)
            handlername=os.path.join(self.logdir,'GeogTool.log')
            logging.basicConfig(
                handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=1)],
                level=logging.DEBUG,
                format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%Y-%m-%dT%H:%M:%S')
            self.logger = logging.getLogger(handlername)
            self.logger.info('GeogTool starting new logger',exc_info=True)
        try: self.geogdatadir
        except:
            self.geogdatadir=os.path.join(os.getcwd(),'NHDplus_data')
        self.NHDplus_path=os.path.join(self.geogdatadir,'NHDplus.pickle')
        self.huc12comiddict_path=os.path.join(self.geogdatadir,'huc12comiddict.pickle')
        self.NHDdbf_path=os.path.join(self.geogdatadir,'HUC12_PU_COMIDs_CONUS.dbf')
        self.NHDhuchuc_path=os.path.join(self.geogdatadir,'NHDhuchuc.pickle')
        if sc_data_dir is None:
            try: self.sc_data_dir
            except: 
                localpath=os.path.join(self.cwd,'..','sc_data')
                if os.path.exists(localpath):
                    self.sc_data_dir=localpath
                else:
                    networkpath=os.path.join('o:','public','streamcat','sc_data')
                    if os.path.exists(networkpath):
                        self.sc_data_dir=networkpath
                    else:
                        assert False, 'cannot locate local streamcat data'
        else: self.sc_data_dir=sc_data_dir
        print("streamcat data directory:",self.sc_data_dir)
            
    def getpickle(self,path):
        with open(path,'rb') as f:
            thefile=pickle.load(f)
        return thefile
    
    def savepickle(self,obj,path):
        with open(path,'wb') as f:
            pickle.dump(obj,f)
            
            
    def savejson(self,thing,path):
        while os.path.exists(path):
            path=path[:-5]+'_1.json'
        with open(path,'w') as f:
            json.dump(thing,f)
        return
    
    def openjson(self,path):
        with open(path,'r') as f:
            thing=json.load(f)
        return thing     
        
    
    def processStreamCat(self,streamcatdata):
        try:
            if type(streamcatdata) is str:
                self.logger.debug(f'streamcatdata type is a str: {streamcatdata}')
                streamcatdata=self.getpickle(streamcatdata)
            outdict={}
            
            for comid,data in streamcatdata.items():
                metricdict={'Ws':{},'Cat':{},'CatRp100':{},'WsRp100':{},'Rp100':{},'other':{}}
                metadict=None
                mkeylist=[*metricdict] #the keys or mkeys
                #print('mkeylist',mkeylist)
                keylengths=[len(key) for key in mkeylist]
                keycount=len(mkeylist)
                regex_y2k=re.compile('20[0-9][0-9]')
                #nlcd_regex_y2k=re.compile('nlcd20[0-9][0-9]')
                    metrics=[*data]
                    for metric in metrics:
                        for k in range(keycount-1):
                            endstring=metric[-keylengths[k]:]
                            #self.logger.critical(f'endstring:{endstring} for metric:{metric}')
                            if endstring==mkeylist[k]:
                                mkey=mkeylist[k]
                                break
                            if k==keycount-2:
                                mkey='other'
                        srch_y=re.search(regex_y2k,metric)
                        if srch_y:
                            yr=metric[srch_y.start():srch_y.end()]
                        else:
                            yr='all'
                        if not yr in metricdict[mkey]:
                            metricdict[mkey][yr]={}
                        metricdict[mkey][yr][metric]=data[metric]
                        
                            
            return outdict
        except:
            self.logger.exception('streamcat error') 
                
                
    def pullStreamCatForComidList(self,huc2comid_dict,meta=0,error_lookup=0):
        try:self.rvrs_huc12comiddict
        except: self.reverse_huc12comid()
        #try: self.huchuc
        #except:self.build_huchuc()
        pathlist=[path for path in os.listdir(self.sc_data_dir) if path[-5:]=='.json']
        SCoutdict={}
        if type(huc2comid_dict) is list:
            returncomiddict=1 # i.e., comd:data rather than a huc8:comid:data dict
            comiddict={}
            comidlist=deepcopy(huc2comid_dict)
            huc2comid_dict={}
            for comid in comidlist:
                huc12=self.rvrs_huc12comiddict[comid]
                huc2=huc12[0:2]
                if huc2 in huc2comid_dict:
                    huc2comid_dict[huc2].append(comid)
                else: huc2comid_dict[huc2]=[comid]
        else: returncomiddict=0        
        for huc2,val in huc2comid_dict.items():
            
            if type(val) is list:
                
                huc8dict={}
                for comid in val:
                    huc8=self.rvrs_huc12comiddict[comid][0:8]
                    try: huc8dict[huc8].append(comid)
                    except KeyError: huc8dict[huc8]=[comid]
                    except: 
                        self.logger.exception('')
                        assert False, "halt, unexpected error"
            elif type(val) is dict:
                huc8dict=val
            elif type(val) is str:
                val=[val]
            huc8comid_dict={}
            for huc8,comidlist in huc8dict.items():
                huc8comid_dict[huc8]={}
                path=self.pathfromhuc8(huc8) # no group by for faster load
                if meta: path=path[:-5]+'_meta.json'
                huc8scdata=self.openjson(path)
                for comid in comidlist:
                    try:
                        comiddata=huc8scdata[str(comid)]
                    except KeyError:
                        self.logger.exception(f'could not find stream cat for huc8:{huc8},comid:{comid}')#,huc8scdata:{huc8scdata}')
                        if error_lookup:
                            message=self.checkNHDPlus(comid)
                            self.logger.critical(message)    
                        comiddata='error'
                    except:
                        assert False, 'halt, unexpected error'
                    if returncomiddict:
                        comiddict[comid]=comiddata
                    else:
                        huc8comid_dict[huc8][comid]=comiddata
            if not returncomiddict:
                SCoutdict[huc2]=huc8comid_dict    
        if returncomiddict:
            return comiddict
        else:
            return SCoutdict    
 

    def pathfromhuc8(self,huc8):
        try:
            self.pathtool
        except:
            import streamcat_unzip_tool as sct
            self.pathtool=sct.SCDataTool(sc_data_dir=self.sc_data_dir,groupby_huc8=0).makeSCHucSavePath
        return self.pathtool(huc8)
        
        
    def gethuc12comiddict(self):
        try: 
            huc12comiddict_path=self.huc12comiddict_path
            self.huc12comiddict=self.getpickle(huc12comiddict_path)
            self.logger.info(f'opening {self.huc12comiddict_path} with length:{len(self.huc12comiddict)} and type:{type(self.huc12comiddict)}')
        except: 
            self.logger.exception(f"{self.huc12comiddict_path} exists but could not open, rebuilding")
            self.buildNHDplus()
        
        return self.huc12comiddict
    
    
    def selectRandComidByHuc(self, comid_count=10000, huc2list=None, huc8count=None, huc8list=None, seed=0, evenshare=1):
        try: self.huc12comiddict
        except: self.gethuc12comiddict()
        try:
            huchuc=self.huchuc
        except:
            try:
                huchuc=self.getpickle(self.NHDhuchuc_path)
            except:
                huchuc=self.build_huchuc()
        huc2_huc8dict=huchuc['huc2_huc8dict']
        huc8_huc12dict=huchuc['huc8_huc12dict']        
        
        if huc2list is None: 
            huc2list=[huc2 for huc2 in huc2_huc8dict if huc2 !='No']
            
                
        huc2count=len(huc2list)
        if huc8list is None:
            if huc8count is None: huc8count=500
            
            huc2_huc8count=max([1,int(huc8count/huc2count)])
            huc8count=huc2_huc8count*len(huc2list)
            if evenshare:
                huc2huc8comid_dict={}
                for huc2 in huc2list:
                    huc2huc8comid_dict[huc2]={}
                    huc8list=self.select_random_huc8(count=huc2_huc8count, huc2list=[huc2],seed=seed, evenshare=evenshare)
                    huc8comidcount=max([1,int(comid_count/(huc2count*len(huc8list)))])
                    huc8comidselect=[]
                    for huc8 in huc8list:
                        
                        huc8comidlist=[]
                        huc12list=huc8_huc12dict[huc8]
                        for huc12 in huc12list:
                            huc8comidlist.extend(self.huc12comiddict[huc12])
                        shuffle(huc8comidlist)
                        huc8comidselect.extend(huc8comidlist[:huc8comidcount])    
                    huc2huc8comid_dict[huc2][huc8]=huc8comidselect
                return huc2huc8comid_dict
                
            else:
                huc8list=self.select_random_huc8(count=huc8count, huc2list=huc2list,seed=seed, evenshare=evenshare)
        for huc8 in huc8list:
            assert False, 'not developed'
            
            huc2comidselectdict={huc2:[] for huc2 in huc2list}
            huc2_comidcount=max([1,int(count/len(huc2list))])
            if evenshare:
                for huc2 in huc2list:
                    huc2_huc8list=[]
                #huc2_huc8count=
                huc8_comidcount=max([1,huc2_comidcount/huc2_huc8count])
        
        
        
            
    
                
    def select_random_huc8(self,count=2,huc2list=None,seed=0,evenshare=1):
        if not seed is None:
            np.random.seed(seed)
        try:
            huchuc=self.huchuc
        except:
            huchuc=self.build_huchuc()
        huc2_huc8dict=huchuc['huc2_huc8dict']
        huc8_huc12dict=huchuc['huc8_huc12dict']
        huc2_huc8dict_select={}
        huc8_huc12dict_select={}
        
        if huc2list is None:
            huc2list=[huc2 for huc2 in huc2_huc8dict]
        else:
            for huc2,huc8list in huc2_huc8dict.items():
                if huc2 in huc2list:
                    huc2_huc8dict_select[huc2]=huc8list
                    for huc8 in huc8list:
                        huc8_huc12dict_select[huc8]=huc8_huc12dict[huc8]
            huc2_huc8dict=huc2_huc8dict_select
            huc8_huc12dict=huc8_huc12dict_select
            
        if evenshare:
            huc2_huc8count=max([1,int(count/len(huc2list))]) # at least 1
            huc8_selection=[]
            for huc2 in huc2list:
                huc8list=huc2_huc8dict[huc2]
                shuffle(huc8list)
            huc8_selection.extend(huc8list[:huc2_huc8count])
        else:
            huc8list=[huc8 for huc8 in huc8_huc12dict]
            shuffle(huc8list)
            huc8_selection=huc8list[:count]
            
        huc8_huc12dict_finalselect={}
        
        for huc8 in huc8_selection:
            huc8_huc12dict_finalselect[huc8]=huc8_huc12dict[huc8]
        return huc8_huc12dict_finalselect
        
            
        
    def reverse_huc12comid(self):
        try: self.huc12comiddict
        except: self.gethuc12comiddict()
        
        self.rvrs_huc12comiddict={comid:huc12 for huc12,comidlist in self.huc12comiddict.items() for comid in comidlist }
        
        
        
    def build_huchuc(self):
        try:
            self.huchuc=self.getpickle(self.NHDhuchuc_path)
            return self.huchuc
        except: pass
        try: self.huc12comiddict
        except: self.gethuc12comiddict()
        huc2_huc8dict={}
        huc8_huc12dict={}
        
        for huc12 in self.huc12comiddict:
            a_huc8=huc12[0:8]
            a_huc2=huc12[0:2]
            try: 
                huc8_huc12dict[a_huc8].append(huc12)
            except KeyError: # if huc8 is new
                huc8_huc12dict[a_huc8]=[huc12]
                try: # iff huc8 is new, add it to  huc2 list
                    huc2_huc8dict[a_huc2].append(a_huc8)
                except KeyError: # iff huc2 is new, create new dict entry for it too.
                    huc2_huc8dict[a_huc2]=[a_huc8]
                except:
                    self.logger.exception(f'unexpected error with huc12:{huc12}, a_huc8:{a_huc8}, a_huc2:{a_huc2}')
            except:
                self.logger.exception(f'unexpected error with huc12:{huc12}, a_huc8:{a_huc8}, a_huc2:{a_huc2}')
        huchuc={'huc2_huc8dict':huc2_huc8dict,'huc8_huc12dict':huc8_huc12dict}
        self.savepickle(huchuc,self.NHDhuchuc_path)
        self.huchuc=huchuc
        return huchuc
        
            
    def checkNHDPlus(self,comid):
        try: self.NHDplus
        except: self.buildNHDplus(setNHDplus_attribute=1)
        message=f'checking NHDplus for comid:{comid}. '
        comidlist=self.NHDplus['COMID'].to_list()
        #huc12list=self.NHDplus['HUC12'].to_list()
        #huc8list=[huc12[0:8] for huc12 in huc12list]
        try:
            comid_data=self.NHDplus.iloc[comidlist.index(comid)]
            
        except ValueError:
            comid_data='comid not found'
        except:
            self.logger.exception('unexpected error')
            assert False, 'halt'
        message+=f'comid_data:{comid_data}'
        return message
        
    
    def buildNHDplus(self,setNHDplus_attribute=0):
        
        savefilename=self.NHDplus_path
        if os.path.exists(savefilename):
            try: 
                NHDplus=self.getpickle(savefilename)
                self.logger.info(f'opening {savefilename} with length:{len(NHDplus)} and type:{type(NHDplus)}')
                # self.logger.info(NHDplus)
            except: 
                self.logger.info(f"{savefilename} exists but could not open, rebuilding")
                
        try: NHDplus
        except:
            filename=self.NHDdbf_path
            self.logger.info(f'starting read of {filename}')
            NHDplus=gpd.read_file(filename)
            self.logger.info('finished read of NHDplus')
            self.logger.info(f'opened {filename} with length:{len(NHDplus)} and type:{type(NHDplus)}')
        if os.path.exists(self.huc12comiddict_path):
            try: 
                self.huc12comiddict=self.getpickle(self.huc12comiddict_path)
                self.logger.info(f'opening {self.huc12comiddict_path} with length:{len(self.huc12comiddict)} and type:{type(self.huc12comiddict)}')
                # self.logger.info(self.huc12comiddict)
                if setNHDplus_attribute:
                    self.NHDplus=NHDplus
                return 
            except: 
                self.logger.info(f"{savefilename} exists but could not open, rebuilding")

        self.logger.info(f'NHDplus.columns.values:{NHDplus.columns.values}')
        
        NHDplusHUC12array=NHDplus.loc[:,('HUC12')].to_numpy(dtype='str')
        self.logger.info('buildng huc12comiddict')
        huc12dict={}
        for comid_idx,huc12 in enumerate(NHDplusHUC12array):
            if len(huc12)==11:huc12='0'+huc12
            comid=NHDplus.loc[comid_idx,'COMID']
            if huc12 in huc12dict:
                huc12dict[huc12].append(comid)
            else: 
                huc12dict[huc12]=[comid]
        self.savepickle(huc12dict,self.huc12comiddict_path)
        self.savepickle(NHDplus,savefilename)
        self.huc12comiddict=huc12dict
        if setNHDplus_attribute:
            self.NHDplus=NHDplus
        return 
    
    
    
    
    
    def getstreamcat(self,comidlist,process=1,local=1):
        #url = "https://ofmpub.epa.gov/waters10/streamcat.jsonv25?pcomid={}&pLandscapeMetricType=Topography"
        url = "https://ofmpub.epa.gov/waters10/streamcat.jsonv25?pcomid={}"
        #url="https://ofmpub.epa.gov/waters10/Watershed_Characterization.Control?pComID={}"
        if type(comidlist) is str:
            comidlist=[comidlist]
        comidcount=len(comidlist)
        comidlist_datadict={}
        if local:
            comidlist_datadict=self.pullStreamCatForComidList(comidlist)
        else:
            for idx,comid in enumerate(comidlist):
                self.logger.info(f'starting {idx}/{comidcount}')
                result=requests.get(url.format(str(comid)))
                self.logger.info(f'retrieved {idx}/{comidcount}')
                success=0
                self.logger.info(f'type(result):{type(result)}')
                try:
                    data=result.text
                    try:
                        comiddatadict=json.loads(data)
                    except:
                        self.logger.exception(f'failed to json.loads. comid:{comid}')
                        datadict={'fail':data}
                except:
                    self.logger.exception(f'failed to result.text. comid:{comid}')
                    self.logger.debug(f'{data}')
                    comiddatadict=result
                comidlist_datadict[str(comid)]=comiddatadict
            #infodict={'streamcatdata':comidlist_datadict}
        if process:
            streamcatdict=self.processStreamCat(comidlist_datadict,local=local)
            return streamcatdict
        else: 
            return comidlist_datadict
        
    
    def getNHDplus(self,huc12):
        try: self.NHDplus
        except: self.buildNHDplus(setNHDplus_attribute=1)
        huc12dataframerows=self.NHDplus.loc[self.NHDplus['HUC12']==huc12]
        self.logger.info(f'type(huc12dataframerows):{type(huc12dataframerows)}')
        jsonfile=json.loads(huc12dataframerows.to_json())
        infodict={'NHDplusdata':jsonfile}
        return infodict
        
      
                            
    
        
        
if __name__=="__main__":
    gt=GeogTool()
    gt.buildNHDplus()
    huc12list=[key for key in gt.huc12comiddict]
    gt.logger.info(f'huc12list[0:10]:{huc12list[0:10]}')
    gt.logger.info(f'gt.huc12comiddict[huc12list[0]]:{gt.huc12comiddict[huc12list[0]]}')
    gt.logger.info(f'gt.huc12comiddict[huc12list[-1]]:{gt.huc12comiddict[huc12list[-1]]}')
    gt.logger.info(f'gt.huc12comiddict[huc12list[-1]]:{gt.huc12comiddict["030701010307"]}')
    
    
                
        
            
            
            
            
