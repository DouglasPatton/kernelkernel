import traceback
import geopandas as gpd
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
from mylogger import myLogger
from sqlitedict import SqliteDict
import zlib, pickle, sqlite3

class GeogTool(myLogger):
    def __init__(self,sc_data_dir=None):
        myLogger.__init__(self,name='GeogTool.log')
        self.logger.info('starting GeogTool logger')
        self.cwd=os.getcwd()
        try: self.geogdatadir
        except:
            self.geogdatadir=os.path.join(os.getcwd(),'NHDplus_data')
        self.NHDplus_path=os.path.join(self.geogdatadir,'NHDplus')
        self.huc12comiddict_path=os.path.join(self.geogdatadir,'huc12comiddict')
        self.NHDdbf_path=os.path.join(self.geogdatadir,'HUC12_PU_COMIDs_CONUS.dbf')
        self.NHDhuchuc_path=os.path.join(self.geogdatadir,'NHDhuchuc')
        self.failed_SC_comid_path=os.path.join('data_tool','failedSCcomidlist.dbf')
        if sc_data_dir is None:
            try: self.sc_data_dir
            except: 
                localpath=os.path.join(self.cwd,'../../../hdd3','sc_data')
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
        self.reverse_huc12comid() # this will build most/all of the NHDplus related files if they don't exist
    
    def my_encode(self,obj):
        return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL),level=9))
    def my_decode(self,obj):
        return pickle.loads(zlib.decompress(bytes(obj)))
    #mydict = SqliteDict('./my_db.sqlite', encode=self.my_encode, decode=self.my_decode)
    
    def anyNameDB(self,dbname,tablename='data',folder=None):
        name=dbname+'.sqlite'
        if folder:
            path=os.path.join(folder,name)
        else:
            path=name
        return SqliteDict(
            filename=path,tablename=tablename,)
            #encode=self.my_encode,decode=self.my_decode)
    
    def addtoDB(self,dict_to_save,path):
        with self.anyNameDB(path) as db:
            for key,val in dict_to_save.items():
                db[key]=val
            db.commit()

      
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
    
    def getstreamcat(self,comidlist,process=1,local=1,add_huc12=1):
        #url = "https://ofmpub.epa.gov/waters10/streamcat.jsonv25?pcomid={}&pLandscapeMetricType=Topography"
        url = "https://ofmpub.epa.gov/waters10/streamcat.jsonv25?pcomid={}"
        #url="https://ofmpub.epa.gov/waters10/Watershed_Characterization.Control?pComID={}"
        if type(comidlist) is str:
            comidlist=[comidlist]
        comidcount=len(comidlist)
        comidlist_datadict={}
        
        if local:
            comidlist_datadict=self.pullStreamCatForComidList(comidlist,add_huc12=add_huc12)
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
            streamcatdict=self.processStreamCat(comidlist_datadict)
            return streamcatdict
        else: 
            return comidlist_datadict
        
        
    def processStreamCat(self,streamcatdata,collapse=1):
        try:
            if type(streamcatdata) is str:
                self.logger.debug(f'streamcatdata type is a str: {streamcatdata}')
                streamcatdata=self.anyNameDB(streamcatdata)
            outdict={}
            error_dict={}
            for comid,data in streamcatdata.items():
                collapse_dict={}
                error_dict[comid]={}
                metricdict={'Ws':{},'Cat':{},'CatRp100':{},'WsRp100':{},'Rp100':{},'other':{}}
                metadict=None
                mkeylist=[*metricdict] #the keys or mkeys
                #print('mkeylist',mkeylist)
                keylengths=[len(key) for key in mkeylist]
                keycount=len(mkeylist)
                regex_y2k=re.compile('20[0-9][0-9]')
                #nlcd_regex_y2k=re.compile('nlcd20[0-9][0-9]')
                #metrics=[*data]
                
                for metric,data_pt in data.items():
                    
                    srch_y=re.search(regex_y2k,metric)
                    if srch_y:
                        yr=metric[srch_y.start():srch_y.end()]
                        metric_drop_yr=metric[:srch_y.start()]+metric[srch_y.end():]+'_avg'
                    else:
                        metric_drop_yr=metric
                        yr='all'
                    if collapse:
                        try:
                            try:
                                float_data_pt=float(data_pt)
                            except:
                                if len(data_pt)==0 or data_pt=='NA':
                                    float_data_pt=np.nan
                                else: 
                                    float(data_pt)#force error
                            if not metric_drop_yr in collapse_dict:
                                collapse_dict[metric_drop_yr]=[float_data_pt]
                            else:
                                collapse_dict[metric_drop_yr].append(float_data_pt)
                        except:
                            if not metric in error_dict[comid]:
                                error_dict[comid][metric]=[data_pt]
                            else:
                                error_dict[comid][metric].append(data_pt)
                    else:
                        for k in range(keycount-1):
                            endstring=metric[-keylengths[k]:]
                            #self.logger.critical(f'endstring:{endstring} for metric:{metric}')
                            if endstring==mkeylist[k]:
                                mkey=mkeylist[k]
                                break
                            if k==keycount-2:
                                mkey='other'
                        if not yr in metricdict[mkey]:
                            metricdict[mkey][yr]={}
                        metricdict[mkey][yr][metric]=data_pt
                    
                if collapse:
                    #many of these are just divided by 1
                    outdict[comid]={metric:sum(val)/len(val) for metric,val in collapse_dict.items()}
                    
                else:
                    outdict[comid]=metricdict
            self.logger.exception(f'float error_dict:{error_dict}')
            return outdict
        except:
            self.logger.exception('streamcat error') 
    
    def filterfailedcomids(self,comidlist):
        try:previously_failed_comids=self.anyNameDB(self.failed_SC_comid_path)['failed']
        except KeyError:
            self.logger.info(f'"failed" key not found in failed SC db')
            return comidlist
        except:
            assert False, 'unexpected error'
        comidlist=[comid for comid in comidlist if not comid in previously_failed_comids]
        return comidlist
                
    def addfailed(self,comidlist):
        with self.anyNameDB(self.failed_SC_comid_path) as db:
            try:previously_failed_comids=db['failed']
            except KeyError: previously_failed_comids={}
            except:assert False,'unexpected error'
            comiddict=dict.fromkeys([*previously_failed_comids.keys(),*comidlist])
            db['failed']=comiddict#just a dict for fast searching
            db.commit()
        return
            
        
        
    def pullStreamCatForComidList(self,comidlist,meta=0,error_lookup=0,add_huc12=1):
        try:self.rvrs_huc12comiddict
        except: self.reverse_huc12comid()
        comidlist=self.filterfailedcomids(comidlist)
        if len(comidlist)==0:
            self.logger.info(f'geogtool has only previously failed comids. returning empty dict')
            return {}
        #try: self.huchuc
        #except:self.build_huchuc()
        pathlist=[path for path in os.listdir(self.sc_data_dir) if path[-5:]=='.json']
        SCoutdict={}
        assert type(comidlist) is list, f'expecting list got {type(comidlist)}'
        comiddict={}
        huc12list=[]
        """if type(comidlist[0]) is str:
            self.logger.debug(f'converting comids from string to int')
            comidlist=[int(comid) for comid in comidlist]"""
        failed_comids=[]
        new_comidlist=[]
        huc8dict={}
        for comid in comidlist:
            try:
                huc8=self.rvrs_huc12comiddict[comid][:8]
                if not huc8 in huc8dict:
                    huc8dict[huc8]=[comid]
                else:
                    huc8dict[huc8].append(comid)
            except:
                failed_comids.append(comid)
                self.logger.warning(f'no huc12 for comid:{comid}')
        
        huc8_errordict={}    
        found=0
        fail=0
        for huc8,comids in huc8dict.items():
            huc8_errordict[huc8]=[]
            path=self.pathfromhuc8(huc8) # no group by for faster load
            if meta: path=path[:-5]+'_meta.json'
            try:
                huc8scdata=self.openjson(path)
                for comid in comids:
                    #self.logger.info(f'comid:{comid}')
                    comiddata=None
                    try:
                        comiddata=huc8scdata[int(comid)] # streamcat is not yet a string comid in the data
                    except KeyError:
                        try:

                            comiddata=huc8scdata[comid]
                            
                        except KeyError:

                            self.logger.info(f'could not find streamcat as str or int for huc8:{huc8},comid:{comid}')#,huc8scdata:{huc8scdata}')
                            if error_lookup:
                                
                                message=self.checkNHDPlus(comid)
                                self.logger.critical(message)    
                        except:
                            assert False, 'halt, unexpected error'
                    except:
                        assert False, 'halt, unexpected error'
                    if not comiddata is None:
                        found+=1
                        self.logger.info(f'streamcat data found for comid:{comid}')
                        if add_huc12:
                                comiddata['HUC12']=self.rvrs_huc12comiddict[comid]
                        comiddict[comid]=comiddata # comid as a string
                    else:
                        failed_comids.append(comid)
                        huc8_errordict[huc8].append(comid)
                        fail+=1
            except: 
                self.logger.exception(f'Streamcat problem for huc8:{huc8}')
        self.addfailed(failed_comids)
        self.logger.info(f'huc8_errordict:{huc8_errordict}')
        self.logger.info(f'counts for found:{found} and fail:{fail}')
        return comiddict
 

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
            self.huc12comiddict=self.anyNameDB(huc12comiddict_path)
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
                huchuc=self.anyNameDB(self.NHDhuchuc_path)
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
            self.huchuc=self.anyNameDB(self.NHDhuchuc_path)
            if len(self.huchuc)==2:
                return self.huchuc
            else: self.logger.info(f'rebuilding huchuc')
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
        self.addtoDB(huchuc,self.NHDhuchuc_path)
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
                NHDplus=self.anyNameDB(savefilename)['data']
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
                self.huc12comiddict=self.anyNameDB(self.huc12comiddict_path)
                assert len(self.huc12comiddict)>0,f'len huc12comiddict:{len(self.huc12comiddict)}'
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
            comid=NHDplus.loc[comid_idx,'COMID'].astype(str)
            if huc12 in huc12dict:
                huc12dict[huc12].append(comid)
            else: 
                huc12dict[huc12]=[comid]
        self.addtoDB(huc12dict,self.huc12comiddict_path)
        self.addtoDB({'data':NHDplus},savefilename)
        self.huc12comiddict=huc12dict
        if setNHDplus_attribute:
            self.NHDplus=NHDplus
        return 
    
    
    
    
    

    
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
    
    
                
        
            
            
            
            
