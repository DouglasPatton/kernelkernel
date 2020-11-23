import os,joblib
import csv
import numpy as np
import pickle
from time import sleep,strftime,time
import multiprocessing as mp
#mport geopandas as gpd
import logging
import pandas as pd
from geogtools import GeogTool as gt
from mylogger import myLogger
from pi_data_helper import MpBuildStreamcatFromComids,MpBuildSpeciesData01,Helper
import re
from pi_db_tool import DBTool


class PiscesDataTool(myLogger,DBTool,Helper):
    def __init__(self,):
        myLogger.__init__(self,name='pisces_data_huc12.log')
        self.logger.info('starting pisces_data_huc12 logger')
        DBTool.__init__(self)
        Helper.__init__(self)
        self.savedir=os.path.join(os.getcwd(),'data_tool')
        if not os.path.exists(self.savedir): os.mkdir(self.savedir)
        self.processcount=15
        self.pi_db=self.pidataDBdict()
        #slf.gt=gt() # called later
                
    
    def retrievespeciesdata(self,species_idx=None,species_name=None):
        try: 
            self.specieslist
            #print(type(self.specieslist))
        except: self.buildspecieslist()
        try:
            if species_name==None and type(species_idx) is int:
                species_name=self.specieslist[species_idx]
            
            if not species_name is None:
                return self.pi_db[species_name]
            else:
                assert False,'expecting a species name or species idx'
        except: self.logger.exception(f'retrievespeciesdata outer catch')
        
    
    def viewNHDplus_picklefile(self,):
        savefilename=os.path.join(self.savedir,'NHDplus.data')
        if os.path.exists(savefilename):
            try: 
                with open(savefilename, 'rb') as f:
                    self.NHDplus=pickle.load(f)
                print(f'opening {savefilename} with length:{len(self.NHDplus)} and type:{type(self.NHDplus)}')
                #print(self.NHDplus)
                return
            except:
                self.logger.exception('viewNHDplus_picklefile could not open saved NHDplus.data')
                
    
    def getNHDplus(self,):
        
        savefilename=os.path.join(self.savedir,'NHDplus.data')
        if os.path.exists(savefilename):
            try: 
                with open(savefilename, 'rb') as f:
                    (self.NHDplus,self.NHDpluscomidlist,self.NHDvarlist)=pickle.load(f)
                return
            except: 
                print(f"{savefilename} exists but could not open, rebuilding")
        
        filename=os.path.join(os.getcwd(),'NHDplus_data','HUC12_PU_COMIDs_CONUS.dbf')
        print(f'starting read of {filename}')
        dbf=gpd.read_file(filename)
        print('finished read of NHDplus')
        print(f'opened {filename} with length:{len(dbf)} and type:{type(dbf)}')
        self.NHDvarlist=['COMID','HUC12','REACHCODE','TOHUC','Length']
        #print('self.NHDvarlist: ',self.NHDvarlist)
        self.NHDplus=dbf.loc[:,self.NHDvarlist] 
        strvarlist=self.NHDvarlist[:-1]
        for strvar in strvarlist:
            self.NHDplus.loc[:,(strvar)]=self.NHDplus.loc[:,(strvar)].to_numpy().astype('str')
        self.NHDpluscomidlist=list(self.NHDplus.loc[:,('COMID')].to_numpy()) # string comid's

        #print(self.NHDplus)
        
        try:
            with open(savefilename,'wb') as f:
                pickle.dump((self.NHDplus,self.NHDpluscomidlist,self.NHDvarlist),f)
        except:
            self.logger.exception('problem saving NHDplus:')
        
        
        
    def getcsvfile(self,filename):
        thisdir=os.getcwd()
        datadir=os.path.join(thisdir,'fishfiles',filename)
        #if os.path.exists(datadir):
        
        with open(datadir, 'r') as f:
            datadict=[row for row in csv.DictReader(f)]
        print(f'opening {filename} with length:{len(datadict)} and type:{type(datadict)}')
        
            
        
        keylist=[key for row in datadict for key,val in row.items()]
        return datadict

    def getfishdata(self,):
        self.fishsurveydata=self.getcsvfile('surveydata.csv')
        print(self.fishsurveydata[0:5])

    def gethucdata(self,):
        self.getNHDplus()
        self.viewNHDplus_picklefile()

    def getsitedata(self,):
        self.sitedata=self.getcsvfile('siteinfo.csv')
        self.sitedata_comid_digits=[''.join([char for char in datarow['COMID'] if char.isdigit()]) for datarow in self.sitedata]
        print(self.sitedata[0:5])

    
    def getfishhucs(self,):
        self.fishhucs=self.getcsvfile('fishhucs.csv')
        print(self.fishhucs[0:5])
    
    def returnspecieslist(self,):
        try:
            specieslist=self.specieslist
            return specieslist
        except:
            pass
        
        specieslistpath=os.path.join(self.savedir,'specieslistfiles')
        if os.path.exists(specieslistpath):
            
            try:
                with open(specieslistpath,'rb') as f:
                    speciestup=pickle.load(f)
                self.specieslist=speciestup[0]
                return self.specieslist
            except:
                pass
        self.buildspecieslist()
        return self.specieslist
        
        
        
    def buildspecieslist(self,):
        specieslistpath=os.path.join(self.savedir,'specieslistfiles')
        if os.path.exists(specieslistpath):
            try:
                with open(specieslistpath,'rb') as f:
                    speciestup=pickle.load(f)
                self.specieslist=speciestup[0] # variable descriptions below
                self.speciesoccurencelist=speciestup[1] 
                self.speciescomidlist=speciestup[2] 
                self.specieshuclist=speciestup[3]
                self.huclist_survey=speciestup[4]
                self.huccomidlist_survey=speciestup[5]
                self.specieshuclist_survey_idx=speciestup[6]
                
                #print(f'opening {specieslistpath} with length:{len(speciestup)} and type:{type(speciestup)}')
                return
            except:
                self.logger.exception(f'buildspecieslist found {specieslistpath} but there was an error loading variables')
                
        try:self.fishsurveydata
        except: self.getfishdata()
        (longlist,huclist,comidlist)=zip(*[(obs['genus_species'],obs['HUC'],obs['COMID']) for obs in self.fishsurveydata])
        comidlist=[''.join([char for char in comidi if char.isdigit()]) for comidi in comidlist]
        shortlist=[];occurencelist=[];speciescomidlist=[];specieshuclist=[]
        shorthuclist=[];huccomidlist=[];specieshuclist=[];specieshuclist_survey_idx=[]
        print('building specieslist')
        length=len(longlist)
        for idx,fish in enumerate(longlist):
            if idx%(length//10)==0:
                print(str(round(100*idx/length))+'%',sep=',',end=',')
            found=0
            try:
                shortidx=shortlist.index(fish)
                found=1
                
            except ValueError:
                shortidx=len(shortlist)#-1 not needed since not yet appended to end of shortlist
                shortlist.append(fish)
                specieshuclist.append([])
                specieshuclist_survey_idx.append([])
                occurencelist.append([idx])
                speciescomidlist.append([comidlist[idx]])
                specieshuclist.append([huclist[idx]])
                #p#rint(f'new fish:{fish}')
            if found==1:
                occurencelist[shortidx].append(idx)
                speciescomidlist[shortidx].append(comidlist[idx])
                specieshuclist[shortidx].append(huclist[idx])
                
            hucfound=0
            huc=huclist[idx]; 
            if not type(huc) is str:
                huc=str(huc)
            if len(huc)==7:
                #p#rint(f'expecting 8 characters: huc8_i:{huc}, prepending a zero')
                huc='0'+huc
            assert len(huc)==8,print(f'expecting 8 characters: huc8_i:{huc}')
            found=0
            try:
                hucshortidx=shorthuclist.index(huc)
                found=1    
            except:
                hucshortidx=len(shorthuclist)#this will be its address after appending
                shorthuclist.append(huc)
                huccomidlist.append([comidlist[idx]])
                specieshuclist[shortidx].append(huc)
                specieshuclist_survey_idx[shortidx].append(hucshortidx)
            if found==1:
                
                comid_i=comidlist[idx]
                try: huccomidlist[hucshortidx].index(comid_i)
                except ValueError: huccomidlist[hucshortidx].append(comid_i)
                try: specieshuclist[shortidx].index(huc)
                except ValueError: specieshuclist[shortidx].append(huc)
                try: specieshuclist_survey_idx[shortidx].index(hucshortidx)
                except ValueError: specieshuclist_survey_idx[shortidx].append(hucshortidx)
                
        print(f'buildspecieslist found {len(shortlist)} unique strings')
        
        with open(specieslistpath,'wb') as f:
            pickle.dump((shortlist,occurencelist,speciescomidlist,specieshuclist,shorthuclist,huccomidlist,specieshuclist_survey_idx),f)
        
        self.specieslist=shortlist#a list of unique species
        #next lists have an item for each species in self.specieslist
        self.speciesoccurencelist=occurencelist#for each species, a list of indices from the original long list of species
        self.speciescomidlist=speciescomidlist#for each species, a list of comids where the species was observed
        self.specieshuclist=specieshuclist#for each species, a list of huc8s where the species was observed 
        
        
        #next 2 lists are not for each species
        self.huclist_survey=shorthuclist#self.huclist_survey is a list of unique huc8's
        self.huccomidlist_survey=huccomidlist#or each huc in self.huclist_survey, a list of comids in that huc that were found in the survey
        
        self.specieshuclist_survey_idx=specieshuclist_survey_idx#for each species, a list of idx for self.huclist_survey indicating hucs that species appeared in

        
    def buildspecieshuc8list(self,):
        
        huc8listpath=os.path.join(self.savedir,'specieshuc8files')
        if os.path.exists(huc8listpath):
            try:
                with open(huc8listpath,'rb') as f:
                    huc8tup=pickle.load(f)
                (self.specieshuc8list,self.otherhuclist,self.specieshuclist_newhucs,self.specieshuclist_survey_idx_newhucs)=huc8tup
                
                
                print(f'opening {huc8listpath} with length:{len(huc8tup)} and type:{type(huc8tup)}')
                return
            except:
                self.logger.exception(f'buildhuc8list found {huc8listpath} but there was an error loading variables')
        
        self.otherhuclist=[]
        
        try: self.fishhucs
        except: self.getfishhucs()
        try: self.specieslist,self.huclist_survey
        except: self.buildspecieslist()
        self.specieshuclist_newhucs=[[] for _ in range(len(self.specieslist))]
        self.specieshuclist_survey_idx_newhucs=[[] for _ in range(len(self.specieslist))]
        specieshuc8list=[[] for _ in range(len(self.specieslist))]
        lastspec_i=''
        for i,item in enumerate(self.fishhucs):
            spec_i=item['Scientific_name']
            
            #d_i=item['ID']
            huc8_i=item['HUC']
            if not type(huc8_i) is str:
                huc8_i=str(huc8_i)
            if len(huc8_i)==7:
                huc8_i='0'+huc8_i
            assert len(huc8_i)==8,print(f'expecting 8 characters: huc8_i:{huc8_i}')
            if not spec_i==lastspec_i:
                specfound=0
                try: 
                    specieslist_idx=self.specieslist.index(spec_i)
                    specfound=1                
                except ValueError: 
                    try: self.otherspecieslist.append(spec_i)
                    except AttributeError: self.otherspecieslist=[spec_i]
            if specfound==1:
                specieshuc8list[specieslist_idx].append(huc8_i)
                try:
                    huclist_survey_idx=self.huclist_survey.index(huc8_i)
                    try: self.specieshuclist[specieslist_idx].index(huc8_i)
                    except ValueError: 
                        self.logger.info(f'{spec_i} has new huc:{huc8_i}')
                        self.specieshuclist_newhucs[specieslist_idx].append(huc8_i)
                        self.specieshuclist_survey_idx_newhucs[specieslist_idx].append(huclist_survey_idx)
                except ValueError: 
                    self.otherhuclist.append(huc8_i)
                    self.logger.info(f'{spec_i} has new huc:{huc8_i}, but it does not show up in survey, so not relevant')
            lastspec_i=spec_i    
        self.specieshuc8list=specieshuc8list
        with open(huc8listpath,'wb') as f:
            pickle.dump((self.specieshuc8list,self.otherhuclist,self.specieshuclist_newhucs,self.specieshuclist_survey_idx_newhucs),f)

            
    
    def buildCOMIDlist(self,):
        comidlistpath=os.path.join(self.savedir,'comidlist')
        if os.path.exists(comidlistpath):
            try:
                with open(comidlistpath,'rb') as f:
                    comidtup=pickle.load(f)
                self.comidlist=comidtup[0]
                self.comidcurrurencelist=comidtup[1]
                print(f'opening {comidlistpath} with length:{len(comidtup)} and has first item length: {len(self.comidlist)} and type:{type(self.comidlist)}')
                return
            except:
                self.logger.exception(f'error when opening {comidlistpath}')
            
        try: self.fishsurveydata
        except: self.getfishdata()
        print('building comidlist')
        longlist=[obs['COMID'] for obs in self.fishsurveydata]
        length=len(longlist)
        i=0
        shortlist=list(set([''.join([char for char in comid if char.isdigit()]) for comid in longlist]))
        self.logger.warning(f'shortlist:{shortlist}')
        occurencelist=[[] for _ in range(len(shortlist))]
        self.logger.info(f'buildCOMIDlist found {len(shortlist)}')
        with open(comidlistpath,'wb') as f:
            pickle.dump((shortlist,occurencelist),f)
        self.comidlist=shortlist # no repeats made of comid_digits, a string
    

    def comidsiteinfo_db(self): #to make the db callable
        name='sitedatacomid_dict'
        folder=os.path.expanduser('~/gits/kernelkernel/data_tool')
        path=os.path.join(folder,name+'.sqlite')
        if not os.path.exists(path):
            folder='data_tool'
            print(f'using local data_tool bc {path}')
        return self.anyNameDB(name,folder=folder)

    def buildCOMIDsiteinfo(self,comidlist=None,predict=False,rebuild=False):
        comidsiteinfo_callable_db=self.comidsiteinfo_db
        
        if not rebuild:
            if not predict:
                self.sitedatacomid_dict=comidsiteinfo_callable_db#{}
                return
            else:
                return comidsiteinfo_callable_db
            
        if comidlist is None:
            try:self.comidlist
            except:self.buildCOMIDlist()
            comidlist=self.comidlist
        
                #pool.close()
        self.logger.info(f'getting built comid list')    
        with comidsiteinfo_callable_db() as db:
            built_comids=dict.fromkeys(db.keys()) #make a dict for search performance
        self.logger.info(f'there are {len(built_comids)} built comids, creating build list')
        build_comidlist=[]
        for comid in comidlist:
            if not comid in built_comids:
                build_comidlist.append(comid)
        gtool=gt()
        build_comidlist=gtool.filterfailedcomids(build_comidlist)
        comidcount=len(build_comidlist)
        self.logger.info(f'about to build comiddata for {comidcount} comids')
        if comidcount<self.processcount:
            processcount=1
        else:
            processcount=self.processcount
        if comidcount>0:
            com_idx=np.array_split(np.arange(comidcount,dtype=np.int64),processcount)
            
            args_list=[[[build_comidlist[c] for c in com_idx[i]],gtool] for i in range(processcount)]
            outlist=self.runAsMultiProc(MpBuildStreamcatFromComids,args_list,add_to_db=comidsiteinfo_callable_db)
            #sitedatacomid_dict=self.mergelistofdicts(outlist)
            self.logger.info(f'expecting {processcount} all "complete" outlist:{outlist}')
            #sitedatacomid_dict=gt().getstreamcat(comidlist)

        #self.addToDBDict(sitedatacomid_dict,db=db)
        if not predict:
            self.sitedatacomid_dict=comidsiteinfo_callable_db#{}
            return
        else:
            return comidsiteinfo_callable_db


    
    def buildspeciesdata01_file(self,):
        thisdir=self.savedir
        datadir=os.path.join(thisdir,'speciesdata01')
        if not os.path.exists(datadir):
            os.mkdir(datadir)
        try:self.specieslist
        except:self.buildspecieslist()
        try:self.comidlist
        except:self.buildCOMIDlist()
        try: self.sitedata
        except:self.getsitedata()
        try:self.sitedatacomid_dict
        except: self.buildCOMIDsiteinfo()
        speciescount=len(self.specieslist)
        
        split_idx=[int(i) for i in np.linspace(0,speciescount,self.processcount+1)]#+1 to include the end
        speciesidx_list=[i for i in range(speciescount)]
        speciesidx_listlist=[]
        for i in range(self.processcount):
            speciesidx_listlist.append(speciesidx_list[split_idx[i]:split_idx[i+1]])
        args_list=[
            [speciesidx_listlist[i],self.savedir,self.specieslist,self.sitedatacomid_dict,
            self.specieshuclist_survey_idx,self.specieshuclist_survey_idx_newhucs,self.huccomidlist_survey,self.speciescomidlist] 
            for i in range(self.processcount)]
        outlist=self.runAsMultiProc(MpBuildSpeciesData01,args_list)
        self.outlist2=outlist # just for debugging
        print(f'the length of each list in outlist:{[len(item) for item in outlist]}.')
        if self.processcount>1:
            fail_record=zip(*outlist)
            fail_record=[record for recordlist in fail_record for record in recordlist]
        else:
            fail_record=outlist
        self.buildspeciesdata01_file_fail_record=outlist
        
        
    
        
    
if __name__=='__main__':
    test=PiscesDataTool()
    test.getfishdata()
    test.getsitedata()
    test.getfishhucs()
    test.gethucdata()
    test.getNHDplus()
    test.buildspecieslist()
    test.buildspecieshuc8list()
    test.buildCOMIDlist()
    test.buildCOMIDsiteinfo()
    test.buildspeciesdata01_file()
    
