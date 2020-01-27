import os
import csv
#import traceback
import numpy as np
import pickle
from time import sleep,strftime,time
import multiprocessing as mp
import geopandas as gpd
import logging
import traceback
#import pandas as pd


class PiscesDataTool():
    def __init__(self,):
        #logging.basicConfig(level=logging.INFO)
        logging.basicConfig(level=logging.INFO)
        self.savedir=os.path.join(os.getcwd(),'data_tool')
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        self.processcount=6
        logdir=os.path.join(self.savedir,'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername='Datatool.log'
        handler=logging.FileHandler(os.path.join(logdir,handlername))
        self.logger1 = logging.getLogger(__name__)
        self.logger1.addHandler(handler)

    
        
        
    def retrievespeciesdata(self,species_idx=None,species_name=None):
        try: 
            self.specieslist
            print(type(self.specieslist))
        except: self.buildspecieslist()
        if species_name==None and type(species_idx) is int:
            species_name=self.specieslist[species_idx]
            
        datadir=os.path.join(self.savedir,'speciesdata01')
        if not species_name is None:
            species_filename=os.path.join(datadir,species_name+'.data')  
            with open(species_filename, 'rb') as f:
                species_data=pickle.load(f)
            return species_data
        else:
            try:
                with open(os.path.join(datadir,'sitedatakeylist'),'rb')as f:
                    sitevarlist=pickle.load(f)
                return sitevarlist   
            except: 
                print(traceback.format_exc())
                return 'sitevarlist not found'
        
        
            
        
    
    
    
    def viewNHDplus_picklefile(self,):
        savefilename=os.path.join(self.savedir,'NHDplus.data')
        if os.path.exists(savefilename):
            try: 
                with open(savefilename, 'rb') as f:
                    self.NHDplus=pickle.load(f)
                print(f'opening {savefilename} with length:{len(self.NHDplus)} and type:{type(self.NHDplus)}')
                print(self.NHDplus)
                return
            except:
                self.logger1.exception('viewNHDplus_picklefile could not open saved NHDplus.data')
                
    
    def getNHDplus(self,):
        
        savefilename=os.path.join(self.savedir,'NHDplus.data')
        if os.path.exists(savefilename):
            try: 
                with open(savefilename, 'rb') as f:
                    (self.NHDplus,self.NHDpluscomidlist,self.NHDvarlist)=pickle.load(f)
                print(f'opening {savefilename} with length:{len(self.NHDplus)} and type:{type(self.NHDplus)}')
                print(self.NHDplus)

                self.NHDpluscomidlist=list(self.NHDplus.loc[:,('COMID')].to_numpy())

                return
            except: 
                print(f"{savefilename} exists but could not open, rebuilding")
        
        filename=os.path.join(os.getcwd(),'fishfiles','huc12_pu_comids_conus.dbf')
        
        #dbf = gpd.GeoDataFrame.from_file(filename)
        print(f'starting read of {filename}')
        dbf=gpd.read_file(filename)
        print('finished read of NHDplus')
        print(f'opened {filename} with length:{len(dbf)} and type:{type(dbf)}')
        #p#rint(dbf.head())
        #self.NHDvarlist=dbf.columns.values
        self.NHDvarlist=['COMID','HUC12','REACHCODE','TOHUC','Length']
        print('self.NHDvarlist: ',self.NHDvarlist)
        
        #self.dbf=dbf
        #self.NHDplus=[dbf[var].tolist() for var in varlist]
        
        self.NHDplus=dbf[self.NHDvarlist]
        strvarlist=self.NHDvarlist[:-1]
        for strvar in strvarlist:
            self.NHDplus.loc[:,(strvar)]=self.NHDplus.loc[:,(strvar)].to_numpy().astype('str')
        self.NHDpluscomidlist=list(self.NHDplus.loc[:,('COMID')].to_numpy())

        print(self.NHDplus)
        
        try:
            with open(savefilename,'wb') as f:
                pickle.dump((self.NHDplus,self.NHDpluscomidlist,self.NHDvarlist),f)
        except:
            self.logger1.exception('problem saving NHDplus:')
        
        
        
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
        self.sitedata_k=len(self.sitedata[0])
        self.sitevarkeylist=[key for key,_ in self.sitedata[0].items()]
        self.sitedata_comid_digits=[''.join([char for char in datarow['COMID'] if char.isdigit()]) for datarow in self.sitedata]
        print(self.sitedata[0:5])
        
    def getfishhucs(self,):
        self.fishhucs=self.getcsvfile('fishhucs.csv')
        print(self.fishhucs[0:5])

        
    def buildspecieslist(self,):
        specieslistpath=os.path.join(self.savedir,'specieslistfiles')
        if os.path.exists(specieslistpath):
            try:
                with open(specieslistpath,'rb') as f:
                    speciestup=pickle.load(f)
                self.specieslist=speciestup[0]
                self.speciesoccurencelist=speciestup[1]
                self.speciescomidlist=speciestup[2] 
                self.specieshuclist=speciestup[3]
                self.huclist_survey=speciestup[4]
                self.huccomidlist_survey=speciestup[5]
                self.specieshuclist_survey_idx=speciestup[6]
                
                print(f'opening {specieslistpath} with length:{len(speciestup)} and type:{type(speciestup)}')
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
                #print(spec_i)
                specfound=0
                #for j,spec_j in enumerate(self.specieslist):
                #    if spec_i==spec_j:
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
                        self.logger1.info(f'{spec_i} has new huc:{huc8_i}')
                        self.specieshuclist_newhucs[specieslist_idx].append(huc8_i)
                        self.specieshuclist_survey_idx_newhucs[specieslist_idx].append(huclist_survey_idx)

                except ValueError: 
                    self.otherhuclist.append(huc8_i)
                    self.logger1.info(f'{spec_i} has new huc:{huc8_i}, but it does not show up in survey, so not relevant')
 
                
            
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
                self.logger1.exception(f'error when opening {comidlistpath}')
            
        try: self.fishsurveydata
        except: self.getfishdata()
        print('building comidlist')
        longlist=[obs['COMID'] for obs in self.fishsurveydata]
        
        shortlist=[];occurencelist=[]
        length=len(longlist)
        i=0
        for idx,comid in enumerate(longlist):
            if idx%(length//33)==0:
                print(str(round(100*idx/length))+'%',sep=',')
            found=0
            try:
                shortidx=shortlist.index(comid_i)
                found=1
                #p#rint(f'old comid:{comid}')
            except:
                comid_digits=''.join([char for char in comid if char.isdigit()])
                shortlist.append(comid_digits)
                occurencelist.append([idx])
            if found==1:
                occurencelist[shortidx].append(idx)
        self.logger1.info(f'buildCOMIDlist found {len(shortlist)}')
        with open(comidlistpath,'wb') as f:
            pickle.dump((shortlist,occurencelist),f)
        
        self.comidlist=shortlist  
        self.comidoccurenclist=occurencelist
    
    


        
    
    
    def mergelistofdicts(self,listofdicts):
        mergedict={}
        for i,dict_i in enumerate(listofdicts):
            for key,val in dict_i.items():
                
                if not key in mergedict:
                    mergedict[key]=val
                else:
                    newkey=key+f'_{i}'
                    #p#rint(f'for dict_{i} oldkey:{key},newkey:{newkey}')
                    mergedict[newkey]=val
        return mergedict

####################################################################
####################################################################
##################the next 3 methods run together###################
####################################################################
####################################################################
        
    def buildCOMIDsiteinfo(self,):
        try:self.comidlist
        except:self.buildCOMIDlist()
        filepath=os.path.join(self.savedir,'sitedatacomid_dict')
        if os.path.exists(filepath):
            try:
                with open(filepath,'rb') as f:
                    savefile=pickle.load(f)
                self.logger1.info(f'buildCOMIDsiteinfo opened {filepath}, type: {type(savefile)}, length:{len(savefile)}')
                self.logger1.info(f'first item has type: {type(savefile[0])}, length:{len(savefile[0])}')
                
                self.sitedatacomid_dict=savefile[0]
                self.comidsitedataidx=savefile[1]
                self.comidsiteinfofindfaillist=savefile[2]
                self.huc12findfaillist=savefile[3]
                print(f'opening {filepath} with length:{len(savefile)} and has first item length: {len(self.sitedatacomid_dict)} and type:{type(self.sitedatacomid_dict)}')
                self.sitedatakeylist=[key for key,_ in self.sitedatacomid_dict[self.comidlist[0]].items()]
                return
            except:
                self.logger1.exception(f'buildCOMIDsiteinfo found {filepath} but could not load it, so rebuilding')
                
        else:
            print(f'{filepath} does not exist, building COMID site info')   
        try: self.sitedata
        except:self.getsitedata()
            
        comidcount=len(self.comidlist)

        if self.processcount>1:
            com_idx=[int(i) for i in np.linspace(0,comidcount,self.processcount+1)]#+1 to include the end
            print(com_idx)
            
            comidlistlist=[]
            for i in range(self.processcount):
                comidlistlist.append(self.comidlist[com_idx[i]:com_idx[i+1]])

            
            starttime=time()
            self.logger1.info(f'pool starting at {starttime}')
            with mp.Pool(processes=self.processcount) as pool:
                outlist=pool.map(self.mpsearchcomidhuc12,comidlistlist)
                sleep(2)
                pool.close()
                pool.join()
            self.outlist=outlist
            endtime=time()
            self.logger1.info(f'pool complete at {endtime}, time elapsed: {(endtime-starttime)/60} minutes')
            comidsitedataidx,sitedatacomid_dict,comidsiteinfofindfaillist,huc12findfaillist=zip(*outlist)
            self.comidsiteinfofindfaillist=[i for result in comidsiteinfofindfaillist for i in result]
            self.huc12findfaillist=[i for result in huc12findfaillist for i in result]

            self.sitedatacomid_dict=self.mergelistofdicts(sitedatacomid_dict)

            self.comidsitedataidx=[]
            for i in range(self.processcount):
                self.logger1.info(f'len(comidsitedataidx[i]) {len(comidsitedataidx[i])}')
                self.comidsitedataidx.extend([j+com_idx[i] for j in comidsitedataidx[i]])
        else:
            outlist=self.mpsearchcomidhuc12(self.comidlist)
            self.outlist=outlist
            self.comidsitedataidx,self.sitedatacomid_dict,self.comidsiteinfofindfaillist,self.huc12findfaillist=outlist

        
        with open(filepath,'wb') as f:
            pickle.dump((self.sitedatacomid_dict,self.comidsitedataidx,self.comidsiteinfofindfaillist,self.huc12findfaillist),f)
        self.sitedatakeylist=[key for key,_ in self.sitedatacomid_dict[self.comidlist[0]].items()]
        self.comidsiteinfofindfail=[];self.huc12findfail=[]
        if sum(self.comidsiteinfofindfaillist)>0:
            for i in range(comidcount):
                if self.comidsiteinfofindfaillist[i]==1:
                    self.logger1.warning(f'comidsiteinfofind failed for comid:{self.comidlist[i]}')
                    self.comidsiteinfofindfail.append(self.comidlist[i])
                
        if sum(self.huc12findfaillist)>0:
            for i in range(comidcount):
                if self.huc12findfaillist[i]==1:
                    self.logger1.warning(f'huc12find failed for comid:{self.comidlist[i]}')
                    self.huc12findfail.append([self.comidlist[i]])

        return
 
    def mpsearchcomidhuc12(self,comidlist):
        
        logdir=os.path.join(self.savedir,'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=f'mpsearchcomidhuc12{os.getpid()}.log'
        handler=logging.FileHandler(os.path.join(logdir,handlername))
        self.logger = logging.getLogger(__name__+str(os.getpid()))
        self.logger.addHandler(handler)
        mypid=os.getpid()
        comidcount=len(comidlist)
        comidsitedataidx=[]
        sitedatacomid_dict={}
        huc12findfaillist=[0]*comidcount
        huc12failcount=0
        comidsiteinfofindfaillist=[0]*comidcount
        
        printselection=[int(idx) for idx in np.linspace(0,comidcount,11)]
        for i,comid_i in enumerate(comidlist):
            if i in printselection and i>0:
                progress=np.round(100.0*i/comidcount,1)
                failrate=np.round(100.0*huc12failcount/i,5)
                self.logger.info(f"{mypid}'s progress:{progress}%, failrate:{failrate}")
            hucdatadict=self.findcomidhuc12reach(comid_i)
            if hucdatadict==None: 
                huc12findfaillist[i]=1
                huc12failcount+=1

            found=0
            try:
                j=self.sitedata_comid_digits.index(comid_i)
                found=1
            except:                
                sitedatacomid_dict[comid_i]='Not Found'
                comidsitedataidx.append('Not Found')

                comidsiteinfofindfaillist[i]=1
            if found==1:
                sitedict=self.sitedata[j]
                comidsitedataidx.append(j)
                if type(hucdatadict) is dict:
                    sitedict=self.mergelistofdicts([sitedict,hucdatadict])
                sitedatacomid_dict[comid_i]=sitedict
            #if i in printselection:print('i==',i,comid_i)
        return (comidsitedataidx,sitedatacomid_dict,comidsiteinfofindfaillist,huc12findfaillist)
                
        
    def findcomidhuc12reach(self,comid):
        try:self.NHDplus,self.NHDpluscomidlist,self.NHDvarlist
        except:
            self.logger.exception('could not access NHDplus in self')
            self.getNHDplus()
        #p#rint('self.NHDpluscomidlist[0]',self.NHDpluscomidlist[0],type(self.NHDpluscomidlist[0]))
        itemcount=len(self.NHDplus)
        #comid_digits=comid#''.join(filter(str.isdigit,comid))
        datadict={}
        try:
            i=self.NHDpluscomidlist.index(comid)
            
        except:
            print(f'{comid} huc12find failed.',end=',')
            return None
        for key in self.NHDvarlist:
            datadict[key]=self.NHDplus.loc[i,key]
        return datadict    

####################################################################
####################################################################
##################the next 3 methods run together###################
####################################################################
####################################################################
        
    def buildspecieshuccomidlist(self,species_idx_list=None):
        if species_idx_list is None:
            
            full_list=1
            species_searchcount=len(self.specieslist)
            species_idx_list=[i for i in range(species_searchcount)]
            filepath=os.path.join(self.savedir,'specieshuccomid')
            if os.path.exists(filepath):
                try:
                    with open(filepath,'rb') as f:
                        specieshuccomidtup=pickle.load(f)
                    self.specieshuc_allcomid=specieshuccomidtup[0]
                    self.species01list=specieshuccomidtup[1]
                    print(f'opening {filepath} with length:{len(specieshuccomidtup)} and has first item length: {len(self.specieshuc_allcomid)} and type:{type(self.specieshuc_allcomid)}')
                    return
                except: print(f'error when opening {filepath}, rerunning buildspecieshuccomidlist')
        else: 
            full_list=0
            species_searchcount=len(species_idx_list)
        species01list=[[] for _ in range(species_searchcount)]
        specieshuc_allcomid=[[] for _ in range(species_searchcount)]
        
        for i,idx in enumerate(species_idx_list):
            try:
                if i%(species_searchcount//2)==0:
                    print('')
                    print(f'{round(100*i/species_searchcount,1)}%',end=',')
            except:
                
                pass
            
            print(f'idx:{idx}',end='. ')
            foundincomidlist=self.speciescomidlist[idx]
            hucidxlist=self.specieshuclist_survey_idx[idx]
            try:
                hucidxlist.extend(self.specieshuclist_survey_idx_newhucs[idx])
            except:
                self.logger1.exception('error appending new hucs.')
            species_huc_count=len(hucidxlist)
            #p#rint('huc_count:',species_huc_count,sep=',')
            
            for j,hucidx in enumerate(hucidxlist):
                try:
                    if j%(species_huc_count//10)==0:
                        print(f'{round(100*j/species_huc_count,1)}%',end=',')
                except:pass
                allhuccomids=self.huccomidlist_survey[hucidx]
                specieshuc_allcomid[i].extend(allhuccomids)
                #p#rint('len(specieshuc_allcomid[i])',len(specieshuc_allcomid[i]),'specieshuc_allcomid[i][-1]',specieshuc_allcomid[i][-1],end=',')
                for comid in allhuccomids:
                    if comid in foundincomidlist:
                        species01list[i].append(1)
                    else:
                        species01list[i].append(0)
        #print('len(species01list)',len(species01list))
        #print(f'specieshuc_allcomid length: {len(specieshuc_allcomid)} and type:{type(specieshuc_allcomid)}')
        #print(f'species01list length: {len(species01list)} and type:{type(species01list)}')
        if full_list==1:    
            self.specieshuc_allcomid=specieshuc_allcomid
            self.species01list=species01list    
            with open(filepath,'wb') as f:
                pickle.dump((specieshuc_allcomid,species01list),f) 
            return
        if full_list==0:
            return specieshuc_allcomid,species01list

    
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
        try:self.sitedatacomid_dict,self.comidsitedataidx
        except: self.buildCOMIDsiteinfo()
        #try: self.species01list,self.specieshuc_allcomid
        #except: self.buildspecieshuccomidlist()
        with open(os.path.join(datadir,'sitedatakeylist'),'wb') as f:
            pickle.dump(self.sitedatakeylist,f)
        speciescount=len(self.specieslist)
        
        split_idx=[int(i) for i in np.linspace(0,speciescount,self.processcount+1)]#+1 to include the end
        speciesidxlist=[i for i in range(speciescount)]
        speciesidxlistlist=[]
        for i in range(self.processcount):
            speciesidxlistlist.append(speciesidxlist[split_idx[i]:split_idx[i+1]])
        starttime=time()
        print('pool starting at',starttime)
        with mp.Pool(processes=self.processcount) as pool:
            outlist=pool.map(self.mp_buildspeciesdata01_file,speciesidxlistlist)
            sleep(5)
            i=0
            while len(outlist)<self.processcount:
                print(f'len(outlist):{len(outlist)}, i:{i}')
                sleep(2)
                
                
            pool.close()
            pool.join()
        self.outlist2=outlist
        endtime=time()
        print(f'pool complete at {endtime}. time elapsed: {endtime-starttime}, len(outlist)={len(outlist)}')
        print(f'the length of each list in outlist:{[len(item) for item in outlist]}.')
        if self.processcount>1:
            fail_record=zip(*outlist)
            fail_record=[record for recordlist in fail_record for record in recordlist]
        else:
            fail_record=outlist
        self.buildspeciesdata01_file_fail_record=outlist
        
    def mp_buildspeciesdata01_file(self,speciesidx_list):
        datadir=os.path.join(self.savedir,'speciesdata01')
        logdir=os.path.join(datadir,'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=f'pidatahuc12{os.getpid()}.log'
        handler=logging.FileHandler(os.path.join(logdir,handlername))
        self.logger = logging.getLogger(__name__+str(os.getpid()))
        self.logger.addHandler(handler)
        fail_record=[];recordfailcount=0
        for i,idx in enumerate(speciesidx_list):
            spec_i=self.specieslist[idx]
            try:
                species_filename=os.path.join(datadir,spec_i+'.data')
                if not os.path.exists(species_filename):
                    try: 
                        self.specieshuc_allcomid
                        specieshuc_allcomid=self.specieshuc_allcomid[idx]
                        species01list=self.species01list[idx]
                    except AttributeError:
                        specieshuc_allcomid_list,species01list_list=self.buildspecieshuccomidlist(species_idx_list=[idx])
                        specieshuc_allcomid=specieshuc_allcomid_list[0]
                        species01list=species01list_list[0]
                    species_n=len(specieshuc_allcomid)
                    varcountlist=[len(self.sitedatacomid_dict[comidk].items()) for comidk in specieshuc_allcomid]
                    varcount=max(varcountlist)
                    maxvarcountcomid=specieshuc_allcomid[varcountlist.index(varcount)]
                    keylist=[key for key,_ in self.sitedatacomid_dict[maxvarcountcomid].items()]
                    #p#rint('varcount',varcount)
                    speciesdata=np.empty((species_n,varcount+1),dtype=object)#+1 for dep var
                    speciesdata[:,0]=np.array(species01list)
                    for j,comidj in enumerate(specieshuc_allcomid):
                        sitevars=[val for _,val in self.sitedatacomid_dict[comidj].items()]
                        try: speciesdata[j,1:]=np.array(sitevars)
                        except: 
                            self.logger.exception(f'i:{i},idx:{idx},species:{spec_i}, comid:{comidj}')
                            keylistj=[key for key,_ in self.sitedatacomid_dict[comidj].items()]
                            missingkeys=[]
                            for k,key in enumerate(keylist):
                                try:#added try: to handle missing bmmi values even if key exists
                                    speciesdata[j,1+k]=self.sitedatacomid_dict[comidj][key]
                                except:
                                    missingkeys.append(key)
                                    speciesdata[j,1+k]='999999'
                            self.logger.warning(f'missing keys from exception are: {missingkeys}')
                    with open(species_filename,'wb') as f:
                        pickle.dump(speciesdata,f)  
                    self.logger.info(f'i:{i},idx:{idx},species:{spec_i}. speciesdata.shape:{speciesdata.shape}')
                else:
                    self.logger.info(f'{species_filename} already exists')
                fail_record.append(0)
            except:
                self.logger.exception(f'problem came up for species number {i},{spec_i}')
                try: 
                    fail_record.append((spec_i,missingkeys))
                except: 
                    fail_record.append((spec_i,'none'))
                recordfailcount+=1
        self.logger.warning(f'succesful completion. len(speciesidx_list): {len(speciesidx_list)}, recordfailcount: {recordfailcount}')
        return fail_record
            
        
        
    
    
if __name__=='__main__':
    test=DataTool()
    test.getfishdata()
    test.getsitedata()
    test.getfishhucs()
    test.gethucdata()
    test.getNHDplus()
    test.buildspecieslist()
    test.buildspecieshuc8list()
    test.buildCOMIDlist()
    test.buildCOMIDsiteinfo()
    #test.buildspecieshuccomidlist()
    test.buildspeciesdata01_file()
    
