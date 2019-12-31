import os
import csv
import traceback

class DataTool():
    def __init__(self,):
        self.savedir=os.path.join(os.getcwd(),'data_tool')
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        #self.filenamelist=['HUC12_PU_COMIDs_CONUS.csv','siteinfo.csv','surveydata.csv']
        
        
    def getdatafile(self,filename):
        thisdir=os.getcwd()
        datadir=os.path.join(thisdir,'fishfiles',filename)
        if os.path.exists(datadir):
            with open(datadir, 'r') as f:
                datadict=[row for row in csv.DictReader(f)]
        print(f'opening {filename} with length:{len(datadict)} and type:{type(datadict)}')
        keylist=[key for row in datadict for key,val in row.items()]
        return datadict

    def getfishdata(self,):
        self.fishsurveydata=self.getdatafile('surveydata.csv')

    def gethucdata(self,):
        self.hucdata=self.gatdatafile('HUC12_PU_COMIDs_CONUS.csv')

    def getsitedata(self,):
        self.sitedata=self.getdatafile('siteinfo.csv')
        
    def getfishhucs(self,):
        self.fishhucs=self.getdatafile('fishhucs.csv')

        
    def buildspecieslist(self,):
        specieslistpath=os.path.join(self.savedirectory,'specieslist')
        if os.path.exists(specieslistpath):
            try:
                with open(specieslistpath,'rb') as f:
                    speciestup=pickle.load(f)
                self.specieslist=speciestup[0]
                self.speciesoccurencelist=speciestup[1]
                self.speciescomidlist=speciestup[2]
                self.specieshuclist=speciestup[3]
                self.huclist=speciestup[4]
                self.huccomidlist=speciestup[5]
                self.specieshuclistidx=speciestup[6]
                return
            except:
                print(f'buildspecieslist found {specieslistpath} but there was an error loading variables')
                print(traceback.format_exc())
                
        try:self.fishsurveydata
        except: self.getfishdata()
        (longlist,huclist,comidlist)=zip(*[(obs['genus_species'],obs['COMID']) for obs in self.fishsurveydata])
        shortlist=[];occurencelist=[];speciescomidlist=[];specieshuclist=[]
        shorthuclist=[];huccomidlist=[];specieshuclistidx=[]
        print('building specieslist')
        for idx,fish in enumerate(longlist):
            if idx%(length//33)==0:
                print(str(round(100*idx/length))+'%')
            found=0
            for shortidx,fish_i in enumerate(shortlist):
                if fish==fish_i:
                    occurencelist[shortidx].append(idx)
                    speciescomidlist[shortidx].append(comidlist[idx])
                    specieshuclist[shortidx].append(huclist[idx])
                    found=1
                    break
            if found==0:
                shortidx=len(shortlist)#-1 not needed since not yet appended
                shortlist.append(fish)
                occurencelist.append([idx])
                speciescomidlist.append([comidlist[idx]])
                specieshuclist.append([huclist[idx]])
                #print(f'new fish:{fish}')
            
            hucfound=0
            huc=huclist[idx]; 
            if not type(huc) is str:
                huc=str(huc)
            if len(huc)==7:
                huc=0+huc
            assert len(huc)==8,print(f'expecting 8 characters: huc8_i:{huc}')
            
            
            for hucshortidx,huc_i in shorthuclist:
                if huc==huc_i
                huccomidlist[hucshortidx].append(comidlist[idx])
                specieshuclistidx[shortidx].append(huc)
                hucfound==1
                break
            if hucfound==0:
                shorthuclist.append(huc)
                huccomidlist.append([comidlist[idx]])
        print(f'buildspecieslist found {len(shortlist)} unique strings')
        
        with open(specieslistpath,'wb') as f:
            pickle.dump((shortlist,occurencelist,speciescomidlist,specieshuclist,shorthuclist,huccomidlist,specieshuclistidx),f)
        
        self.specieslist=shortlist
        self.speciesoccurencelist=occurencelist
        self.speciescomidlist=speciescomidlist
        self.specieshuclist=specieshuclist
        self.huclist=shorthuclist
        self.huccomidlist=huccomidlist
        self.specieshuclistidx=specieshuclistidx
            
    

            
            
    
    def buildCOMIDlist(self,):
        comidlistpath=os.path.join(self.savedirectory,'comidlist')
        if os.path.exists(comidlistpath):
            with open(comidlistpath,'rb') as f:
                comidtup=pickle.load(f)
            self.comidlist=comidtup[0]
            self.comidcurrurencelist=comidtup[1]
            return
        try: self.fishsurveydata
        except: self.getfishdata()
        print('building comidlist')
        longlist=[obs['COMID'] for obs in self.fishsurveydata]
        
        shortlist=[];occurencelist=[]
        length=len(longlist)
        i=0
        for idx,comid in enumerate(longlist):
            if idx%(length//33)==0:
                print(str(round(100*idx/length))+'%')
            found=0
            for shortidx,comid_i in enumerate(shortlist):
                if comid_i==comid:
                    occurencelist[shortidx].append(idx)
                    #print(f'old comid:{comid}')
                    found=1
                    break
            if found==0:
                shortlist.append(comid)
                occurencelist.append([idx])
        print(f'buildCOMIDlist found {len(shortlist)}')
        with open(comidlistpath,'wb') as f:
            pickle.dump((shortlist,occurencelist),f)
        
        self.comidlist=shortlist  
        self.comidoccurenclist=occurencelist
    
    def buildCOMIDsiteinfo(self,):
        filepath=os.path.join(self.savedirectory,'sidedatacomid_dict')
        if os.path.exists(filepath):
            try:
                with open(filepath,'rb') as f:
                    self.sitedatacomid_dict=pickled.load(f)
                return
            except:
                print(f'buildCOMIDsiteinfo found {filepath} but could not load it, so rebuilding')
           
        try: self.sitedata
        except:self.getsitedata()
        try:self.comidlist
        except:self.buildCOMIDlist()
            
        self.sitedata_k=len(self.sitedata[0])
        self.sitevarkeylist=[key for key,_ in self.sitedata[0]]

        self.sitedataidx=[]
        self.sitedatacomid_dict={}
        for i,comid_i in enumerate(self.comidlist):
            for j,sitedict in enumerate(self.sitedata):
                comid_j=sitedict['COMID']
                if comid_i==comid_j:
                    self.sitedataidx.append(j)
                    self.sitedatacomid_dict[comid_j]=sitedict
        with open(filepath,'wb') as f:
            pickle.dump(self.sitedatacomid_dict,f)
        return
 
    def buildspecieshuc8list():
        try: self.fishhucs
        except self.getfishhucs()
        specieshuc8list=[]*len(self.specieslist)
        lastid=None
        for item,i in enumerate(self.fishhucs):
            spec_i=item['Scientific_name']
            #d_i=item['ID']
            huc8_i=item['HUC']
            if not type(huc8_i) is str:
                huc8_i=str(huc8_i)
            if len(huc8_i)==7:
                huc8_i=0+huc8_i
            assert len(huc8_i)==8,print(f'expecting 8 characters: huc8_i:{huc8_i}')
            if not spec_i==lastspec_i:
                for j,spec_j in enumerate(self.specieslist):
                    if spec_i==spec_j:
                        specieslist_idx=j
            specieshuc8list[specieslist_idx].append(huc8_i)
            lastspec_i=spec_i    
        self.specieshuc8list=specieshuc8list
        
    def buildspecieshuccomidlist(self,):
        speciescount=len(self.specieslist)
        species01list=[]*speciescount
        specieshuc_allcomid=[]*speciescount
        for i,spec in enumerate(self.specieslist):
            foundincomidlist=self.speciescomidlist[i]
            hucidxlist=self.specieshuclistidx[i]
            for hucidx in hucidxlist:
                allhuccomids=self.huccomidlist[hucidx]
                specieshuc_allcomid[i].extend(allhuccomids)
                for comid in allhuccomids:
                    if comid in foundincomidlist:
                        species01list[i].extend(1)
                    else:
                        species01list[i].extend(0)
                
        self.specieshuc_allcomid=specieshuc_allcomid
        self.species01list=species01list

    def buildspeciesdatadict01(self,):
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
        try: self.species01list
        except: self.buildspecieshuccomidlist()
        
        
        for i,spec_i in enumerate(self.specieslist):
            species_dir=os.path.join(datadir,spec_i+'.data')
            if not os.path.exists(species_dir):
                species_n=len(self.specieshuc_allcomid)
                varcount=1+self.sitedata_k
                speciesdata=np.empty(species_n,varcount)
                speciesdata[:,0]=np.array(self.species01list).reshape(species_n,1)
                for j,comid in enumerate(self.specieshuc_allcomid[i]):
                    sitevars=[self.sitedatacomid_dict[comid][key] for key in self.sitevarkeylist]
                    speciesdata[j,1:]=np.array(sitevars).reshape(1,self.sidedata_k)
            else: print(f'{species_dir} already exists')
            species_dir=os.path.join(datadir,spec_i+'.data')
            with open(species_dir,'wb') as f:
                pickle.dump(speciesdata,f)
        return
            
        
        
    
    
if __name__=='__main__':
    test=DataTool()
    #test.getfishdata()
    #test.buildspecieslist()
    #test.buildCOMIDlist()
    #test.buildCOMIDsiteinfo()
    test.buildspeciesdatadict01()
    
