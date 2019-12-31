import os
import csv

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
            with open(specieslistpath,'rb') as f:
                speciestup=pickle.load(f)
            self.specieslist=speciestup[0]
            self.speciesoccurencelist=speciestup[1]
            self.speciescomidlist=speciestup[2]
            self.specieshuclist=specieshuclist[3]
            self.shorthuclist=specieshuclist[4]
            self.huccomidlist=specieshuclist[5]
            return
        
        try:self.fishsurveydata
        except: self.getfishdata()
        (longlist,huclist,comidlist)=zip(*[(obs['genus_species'],obs['COMID']) for obs in self.fishsurveydata])
        shortlist=[];occurencelist=[];speciescomidlist=[];specieshuclist=[]
        shorthuclist=[];huccomidlist=[]
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
                shortlist.append(fish)
                occurencelist.append([idx])
                speciescomidlist.append([comidlist[idx]])
                specieshuclist.append([huclist[idx]])
                #print(f'new fish:{fish}')
            hucfound=0
            huc=huclist[idx]; 
            
            for shortidx,huc_i in shorthuclist:
                if huc==huc_i
                huccomidlist[shortidx].append(comidlist[idx])
                
                hucfound==1
                break
            if hucfound==0:
                shorthuclist.append(huc)
                huccomidlist.append([comidlist[idx]])
        print(f'buildspecieslist found {len(shortlist)} unique strings')
        
        with open(specieslistpath,'wb') as f:
            pickle.dump((shortlist,occurencelist,speciescomidlist,specieshuclist,shorthuclist,huccomidlist),f)
        
        self.specieslist=shortlist
        self.speciesoccurencelist=occurencelist
        self.speciescomidlist=speciescomidlist
        self.specieshuclist=specieshuclist
        self.shorthuclist=shorthuclist
        self.huccomidlist=huccomidlist
            
            
            
    
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
        try: self.sitedata
        except:self.getsitedata()
        try:self.comidlist
        except:self.buildCOMIDlist()

        self.sidedataidx=[]
        for i,comid_i in enumerate(self.comidlist):
            for j,site in enumerate(self.sitedata):
                comid_j=site['COMID']
                if comid_i==comid_j:
                    self.sitedataidx.append(j)
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
        for i,spec_i in enumerate(self.specieslist):
            huc8list=self.specieshuc8list[i]
            idx_list=self.speciesoccurencelist[i]
            foundlist=self.fishsurveydata[idx_list]
            
        
        
    
    
if __name__=='__main__':
    test=DataTool()
    #test.getfishdata()
    #test.buildspecieslist()
    #test.buildCOMIDlist()
    test.buildCOMIDsiteinfo()
    test.buildspeciesdatadict01()
    
