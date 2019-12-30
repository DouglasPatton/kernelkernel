import os
import csv

class DataTool():
    def __init__(self,):
        self.filenamelist=['HUC12_PU_COMIDs_CONUS.csv','siteinfo.csv','surveydata.csv']
        
        
    def getdatafile(self,filename):
        thisdir=os.getcwd()
        datadir=os.path.join(thisdir,'fishfiles',filename)
        if os.path.exists(datadir):
            with open(datadir, 'r') as f:
                datadict=[row for row in csv.DictReader(f)]
        print(len(datadict),type(datadict))
        
        print(datadict[0:5])
        keylist=[key for row in datadict for key,val in row.items()]
        print(keylist[0:20])
        return datadict

    def getfishdata(self,):
        self.fishsurveydata=self.getdatafile('surveydata.csv')

    def gethucdata(self,):
        self.hucdata=self.gatdatafile('HUC12_PU_COMIDs_CONUS.csv')

    def getsitedata(self,):
        self.sitedata=self.getdatafile('siteinfo.csv')

        
    def buildspecieslist(self,):
        (longlist,comidlist)=zip(*[(obs['genus_species'],obs['COMID']) for obs in self.fishsurveydata])
        shortlist=[];occurencelist=[];speciescomidlist=[]
        for idx,fish in enumerate(longlist):
            found=0
            for shortidx,fish_i in enumerate(shortlist):
                if fish==fish_i:
                    occurencelist[shortidx].append(idx)
                    speciescomidlist[shortidx].append(comidlist[idx])
                    found=1
                    break
            if found==0:
                shortlist.append(fish)
                occurencelist.append([idx])
                speciescomidlist.append([comidlist[idx]])
                print(f'new fish:{fish}')
            
        self.specieslist=shortlist
        self.speciesoccurencelist=occurencelist

    def buildCOMIDfishlist(self,):
        longlist=[obs['COMID'] for obs in self.fishsurveydata]
        shortlist=[]
        for fish in longlist:
            if not fish in shortlist:
                shortlist.append(fish)
                print(f'new fish:{fish}')
            #else:print(f'old fish:{fish}')
        
        self.specieslist=shortlist    
    
        
    
    
    
if __name__=='__main__':
    test=DataTool()
    test.getfishdata()
    test.buildspecieslist()
    
