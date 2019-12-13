import os
import csv

class DataTool():
    def __init__(self,):
        huc12comidfile=self.gethuc12comidfile()
        
    def gethuc12comidfile(self,):
        thisdir=os.getcwd()
        datadir=os.path.join(thisdir,'fishfiles','HUC12_PU_COMIDs_CONUS.csv')
        if os.path.exists(datadir):
            with open(datadir, 'r') as f:
                datadict=[row for row in csv.DictReader(f)]
        print(len(datadict),type(datadict))
        
        print(datadict[0:5])
        keylist=[key for row in datadict for key,val in row.items()]
        print(keylist[0:20])
    
    
    
if __name__=='__main__':
    test=DataTool()
    
