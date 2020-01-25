from random import shuffle
import numpy as np
from pisces_data_huc12 import PiscesDataTool

class datagen(PiscesDataTool):
    '''generates numpy arrays of random training or validation for model: y=xb+e or variants
    '''
    #def __init__(self, data_shape=(200,5), ftype='linear', xval_size='same', sparsity=0, xvar=1, xmean=0, evar=1, betamax=10):
    def __init__(self,datagen_dict)
        '''(self,source=None,
                 seed=None,
                 ftype=None,
                 evar=None,
                 batch_n=None,
                 param_count=None,
                 batchcount=None,
                 validate_batchcount=None):'''
        source=datagen_dict['source']
        if source=='monte':
            param_count=datagen_dict['param_count']
            if param_count==None:
                param_count=1
            self.param_count=param_count
            self.p = param_count
            batch_n=datagen_dict['batch_n']
            if batch_n==None:
                batch_n=32
            self.batch_n = batch_n
            batchcount=datagen_dict['batchcount']
            if batchcount==None:
                batchcount=1
            self.batchcount=batchcount
            validate_batchcount=datagen_dict['validate_batchcount']
            if validate_batchcount==None:
                validate_batchcount=batchcount
            self.validate_batchcount=validate_batchcount
            self.source = source
            seed=datagen_dict['seed']
            if seed is None:
                seed=1
            self.gen_montecarlo(seed=seed,ftype=ftype,evar=evar,batch_n=batch_n,param_count=param_count,batchcount=batchcount)
            return
        if source=='pisces':
            PiscesDataTool.__init__(self,)
            
            source=datagen_dict['source']
            batch_n=datagen_dict['batch_n']
            self.batch_n=batch_n
            batchcount=datagen_dict['batchcount']
            self.bachcount=batchcount
            sample_replace==datagen_dict['sample_replace']
            missing=datagen_dict['missing']
            species=datagen_dict['species']
            self.species=species
            seed=1
            
            self.gen_piscesdata01(seed,batch_n,batchcount,sample_replace,missing,species)
            pass
        
        
            

    def gen_piscesdata01(self,seed,batch_n,batchcount,sample_replace,missing,species):
        try:self.specieslist
        except: self.buildspecieslist()
        self.varlist=self.retrievespeciesdata()
        if type(species) is str:
            speciesdata=self.retrievespeciesdata(species_name=species)
        if type(species) is int:
            speciesdata=self.retrievespeciesdata(species_idx=species)
        
            
            
        if species_idx is None:
            speciesdata=self.retrievespeciesdata(species_name=spec_name)
        else:
            speciesdata=self.retrievespeciesdata(species_idx=species_idx)
        
        floatselecttup=(0,1,2,3)
        spatialselecttup=(9,)#9 should be huc12
        dataselecttup=(0,)+floatselecttup+spatialselecttup
        speciesdata=speciesdata[:,dataselecttup]
        speciesdata=self.processmissingvalues(speciesdata,missing)
        
        n=speciesdata.shape[0]
        
        floatselecttup=(0,1,2,3)#5 is bmmi, which is left out for now
        datagen_obj.param_count=len(dataselecttup)-1#-1 bc dep var included in the tupple
        
        self.xvarname_list={}
        self.xvarname_list=self.varlist[floatselecttup-1]#i-1 b/c no dep var in self.varlist
        self.xvarname_list.append(self.varlist[spatialselecttup-1])
        print('self.xvarnames: {self.xvarnames}')
        
        self.xdataarray=np.array(speciesdata[:,floatselecttup+spatialselecttup],dtype=float)
        
        self.spatial=1
        self.float_loc=[i for i in range(len(floatselecttup))]
        #self.xdataarray_spatial=np.array(speciesdata[:,spatialselecttup],dtype=str)
        print(f'self.svarname_dict:{self.svarname_dict}')
        
        
        self.ydataarray=np.array(speciesdata[:,0],dtype='uint8')
        
        modeldict_data_std_tup=([],[i for i in floatselecttup])
        
        self.genpiscesbatchbatchlist(self.ydataarray,self.xdataarray,batch_n,batchcount,sample_replace,missing)
        return
        
    def processmissingvalues(self,nparray,missing_treatment):
        outlist=[]
        if missing_treatment=='drop_row':
            for row in nparray:
                keep=1
                for val in row:
                    if val=='999999':keep=0
                if keep==1:outlist.append(row)
        return np.array(outlist)
                
                
        
        
    def genpiscesbatchbatchlist(self, ydataarray,xdataarray,batch_n,batchcount,sample_replace,missing):
        n=ydataarray.shape[0]; p=xdataarray.shape[1]
        selectlist=shuffle([i for i in range(n)])
        batchsize=batch_n*batchcount
        batchbatchcount=-(-n//batchsize)#ceiling divide
        self.batchbatchcount=batchbatchcount
        fullbatchbatch_n=batchbatchcount*batchsize
        fullbatchbatch_shortby=fulbatchbatch_n-n
        selectlist=selectlist+selectlist[:fullbatchbatch_shortby]#repeat the first group of random observations to fill out the dataset
        
        batchbatchlist=[[[] for b in batchcount] for _ in range(batchbatchcount)]
        for i in range(batchbatchcount):
            for j in range(batchcount):
                start=(i+j)*batch_n
                end=start+batch_n
                batchbatchlist[i][j]=(ydataarray[start:end],xdataarray[start:end,:])
        self.yxtup_batchbatch=batchbatchlist
        
                
            
            
    
        
        
        
        
            
        
            
            

        
        
            
    def gen_montecarlo(self,seed=None,ftype=None,evar=None,batch_n=None,param_count=None,batchcount=None,validate_batchcount=None):
        if ftype==None:
            ftype='linear'
        if batch_n==None:
            batch_n=32
        if param_count==None:
            param_count=1
        if batchcount==None:
            batchcount=1
        if not seed==None:
            seed=1
        if validate_batchcount==None:
            validate_batchcount=batchcount
        
        self.datagen_dict={'validate_batchcount':10,'batch_n':64,'batchcount':10, 'param_count':param_count,'seed':1, 'ftype':'linear', 'evar':1, 'source':'monte'}
        
        p=param_count
        n=batch_n

        if evar==None:
            evar=1
        yxtup_list=[]
        for i in range(batchcount):
            yxtup_list.append(self.buildrandomdataset(n,p,ftype,evar))
        self.yxtup_list=yxtup_list
        self.y=yxtup_list[-1][0]
        self.x = yxtup_list[-1][1]
        val_yxtup_list=[]
        for i in range(validate_batchcount):
            val_yxtup_list.append(self.buildrandomdataset(n,p,ftype,evar))
        self.val_yxtup_list=val_yxtup_list
        return

    def buildrandomdataset(self,n,p,ftype,evar):
        betamax = 10
        xtall=3
        xwide=2
        #random row vector to multiply by each random column of x to allow s.d. upto 5
        spreadx=np.random.randint(xwide, size=(1,p))+1
        shiftx=np.random.randint(0,xtall, size=(1,p))-xtall/2#random row vector to add to each column of x
        randx=np.random.randn(n,p)

        x=shiftx+spreadx*randx
        xvars=x
        if ftype=='quadratic':
            p=2*p#-1 for constant
            x=np.concatenate([x,x**2],axis=1)

        #generate error~N(0,1)
        self.e=np.random.randn(n)*evar**.5

        #make beta integer
        b=(np.random.randint(betamax, size=(p+1,)))

        #add column of 1's to x
        x = np.concatenate((np.ones([n,1]),x),axis=1)
        #calculate y
        y=np.matmul(x, b)+self.e
        #return y and the original x variables, not the squared terms or constant
        return (y,xvars)