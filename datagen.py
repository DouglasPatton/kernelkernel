import logging
import random
import numpy as np
from pisces_data_huc12 import PiscesDataTool
import os
import sklearn
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
from mylogger import myLogger
class datagen(PiscesDataTool,myLogger):
    '''
    
    generates numpy arrays of random training or validation for model: y=xb+e or variants
    '''
    #def __init__(self, data_shape=(200,5), ftype='linear', xval_size='same', sparsity=0, xvar=1, xmean=0, evar=1, betamax=10):
    def __init__(self,datagen_dict,):
        myLogger.__init__(self,name='datagen.log')
        self.logger.info('starting new datagen log')
        
        
        #handler=logging.RotatingFileHandler(os.path.join(logdir,handlername),maxBytes=8000, backupCount=5)
        #self.logger = logging.getLogger(__name__)
        #self.logger.addHandler(handler)
        
        
        source=datagen_dict['source']
        self.initial_datagen_dict=datagen_dict
        self.datagen_dict=datagen_dict
        if source=='monte':
            try:
                theseed=datagen_dict['seed'] #just used for montecarlo
            except:
                theseed=1
            if theseed is None:
                theseed=1
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
              
            
            self.gen_montecarlo(seed=theseed,ftype=ftype,evar=evar,batch_n=batch_n,param_count=param_count,batchcount=batchcount)
            
            ydata=np.concatenate([yxtup[0] for yxtup in self.yxtup_list],axis=0)
            xdata=np.concatenate([yxtup[1] for yxtup in self.yxtup_list],axis=0)
            self.build_sumstats_dict(ydata,xdata)
            return
        
        if source=='pisces':
            PiscesDataTool.__init__(self,)
            
            source=datagen_dict['source']
            """batch_n=datagen_dict['batch_n']
            self.batch_n=batch_n
            batchcount=datagen_dict['batchcount']
            self.batchcount=batchcount
            self.batchbatchcount=datagen_dict['max_maxbatchbatchcount']
            sample_replace=datagen_dict['sample_replace']"""
            #missing=datagen_dict['missing']
            species=datagen_dict['species']
            #self.species=species
            """if datagen_dict['sample'][0]=='y':
                if datagen_dict['sample'][2:]=='balanced':
                    self.min_y=0.5
                else: assert False,'not developed'
            else:
                self.ysample=None
            
            
            
            
            #the next two selection tups are for the independent (x) variable and they select from xdata not yxdata
            floatselecttup=datagen_dict['floatselecttup']
            self.floatselecttup=floatselecttup
            spatialselecttup=datagen_dict['spatialselecttup']
            self.spatialselecttup=spatialselecttup
            self.param_count=datagen_dict['param_count']
            
            seed=1"""
            
            #self.gen_piscesdata01(species,datagen_dict)
            #self.build_sumstats_dict(self.ydataarray,self.xdataarray) # moved inside gen_piscesdata01
            
            pass
        
    def expand_datagen_dict(self,key,val):
        try: self.datagen_dict_expanded
        except: self.datagen_dict_expanded=self.initial_datagen_dict
        self.datagen_dict_expanded[key]=val

            

    def gen_piscesdata01(self,species,datagen_dict):
        
        random.seed(seed)  
        try:self.specieslist
        except: self.buildspecieslist()
        self.fullvarlist=self.retrievespeciesdata()
        if type(species) is str:
            speciesdata=self.retrievespeciesdata(species_name=species)
        if type(species) is int:
            speciesdata=self.retrievespeciesdata(species_idx=species)
        
        floatselect_plus1=[i+1 for i in floatselecttup]
        spatialselect_plus1=[i+1 for i in spatialselecttup]#plus1 required because selecttups are defined for x matrix, not full data matrix
        dataselecttup=[0]+floatselect_plus1+spatialselect_plus1
        speciesdata=speciesdata[:,dataselecttup]
        #print(f'speciesdata[0:5,:]:{speciesdata[0:5,:]}')
        
        speciesdata=self.processmissingvalues(speciesdata,missing)
        if len(spatialselecttup)>0:
            self.spatial=1
        else: self.spatial=0
        
        
        n=speciesdata.shape[0]
        self.logger.info(f'species_n:{n}')
        self.species_n=n
        self.expand_datagen_dict('species_n',self.species_n)
        #floatselecttup=(0,1,2,3)#5 is bmmi, which is left out for now
        #datagen_obj.param_count=len(dataselecttup)-1#-1 bc dep var included in the tupple
        
        #self.xvarname_list=[]
        self.xvarname_list=[self.fullvarlist[i] for i in floatselecttup]
        self.xvarname_list.extend([self.fullvarlist[i]+'(spatial)' for i in spatialselecttup])
        self.expand_datagen_dict('xvarnamelist',self.xvarname_list)
        self.logger.info(f'self.xvarname_list: {self.xvarname_list}')
        try: xdata=np.array(speciesdata[:,1:],dtype=float)
        except ValueError:
            k=speciesdata.shape[1]
            for row in range(n):
                for col in range(k):
                    try: float(speciesdata[row,col])
                    except: 
                        self.logger.exception('')
                        self.logger.info(f'speciesdata[row,:],row,col:{[speciesdata[row,:],row,col]}')
            
        #self.xdataarray=np.array(self.fullxdataarray[:,floatselecttup+spatialselecttup],dtype=float)
        
        '''
        self.float_loc=[i for i in range(len(floatselecttup))]
        #self.xdataarray_spatial=np.array(speciesdata[:,spatialselecttup],dtype=str)
        '''
        
        
        ydata=np.array(speciesdata[:,0])
        xtrain,xtest,ytrain,ytest=train_test_split(xdata,ydata,stratify=ydata,test_size=0.2,random_state=0)
        self.build_sumstats_dict(xtrain,ytrain) 
        xtrain,xtest,ytrain,ytest=self.standardize_yxtraintest(xtrain,xtest,ytrain,ytest,std_data=self.std_data)
        self.xdataarray=xtrain
        self.ydataarray=ytrain    
        self.xtestarray=xtest
        self.ytestarray=ytest
        
        
        train_n=self.ydataarray.shape[0]
        self.logger.info(f'train_n:{train_n}')
        self.train_n=train_n
        self.expand_datagen_dict('train_n',train_n)
        #sss.split(xdata,ydata)
        #print('self.ydataarray',self.ydataarray,type(self.ydataarray))
        
        
        
        self.genpiscesbatchbatchlist(ytrain,xtrain,batch_n,batchcount,min_y=self.min_y)
        self.genpisces_test_batchbatchlist(ytest,xtest,batch_n,self.batchcount)
        return
    
    
        
        
    def build_sumstats_dict(self,xdata,ydata):
        try:
            existing_sum_stats_dict=self.initial_datagen_dict['summary_stats_dict']
            self.summary_stats_dict=existing_sum_stats_dict
            return
        except:
            self.logger.info(f'build sumstats dict building',exc_info=True)        
        self.xmean=np.mean(xdata,axis=0)
        self.ymean=np.mean(ydata,axis=0)
        self.xstd=np.std(xdata,axis=0)
        self.ystd=np.std(ydata,axis=0)
        self.summary_stats_dict={'xmean':self.xmean,
                                'ymean':self.ymean,
                                'ystd':self.ystd,
                                'xstd':self.xstd,
                                'max_train_n':ydata.shape[0]} # remove this too?
        self.expand_datagen_dict('summary_stats_dict',self.summary_stats_dict)
        
    def processmissingvalues(self,nparray,missing_treatment):
        #rewrite with fancy indexing
        outlist=[]
        if missing_treatment=='drop_row':
            for row in nparray:
                keep=1
                for val in row:
                    if val == '999999' or val=='' :keep=0
                if keep==1:outlist.append(row)
        arr= np.array(outlist,dtype=np.float64)
        neg_infs=np.sum(np.isinf(arr))
        if neg_infs:
            self.logger.critical(f'processing missing -infinitis in data: {neg_infs}')
        return arr     
                
        
        
    def genpiscesbatchbatchlist(self, ydataarray,xdataarray,batch_n,batchcount,min_y=1):
        # test data already removed
        batchbatchcount=self.batchbatchcount
        n=ydataarray.shape[0]; p=xdataarray.shape[1]
        onecount=int(ydataarray.sum())
        zerocount=n-onecount
        countlist=[zerocount,onecount]
        if onecount<zerocount:
            smaller=1
        else:
            smaller=0
        
        if not min_y is None:
            if min_y<1:
                min_y=int(batch_n*min_y)
            batch01_n=[None,None]
            batch01_n[smaller]=min_y # this makes it max_y too....
            batch01_n[1-smaller]=batch_n-min_y
            max_batchcount=countlist[smaller]//min_y
            if max_batchcount<batchcount:
                #self.logger.info(f'for {self.species} batchcount changed from {batchcount} to {max_batchcount}')
                batchcount=max_batchcount
            subsample_n=batchcount*batch_n
            oneidx=np.arange(n)[ydataarray==1]
            zeroidx=np.arange(n)[ydataarray==0]
            bb_select_list=[]
            for bb_idx in range(batchbatchcount):
                ones=np.random.choice(oneidx,size=batch01_n[1]*batchcount,replace=False)
                zeros=np.random.choice(zeroidx,size=batch01_n[0]*batchcount,replace=False)
                bb_select_list.append(np.concatenate([ones,zeros],axis=0))
        else:
            max_batchcount=n//batch_n
            if max_batchcount<batchcount:
                #self.logger.info(f'for {self.species} batchcount changed from {batchcount} to {max_batchcount}')
                batchcount=max_batchcount
            subsample_n=batchcount*batch_n
            for bb_idx in range(batchbatchcount):
                bb_select_list.append(np.random.choice(np.arange(n),size=subsample_n),replace=False)
        batchbatchlist=[[None for __ in range(batchcount)] for _ in range(batchbatchcount)]
        SKF=StratifiedKFold(n_splits=batchcount, shuffle=False)
        for bb_idx in range(batchbatchcount):
            bb_x_subsample=xdataarray[bb_select_list[bb_idx],:]
            bb_y_subsample=ydataarray[bb_select_list[bb_idx]]
            for j,(train_index,test_index) in enumerate(SKF.split(bb_x_subsample,bb_y_subsample)):
                batchbatchlist[bb_idx][j]=(bb_y_subsample[test_index],bb_x_subsample[test_index,:])
        batchsize=batch_n*batchcount
        
        
        self.batchcount=batchcount
        self.expand_datagen_dict('batchcount',self.batchcount)
        fullbatchbatch_n=batchbatchcount*batchsize
        self.fullbatchbatch_n=fullbatchbatch_n
        self.expand_datagen_dict('fullbatchbatch_n',self.fullbatchbatch_n)
        self.logger.info(f'yxtup shapes:{[(yxtup[0].shape,yxtup[1].shape) for yxtuplist in batchbatchlist for yxtup in yxtuplist]}')
        self.yxtup_batchbatch=batchbatchlist
        
    def genpisces_test_batchbatchlist(self,y,x,batch_n,batchcount):
        n=y.size
        batchbatchcount=-(-n//(batch_n*batchcount)) # ceiling divide
        batchbatchlist=[[None for b in range(batchcount)] for _ in range(batchbatchcount)]
        idx=0
        for i in range(batchbatchcount):
            for j in range(batchcount):
                start=idx
                stop=idx+batch_n
                if stop>n:stop=n
                batchbatchlist[i][j]=(y[start:stop],x[start:stop,:])
        self.yxtup_batchbatch_test=batchbatchlist
        return 
            
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
        random.seed(seed)
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

    
    def standardize_yxtraintest(self,xtrain,xtest,ytrain,ytest,std_data=None):
        if std_data is None:
            std_data=([],'float')
        #yxtup_list=deepcopy(yxtup_list_unstd)
        #p=yxtup_list[0][1].shape[1]
        p=xtrain.shape[1]
        """#already set by buildsumstatsdict....
        self.xmean=self.datagen_obj.summary_stats_dict['xmean'] # sumstats built off training data
        self.ymean=self.datagen_obj.summary_stats_dict['ymean']
        self.xstd=self.datagen_obj.summary_stats_dict['xstd']
        self.ystd=self.datagen_obj.summary_stats_dict['ystd']
        """
        
        if type(std_data) is str: 
            if  std_data=='all':
                x_stdlist=[i for i in range(p)]
                y_stdlist=[0]
            else:
                assert False, f'modeldict:std_data is {std_data} but expected "all"'
        elif type(std_data) is tuple:
            xmodelstd=std_data[1]
            ymodelstd=std_data[0]
            floatselecttup=self.floatselecttup
            spatialselecttup=self.spatialselecttup
            if type(xmodelstd) is str:
                if xmodelstd=='float':
                    x_stdlist=[i for i in range(len(floatselecttup))]
                if xmodelstd=='all':
                    x_stdlist=[i for i in range(p)]
            if type(xmodelstd) is list:
                x_stdlist=xmodelstd
            if type(ymodelstd) is list:
                y_stdlist=ymodelstd
            #xstdselect=modelstd[1]
            #x_stdlist[modelstd[1]]=1
        
        if x_stdlist:
            xscaler = StandardScaler()
            xscaler.fit(xtrain[:,x_stdlist])                                                   
            xtrain[:,x_stdlist]=xscaler.transform(xtrain[:,x_stdlist])
            xtest[:,x_stdlist]=xscaler.transform(xtest[:,x_stdlist])
        if y_stdlist:
            self.logger.warning(f'y is being standardized!')
            yscaler = StandardScaler()
            yscaler.fit(ytrain)                                                   
            ytrain=yscaler.transform(ytrain)
            ytest=yscaler.transform(ytest)                                                      
                                                
        return xtrain,xtest,ytrain,ytest