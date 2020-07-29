import logging
import random
import numpy as np
from pisces_data_huc12 import PiscesDataTool
import os
import sklearn
from sklearn.model_selection import RepeatedStratifiedKFold,train_test_split
class datagen(PiscesDataTool):
    '''
    
    generates numpy arrays of random training or validation for model: y=xb+e or variants
    '''
    #def __init__(self, data_shape=(200,5), ftype='linear', xval_size='same', sparsity=0, xvar=1, xmean=0, evar=1, betamax=10):
    def __init__(self,datagen_dict):
        
        
        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,__name__)
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
            level=logging.WARNING,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.logger.info('starting new datagen log')
        
        
        try:
            self.max_maxbatchbatchcount=datagen_dict['max_maxbatchbatchcount'] # determines the cut-off between training and validation data
        except:
            self.logger.exception('max_maxbatchbatchcount error')
            self.max_maxbatchbatchcount=None
        #handler=logging.RotatingFileHandler(os.path.join(logdir,handlername),maxBytes=8000, backupCount=5)
        self.logger = logging.getLogger(__name__)
        #self.logger.addHandler(handler)
        
        try:
            theseed=datagen_dict['seed'] #just used for montecarlo
        except:
            theseed=1
        if theseed is None:
            theseed=1
        source=datagen_dict['source']
        self.initial_datagen_dict=datagen_dict
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
              
            
            self.gen_montecarlo(seed=theseed,ftype=ftype,evar=evar,batch_n=batch_n,param_count=param_count,batchcount=batchcount)
            
            ydata=np.concatenate([yxtup[0] for yxtup in self.yxtup_list],axis=0)
            xdata=np.concatenate([yxtup[1] for yxtup in self.yxtup_list],axis=0)
            self.build_sumstats_dict(ydata,xdata)
            return
        
        if source=='pisces':
            PiscesDataTool.__init__(self,)
            
            source=datagen_dict['source']
            batch_n=datagen_dict['batch_n']
            self.batch_n=batch_n
            batchcount=datagen_dict['batchcount']
            self.batchcount=batchcount
            sample_replace=datagen_dict['sample_replace']
            missing=datagen_dict['missing']
            species=datagen_dict['species']
            self.species=species
            if datagen_dict['sample'][0]=='y':
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
            
            seed=1
            
            self.gen_piscesdata01(seed,batch_n,batchcount,sample_replace,missing,species,floatselecttup,spatialselecttup)
            self.build_sumstats_dict(self.ydataarray,self.xdataarray)
            
            pass
        
    def expand_datagen_dict(self,key,val):
        try: self.datagen_dict_expanded
        except: self.datagen_dict_expanded=self.initial_datagen_dict
        self.datagen_dict_expanded[key]=val

            

    def gen_piscesdata01(self,seed,batch_n,batchcount,sample_replace,missing,species,floatselecttup,spatialselecttup):
        
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
        
        
        ydata=np.array(speciesdata[:,0],dtype=np.int8)
        xtrain,xtest,ytrain,ytest=train_test_split(xdata,ydata,stratify=ydata,test_size=0.2,random_state=0)
        self.xdataarray=xtrain
        self.ydataarray=ytrain    
        self.xtestarray=xtest
        self.ytestarray=ytest
        #sss.split(xdata,ydata)
        #print('self.ydataarray',self.ydataarray,type(self.ydataarray))
        
        modeldict_data_std_tup=([],[i for i in floatselecttup])
        
        self.genpiscesbatchbatchlist(ytrain,xtrain,batch_n,batchcount,min_y=self.min_y)
        self.genpisces_test_batchbatchlist(ytest,xtest,batch_n,self.batchcount)
        return
        
        
    def build_sumstats_dict(self,ydata,xdata):
        batchbatchcount=min(self.batchbatchcount,self.max_maxbatchbatchcount)
        batchcount=self.batchcount
        batch_n=self.batch_n
        max_train_n=batchbatchcount*batchcount*batch_n
        try:
            existing_sum_stats_dict=self.initial_datagen_dict['summary_stats_dict']
            old_train_n=existing_sum_stats_dict['max_train_n']
            if old_train_n>=max_train_n:
                self.summary_stats_dict=existing_sum_stats_dict
                return
            else:
                self.logger.warning(f'build_sumstats_dict recalculating. old_train_n:{old_train_n}, self.summary_stats_dict:{self.summary_stats_dict}')
        except KeyError:
            pass
        except:
            self.logger.info(f'build sumstats dict building',exc_info=True)        
        xdata=xdata[0:max_train_n,:]
        ydata=ydata[0:max_train_n]
        self.xmean=np.mean(xdata,axis=0)
        self.ymean=np.mean(ydata,axis=0)
        self.xstd=np.std(xdata,axis=0)
        self.ystd=np.std(ydata,axis=0)
        self.summary_stats_dict={'xmean':self.xmean,
                                'ymean':self.ymean,
                                'ystd':self.ystd,
                                'xstd':self.xstd,
                                'max_train_n':max_train_n}
        self.expand_datagen_dict('summary_stats_dict',self.summary_stats_dict)
        
    def processmissingvalues(self,nparray,missing_treatment):
        #rewrite with fancy indexing
        outlist=[]
        if missing_treatment=='drop_row':
            for row in nparray:
                keep=1
                for val in row:
                    if val=='999999' or val=='':keep=0
                if keep==1:outlist.append(row)
        return np.array(outlist)
                
                
        
        
    def genpiscesbatchbatchlist(self, ydataarray,xdataarray,batch_n,batchcount,min_y=1):
        # test data already removed
        n=ydataarray.shape[0]; p=xdataarray.shape[1]
        ycount=int(ydataarray.sum())
        if ycount/n<0.5:
            order=1 # if we're worried about not enough 1's
        else:
            order=-1
            ycount=n-ycount # if we're worried about not enough 0's
        splits=-(-n//batch_n)
        #self.logger.warning(f'splits:{splits}, n:{n}, batch_n:{batch_n},ycount:{ycount},order:{order}')
        if not min_y is None:
            if min_y<1:
                min_y=int(batch_n*min_y)
            
            if ycount<splits*min_y:
                splits= -(-ycount//min_y)
                #self.logger.warning(f'splits:{splits},ycount:{ycount},min_y:{min_y},int()-(-ycount//min_y):{-(-int(ycount)//int(min_y))},type:ycount:{type(ycount)},type:min_y:{type(min_y)}')
            if batchcount>splits:
                batchcount=splits
        else:
            sortidx=np.arange(batch_n)
        #print('splits',splits,'batchcount',batchcount)
        batchbatchcount=self.max_maxbatchbatchcount
        self.logger.warning(f'batchcount:{batchcount},batchbatchcount:{batchbatchcount}')
        batchbatchlist=[[None for b in range(batchcount)] for _ in range(batchbatchcount)]
        RSKF=RepeatedStratifiedKFold(n_splits=splits,n_repeats=batchbatchcount)
        i=0;j=0
        for train_index,test_index in RSKF.split(xdataarray,ydataarray):
            ydata_ij=ydataarray[test_index]
            if not min_y is None:
                sortidx=np.argsort(ydata_ij)[::-order][:batch_n] # order is 1 if we're woried about including enough 1's 
            batchbatchlist[i][j]=(ydata_ij[sortidx],xdataarray[test_index[sortidx],:])
            j+=1
            if j==batchcount:
                i+=1;j=0
                if i==batchbatchcount:
                    break
        batchsize=batch_n*batchcount
        self.batchcount=batchcount
        self.expand_datagen_dict('batchcount',self.batchcount)
        self.batchbatchcount=batchbatchcount
        self.expand_datagen_dict('batchbatchcount',self.batchbatchcount)
        fullbatchbatch_n=batchbatchcount*batchsize
        self.fullbatchbatch_n=fullbatchbatch_n
        self.expand_datagen_dict('fullbatchbatch_n',self.fullbatchbatch_n)
        self.yxtup_batchbatch=batchbatchlist
        
    def genpisces_test_batchbatchlist(self,y,x,batch_n,batchcount):
        n=y.size
        batchbatchcount=-(-n//(batch_n*batchcount))
        batchbatchlist=[[None for b in range(batchcount)] for _ in range(batchbatchcount)]
        idx=0
        for i in range(batchbatchcount):
            for j in range(batchcount):
                start=idx
                stop=idx+batch_n
                if stop>n:stop=n
                batchbatchlist[i][j]=(y[start:stop],x[start:stop,:])
        self.yxtup_batchbatch_test=batchbatchlist
                
            
        
        """
        n=ydataarray.shape[0]; p=xdataarray.shape[1]
        selectlist=[i for i in range(n)]
        random.shuffle(selectlist)
        #print('selectlist',selectlist)
        batchsize=batch_n*batchcount
        if sample_replace:
            batchbatchcount=-(-n//batchsize) # ceiling divide
            assert False,'not developed/broken due to training/validation split'
        else:
            batchbatchcount=n//batchsize # floor divide
        if self.max_maxbatchbatchcount:
            if self.max_maxbatchbatchcount<batchbatchcount:
                batchbatchcount_train=self.max_maxbatchbatchcount
                batchbatchcount_val=batchbatchcount-batchbatchcount_train
            elif batchbatchcount>1:
                self.max_maxbatchbatchcount=batchbatchcount-1
                self.expand_datagen_dict('max_maxbatchbatchcount',self.max_maxbatchbatchcount)
                batchbatchcount_train=self.max_maxbatchbatchcount
                batchbatchcount_val=batchbatchcount-batchbatchcount_train
            else:
                self.max_maxbatchbatchcount=1
                self.expand_datagen_dict('max_maxbatchbatchcount',1)
                batchbatchcount_train=1
                batchbatchcount_val=0
                self.logger.warning(f'without restructuring, cannot validate self.species:{self.species} due to batchbatchcount:{batchbatchcount}')
        self.batchbatchcount=batchbatchcount_train
        self.expand_datagen_dict('batchbatchcount',self.batchbatchcount)
        fullbatchbatch_n_train=batchbatchcount_train*batchsize
        fullbatchbatch_n=batchbatchcount*batchsize
        self.fullbatchbatch_n=fullbatchbatch_n_train
        self.expand_datagen_dict('fullbatchbatch_n',self.fullbatchbatch_n)"""

        '''fullbatchbatch_shortby=fullbatchbatch_n-n
        self.fullbatchbatch_shortby=fullbatchbatch_shortby
        
        if fullbatchbatch_shortby>0:
            selectfill=selectlist.copy()#fill in the missing values with random observations from the list.
            random.shuffle(selectfill)
            selectlist=selectlist+selectfill[:fullbatchbatch_shortby]
        #assert len(selectlist)==fullbatchbatch_n'''
        
        """batchbatchlist=[[[] for b in range(batchcount)] for _ in range(batchbatchcount)]
        for i in range(batchbatchcount):
            for j in range(batchcount):
                start=(i*batchsize)+(j)*batch_n
                end=start+batch_n
                selectionlist=selectlist[start:end]
                ydataarrayselect=ydataarray[selectionlist]
                xdataarrayselect=xdataarray[selectionlist,:]
                batchbatchlist[i][j]=(ydataarrayselect,xdataarrayselect)
                #print('ydatashape:',batchbatchlist[i][j][0].shape,'xdatashape:',batchbatchlist[i][j][1].shape)
        #print('end',end,'fullbatchbatch_n',fullbatchbatch_n)
        
        yxtup_batchbatch=batchbatchlist[:batchbatchcount_train]
        yxtup_batchbatch_val=batchbatchlist[batchbatchcount_train:batchbatchcount_train+batchbatchcount_val]
        if self.sample_y:
            yxtup_batchbatch=self.doSampleY(yxtup_batchbatch)
            #yxtup_batchbatch_val=self.doSampleY(yxtup_batchbatch_val)
        self.yxtup_batchbatch=yxtup_batchbatch
        self.yxtup_batchbatch_val=yxtup_batchbatch_val
        '''all_y=[ii for i in yxtup_list for ii in i[0]]
        all_x=[ii for i in yxtup_list for ii in i[1]]
        '''
        return"""
    '''def doSampleY(self,yxtup_batchbatch):
        bb_count=len(yxtup_batchbatch)
        y_idx_1tracker=[[] for _ in range(bb_count)]
        bb_b_1counter=[[] for _ in range bb_count] # batchbatch-batch
        if bb_count>1:
            for i in range(bb_count):
                for j in len(yxtup_batchbatch[i]):
                    y_idx_1tracker[i].append([])
                    y=yxtup_batchbatch[i][j][0]
                    bb_b_1counter[i].append(y.sum)
                    y_idx_1tracker[i][j]=np.argsort(y.sum)
                '''
            
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
