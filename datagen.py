
import numpy as np

class datagen():
    '''generates numpy arrays of random training or validation for model: y=xb+e or variants
    '''
    #def __init__(self, data_shape=(200,5), ftype='linear', xval_size='same', sparsity=0, xvar=1, xmean=0, evar=1, betamax=10):
    def __init__(self,source=None, seed=None,ftype=None,evar=None,batch_n=None,param_count=None,batchcount=None,validate_batchcount=None):
        if param_count==None:
            param_count=1
        self.param_count=param_count
        self.p = param_count
        if batch_n==None:
            batch_n=45
        self.batch_n = batch_n
        if batchcount==None:
            batchcount=1
        self.batchcount=batchcount
        if validate_batchcount==None:
            validate_batchcount=batchcount
        self.validate_batchcount=validate_batchcount
        self.source = source
        if source==None or source=='monte':
            self.gen_montecarlo(seed=seed,ftype=ftype,evar=evar,batch_n=batch_n,param_count=param_count,batchcount=batchcount)
            
    def gen_montecarlo(self,seed=None,ftype=None,evar=None,batch_n=None,param_count=None,batchcount=None,validate_batchcount=None):
        if ftype==None:
            ftype='linear'
        if batch_n==None:
            batch_n=45
        if param_count==None:
            param_count=1
        if batchcount==None:
            batchcount=1
        if not seed==None:
            np.random.seed(seed)
        if validate_batchcount==None:
            validate_batchcount=batchcount
        
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