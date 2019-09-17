import numpy as np


class kNdtool():
    """kNd refers to the fact that there will be kernels in kernels in these estimators
    self.doKDE() returns a joint density estimate for the grid or points entered based on user specified model features

    """
    #delete this init section?
    def __init__(self,data,grid=None,optimize_dict=None):
        self.x=xdata
        self.n=np.shape(xdata)[0]
        #self.ngrid=self.n; if grid~=None: self.ngrid=np.shape(grid)[0]
        self.p=np.shape(xdata)[1]
        if optimize_dict~=None:
            #if n> threshold.... then, .....; else:
            self.xdiffmat=self.makediffmat_itoj(xdata) #pre compute differences for smaller n

    

    def makediffmat_itoj(self,xi,xj):
        #return xi[:,None,:]-xj[None,:,:] #replaced with more flexible version
        return np.expand_dims(xi,1)-np.expand_dims(xj,0)
        
    def MYKDE_smalln(self,xin,modeldict)
        #0 make xgrid based on modeldict:kern_grid or not
        if modeldict['kern_grid']=='no':
            makegrid=0; xgrid=xin
        if modeldict['kern_grid']=='n':
            n=self.n;makegrid=1
        if type(modeldict['kern_grid']) is int:
            n=modeldict['kern_grid'];makegrid=1
        if makegrid==1:
            xgrid=np.linspace(-3,3,n)
            for _ in range(self.p-1)
                xgrid=np.concatenate(np.repeat(xgrid,n,axis=0),np.tile(np.linspace(-3,3,n),n,axis=0),axis=1)

        
            
            
        #1. make the stacks of differences and masks that will be needed based on supplied modelfeatures
        alldiffs=make_ddiff_smalln(xin,xgrid,modeldict['max_ddiff'])
        allmasks=make_masks_smalln(modeldict['max_ddiff'])
                          
        #2. feed data and weights to doKDE, which doesn't know where the data or weights came from.
        
    def make_ddiff_smalln(self,x,max_ddiff):
        """returns an 
        """
    def make_masks_smalln(self,max_ddiff):
            
                          
    def doKDEsmalln(self,xin,w,xgrid=None):
        """estimate the density of xgrid using data xin.
        if no data are provided for xgrid, xgrid is set as xin
        return xgrid x 2 array of values and marginal densities.
        doKDE first constructs 
        """
        grid='yes';if xgrid==None: grid='no'; xgrid=xin;
        ngrid,pgrid=xgrid.shape;nin,pin=xin.shape
        assert pgrid==pin,'xout and xin have different numbers of parameters'
        assert np.shape(xin)[-1]==len(w),'len(w) does not match length of last dimension of xin'

        if grid=='no': xmask=np.eye(nin, dtype=int)


    def 
    def make_ddiff_bign(self,x,ddiff_list,ddiff_exp,ddiff_kern,simple_h)
        """takes data and returns multidimensional differenced bandwidths
        ddiff_list is a list of differences to include in bandwidths"""
        return
        

    

    
    def KDEreg(self,xin,xout,modeldict)
    


