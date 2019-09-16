import numpy as np


class k2dtool():
    """k2d refers to the fact that there will be kernels in kernels in these estimators"""
    def __init__(self,data,grid=None,modeldict=none):
        self.x=xdata
        self.n=np.shape(xdata)[0]
        self.ngrid=self.n; if grid~=None: self.ngrid=np.shape(grid)[0]
        self.p=np.shape(xdata)[1]
        self.xdistmat=self.makedistmat_itoj(xdata)
        self.result_tree=climb_model_tree()

    def makedistmat_itoj(self,xi,xj):
        return xi[:,None,:]-xj[None,:,:]
        
        
    def climb_model_tree(self):
        "returns lists of lists of....dict of results"

    def doKDE(self,xin,w,xout=None):
        """estimate the density of xout using data xin.
        if no data are provided for xout, xout is set as xin
        """
        grid='yes';if xout==None: grid='no'; xout=xin;
        nout,pout=xout.shape;nin,pin=xin.shape
        assert pout==pin,'xout and xin have different numbers of parameters'
        assert np.shape(xin)[-1]==len(w),'len(w) does not match length of last dimension of xin'
        if grid=='no': xmask=np.eye(nin, dtype=int)

    def KDEreg(self,xin,xout,modeldict)
    


