import numpy as np
from scipy.optimize import minimize

class kNdtool():
    """kNd refers to the fact that there will be kernels in kernels in these estimators
    self.doKDE() returns a joint density estimate for the grid or points entered based on user specified model features

    """
    '''#delete this init section?
    def __init__(self,data,grid=None,optimize_dict=None):
        self.x=xdata
        self.n=np.shape(xdata)[0]
        self.p=np.shape(xdata)[1]
    '''

    def optimize_hyper_params(self,main_function,ydata,xdata,optimize_dict):
        #hyper_param_array=pull from hyper_paramdict
        self.xgrid,self.diff=MY_KDE_dataprep_smalln(xin,modeldict['kern_grid'])
        method=optimize_dict['method']

        #parse args to pass to the main optimization function
        #need to make this more flexible based on modeldict:hper_param_form_dict, replace with ordered dict?
        hpdict=optimize_dict['hyper_paramdict'] 
        hyper_paramlist=np.concatenate(hpdict['ddiff_exp_params'],hpdict['x_bandwidth_params'],axis=0)
        args_tuple=(ydata,xdata,modeldict)
                
        return minimize(MY_KDEregMSE,hyper_paramlist,args=args_tuple,method=method) 

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
    

    def MY_KDEregMSE(self,hyper_params,yin,xin,modeldict)
    """moves hyper_params to first position and then runs MY_KDEreg to fit the model
    then returns MSE of the fit"""
        print('starting optimization of hyperparameters')
        print('kern grid=',modeldict['kern_grid'])
        #is the masking approach sufficient for leave one out cross validation?
        #kern_grid='no' forces masking of self for predicting self
        return np.sum((yin-MY_KDEreg(yin,xin,xin,hyper_params,modeldict))**2)
        
    
    def MY_KDEreg(self,yin,xin,xpredict,hyper_params,modeldict):
        """returns predited values of y for xpredict based on yin, xin, and modeldict
        """
        yx=np.concatenate(yin,xin,axis=1)
        prob_yx=doKDEsmalln(yx,w)
        prox_x=doKDEsmalln(
        
    def MY_KDE_dataprep_smalln(self,xin,kern_grid):
        """takes in data and makes differences between observed data and potentially a grid
        returns: xgrid, and a matrix of arrays where the i,j element is xin(i)-xgrid(j) 

        """
        #make xgrid based on modeldict:kern_grid or not
        n=xin.shape[0]
        makegrid=1
        
        if kern_grid=='no':
            makegrid=0;xgrid=xin, 
        #if kern_grid=='n': # not needed b/c default
        if type(kern_grid) is int:
            n=kern_grid
        if makegrid==1:
            xgrid=np.linspace(-3,3,n)
            for idx in range(self.p-1)#-1 b/c xgrid already created; need to figure out how to better vectorize this loop
                xgrid=np.concatenate(np.repeat(xgrid,n,axis=0),np.repeat(np.linspace(-3,3,n)[:,None],n**(idx+1),axis=0),axis=1)#need to check this!
        self.makegrid=makegrid #save to help with masking
        return makediffmat_itoj(xin,xgrid)
        
            

    def makediffmat_itoj(self,xi,xj):
        #return xi[:,None,:]-xj[None,:,:] #replaced with more flexible version
        return np.expand_dims(xi,1)-np.expand_dims(xj,0) #check this
            
    def make_masks_smalln(self,max_ddiff):
            
                          

        

    

    def make_ddiff_bign(self,x,ddiff_list,ddiff_exp,ddiff_kern,simple_h)
        """takes data and returns multidimensional differenced bandwidths
        ddiff_list is a list of differences to include in bandwidths"""
        return
        


