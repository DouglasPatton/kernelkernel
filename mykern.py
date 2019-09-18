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

    def optimize_hyper_params(self,main_function,ydata,xdata,optimizedict):

        #First extract optimization information, including modeling information in model dict,
        #param structure in model dict, and param values in paramdict
        modeldict=optimizedict['model_dict'] #reserve _dict for names of dicts in keys of parent dicts
        model_param_formdict=modeldict['hyper_param_form_dict']
        kerngrid=modeldict['kern_grid']
        
        param_valdict=optimizedict['hyper_param_dict']
        method=optimize_dict['method']
        free_paramlist=param_valdict['p_bandwidth']
        
        #generate rhs of difference and
        #the N*N matrix of first differences that will be used so often
        self.xgrid,self.onediffs=MY_KDE_dataprep_smalln(xin,kerngrid)

        self.n,self.p=xdata.shape

        #parse args to pass to the main optimization function
  
        #build pos_hyper_param and fixed hyper_param, add more later
        #Below code commented out b/c too complicated to make it so flexible.
        #can build in flexibility later
        '''fixedparams=[]
        non_neg_params=[]
        for key,val in param_valdict:
            if model_param_formdict[key]='fixed':
                fixedparams.append(val)
            if model_param_formdict[key]='fixed':
                non_negparams.append(val)'''
           
        args_tuple=(ydata,xdata,self.xgrid,self.onediffs,modeldict)#include N*N array, theonediffs since it will be used so many times.
                
        return minimize(MY_KDEregMSE,free_paramlist,args=args_tuple,method=method) 

    def doX_KDEsmalln(self,xin,xout,xbw,modeldict):
        """estimate the density of xout using data xin and weights, xbw
        if no data are provided for xgrid, xgrid is set as xin
        return xgrid x 2 array of values and marginal densities.
        doKDE first constructs 
        """
        nout,pout=xout.shape;nin,pin=xin.shape
        assert pout==pin,'xout and xin have different numbers of parameters'
        assert np.shape(xin)[-1]==len(w),'len(w) does not match length of last dimension of xin'

        if grid=='no': xmask=np.eye(nin, dtype=int)
    
    #def doYX_KDEsmalln(self,yin,xin,xout,ybw,xbw,modeldict):

    
    def MY_KDEregMSE(self,hyper_params,yin,xin,xgrid,onediffs,modeldict)
    """moves hyper_params to first position of the obj function and then runs MY_KDEreg to fit the model
    then returns MSE of the fit"""
        print('starting optimization of hyperparameters')
        #is the masking approach sufficient for leave one out cross validation?
        #kern_grid='no' forces masking of self for predicting self
        if modeldict['Kh_form']=='exp_l2'
            xin=np.product(xin,hyper_params[:-p]**-1)
            onediffs=np.product(onediffs,hyper_params[:-p])
        return np.sum((yin-MY_KDEreg(yin,xin,xin,hyper_params,onediffs,modeldict))**2)
        
    
    def MY_KDEreg(self,yin,xin,xpredict,hyper_params,modeldict):
        """returns predited values of y for xpredict based on yin, xin, and modeldict
        """
        X_bw=        
        prob_yx=doYX_KDEsmalln(yin,xin,xin,ybw,xbw,modeldict)#joint density of y and all of x's
        prox_x=doX_KDEsmalln(xin,xin,xbw,,modeldict)
        
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
        


