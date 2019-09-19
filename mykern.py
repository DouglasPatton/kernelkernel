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
        self.xmean=np.mean(xdata,axis=0)
        self.ymean=np.mean(ydata,axis=0)
        self.xstd=np.std(xdata,axis=0)
        self.ystd=np.std(ydata,axis=0)
        xdata_std=(xdata-self.xmean)/self.xstd
        ydata_std=(ydata-self.ymean)/self.ystd

        #First extract optimization information, including modeling information in model dict,
        #param structure in model dict, and param values in paramdict
        modeldict=optimizedict['model_dict'] #reserve _dict for names of dicts in *keys* of parent dicts
        model_param_formdict=modeldict['hyper_param_form_dict']
        kerngrid=modeldict['kern_grid']
        max_Ndiff=modeldict['max_Ndiff']
        
        param_valdict=optimizedict['hyper_param_dict']
        method=optimize_dict['method']
        free_paramlist=param_valdict['p_bandwidth']

        self.n,self.p=xdata.shape

        #for small data, pre-create the 'grid'/out data  and Ndiffs
        if type(kerngrid) is int:
            xout=MY_KDE_gridprep_smalln(kerngrid,self.p)
            assert xout.shape[1]==self.p,'xout has wrong number of columns'
            assert xout.shape[0]==kerngrid**self.p,'xout has wrong number of rows'

            xyout=MY_KDE_gridprep_smalln(kerngrid,self.p+1)
            assert xyout.shape[1]==self.p+1,'xyout has wrong number of columns'
            assert xyout.shape[0]==kerngrid**(self.p+1),'xyout has wrong number of rows'
            self.outgrid='yes'
        if kerngrid=='no'
            xout=xdata_std;
            xyout=np.concatenate(xdata_std,ydata_std,axis=1)
            self.outgrid='no'

        #for small data pre-build multi dimensional differences and masks and apply them.
        
        xstack=xdata_std
        xoutstack=xout
        Ndifflist=[]
        for ii in range(max_Ndiff):
            Ndifflist.append(makediffmat_itoj(xstack,xoutstack))
            xstack=np.repeat(xstack,[:,None]
                                             
                                           
                                        
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
           
        args_tuple=(ydata_std,xdata_std,xgrid,ygrid,,self.onediffs,modeldict)#include N*N array, theonediffs since it will be used so many times.
                
        return minimize(MY_KDEregMSE,free_paramlist,args=args_tuple,method=method) 


    def makediffmat_itoj(self,xi,xj):
        #return xi[:,None,:]-xj[None,:,:] #replaced with more flexible version
        #below code needs to keep i at first dimension, j,k,.... at nearly last dimension, p at last
        r=xi.ndim
        return np.expand_dims(xi,r)-np.expand_dims(xj,r-1) #check this xj-xi where j varies for each i
            
    def make_masks_smalln(self,max_Ndiff):


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
        prob_yx=doYX_KDEsmalln(yin,xin,xin,ybw,xbw,modeldict)#joint density of y and all of x's
        prox_x=doX_KDEsmalln(xin,xin,xbw,,modeldict)
        
    def MY_KDE_gridprep_smalln(self,n,p,kern_grid):
        agrid=np.linspace(-3,3,n) #assuming variables have been standardized
            for idx in range(-1)#-1 b/c agrid already created; need to figure out how to better vectorize this loop
                agrid=np.concatenate(np.repeat(agrid,n,axis=0),np.repeat(np.linspace(-3,3,n)[:,None],n**(idx+1),axis=0),axis=1)
                #assertions added to check shape of output
        return agrid
        
            

            
                          

        

    

    def make_Ndiff_bign(self,x,Ndiff_list,Ndiff_exp,Ndiff_kern,simple_h)
        """takes data and returns multidimensional differenced bandwidths
        Ndiff_list is a list of differences to include in bandwidths"""
        return
        


