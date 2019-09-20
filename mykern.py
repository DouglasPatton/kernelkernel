import numpy as np
from scipy.optimize import minimize

class kNdtool():
    """kNd refers to the fact that there will be kernels in kernels in these estimators

    """

    def max_Ndiff_datastacker(self,xdata_std,xout,max_Ndiff):
        """take 2 arrays expands their last and second to last dimensions,
        respectively, then repeats into that dimension, then takes differences
        between the two and saves those differences to a list.
        """
        xstack=xdata_std
        xoutstack=xout
        nout=xout.shape[0]
        Ndifflist=[]
        for ii in range(max_Ndiff):
            xstack=np.repeat(np.expand_dims(xstack,ii+1),self.nout,axis=ii+1)
            xoutstack=np.repeat(np.expand_dims(xoutstack,ii),self.n,axis=ii)
            Ndifflist.append(xstack-xoutstack)
        return Ndifflist
        
    def optimize_hyper_params(self,ydata,xdata,optimizedict):
        self.n,self.p=xdata.shape
        assert ydata.shape[0]==xdata.shape[0],'x and y do not match in dimension 0'

        #standardize x and y
        xdata_std,ydata_std=standardize_xy(xdata,ydata)
        
        #extract optimization information, including modeling information in model dict,
        #param structure in model dict, and param values in paramdict
        modeldict=optimizedict['model_dict'] #reserve _dict for names of dicts in *keys* of parent dicts
        model_param_formdict=modeldict['hyper_param_form_dict']
        kerngrid=modeldict['kern_grid']
        max_Ndiff=modeldict['max_Ndiff']
        method=optimize_dict['method']
        param_valdict=optimizedict['hyper_param_dict']
        free_paramlist=param_valdict['p_bandwidth']#not flexible yet

        #prep out data as grid or original dataset
        xout,xyout=prep_out_grid(kerngrid,xdata_std,ydata_std)
    
        #for small data pre-build multi dimensional differences and masks and masks to differences.
        Ndifflist=max_Ndiff_datastacker(xdata_std,xout,max_Ndiff)

        
                                           
                                        
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
           

    def prep_out_grid(self,kerngrid,xdata_std,ydata_std):
        #for small data, pre-create the 'grid'/out data  and Ndiffs
        if self.n<10**5 and not (type(kerngrid)==int and kerngrid**self.p>10**8):
            if type(kerngrid) is int:
                self.nout=kerngrid**self.p             
                xout=MY_KDE_gridprep_smalln(kerngrid,self.p)
                assert xout.shape[1]==self.p,'xout has wrong number of columns'
                assert xout.shape[0]==kerngrid**self.p,'xout has wrong number of rows'

                xyout=MY_KDE_gridprep_smalln(kerngrid,self.p+1)
                assert xyout.shape[1]==self.p+1,'xyout has wrong number of columns'
                assert xyout.shape[0]==kerngrid**(self.p+1),'xyout has wrong number of rows'
                self.outgrid='yes'
            if kerngrid=='no'
                self.nout=self.n
                xout=xdata_std;
                xyout=np.concatenate(xdata_std,ydata_std,axis=1)
                self.outgrid='no'
        return xout,xyout              
        
    def standardize_xy(self,xdata,ydata):
        self.xmean=np.mean(xdata,axis=0)
        self.ymean=np.mean(ydata,axis=0)
        self.xstd=np.std(xdata,axis=0)
        self.ystd=np.std(ydata,axis=0)
        return (xdata-self.xmean)/self.xstd,(ydata-self.ymean)/self.ystd


                             
