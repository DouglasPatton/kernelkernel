import numpy as np
from scipy.optimize import minimize

class kNdtool():
    """kNd refers to the fact that there will be kernels in kernels in these estimators

    """
    def xBWmaker(self,max_Ndiff,masklist,Ndiffs,Ndiff_exp_params,free_paramlist,Kh_form):
        if Kh_form=='exp_l2'
        for ii in self.Ndiff_masklist:
            np.broadcast_to(self.Ndiff
            
        
        
    def max_Ndiff_maskstacker(self,nout,nin,p,max_Ndiff):
        "match the parameter structure of Ndifflist produced by max_Ndiff_datastacker
        ninmask=np.repeat(np.eye(nin)[:,:,None],p,axis=2)
        #change p to 1 if using Kh_form==exp_l2
        if self.outgrid=='no':
            masklist=[ninmask]
        if self.outgrid=='yes':
            masklist=[np.zero([nout,nin,p])]
        
        for ii in range(max_Ndiff-1)+1:#since the first mask has already been computed
            maskpartlist=[]
            for iii in range(ii+2):
                maskpartlist.append(np.repeat(np.expand_dim(ninmask,iii),nin,axis=iii))#use broacast_to instead of repeat?
            masklist.append[np.ma.mask_or(maskpartlist)#syntax to merge masks
        return masklist                    
        

    
        
    def optimize_hyper_params(self,ydata,xdata,optimizedict):
        """This is the method for iteratively running kernelkernel to optimize hyper parameters
        optimize dict contains starting values for free parameters, hyper-parameter structure(not working),
        and a model dict that describes which model to run including how hyper-parameters enter (not working)
        speed and memory usage is a big goal when writing this. I pre-created masks to exclude the increasing
        list of centered data points.
        Flexibility is also a goal. max_Ndiff is the deepest the model goes.
        ------------------
        attributes created
        self.n,self.p,self.optdict 
        self.xdata_std, self.xmean,self.xstd
        self.ydata_std,self.ymean,self.ystd
        self.Ndiff - the nout X nin X p matrix of first differences of xdata_std
        self.Ndiff_masklist - a list of progressively higher dimension (len=nin) 
            masks to broadcast(views) Ndiff to.
        
        """
        self.n,self.p=xdata.shape
        self.optdict=optimizedict
        
        assert ydata.shape[0]==xdata.shape[0],'x and y do not match in dimension 0'

        #standardize x and y and save their means and std to self
        xdata_std,ydata_std=standardize_xy(xdata,ydata)
        #store the standardized (by column or parameter,p) versions of x and y
        self.xdata_std=xdata_std;self.ydata_std=ydata_std
        #extract optimization information, including modeling information in model dict,
        #parameter structure in model dict, and starting free parameter values from paramdict
        #these options are still not fully mature and actually flexible
        modeldict=optimizedict['model_dict'] #reserve _dict for names of dicts in *keys* of parent dicts
        model_param_formdict=modeldict['hyper_param_form_dict']
        kerngrid=modeldict['kern_grid']
        max_Ndiff=modeldict['max_Ndiff']
        method=optimize_dict['method']
        param_valdict=optimizedict['hyper_param_dict']
        
        #if model_param_formdict['p_bandwidth']
        free_params=param_valdict['p_bandwidth']#not flexible yet, add exponents later
        free_params=np.concatenate(param_valdict['all_x_bandwidth'],free_params,axis=0)#two more parameters for x and y
        free_params=np.concatenate(param_valdict['all_y_bandwidth'],free_params,axis=0)                            

        Ndiff_exp_params=param_valdict['Ndiff_exp']#fixed right now
        fixed_paramdict={'Ndiff_exp_params':Ndiff_exp_params}#not flexible yet                   
        
        #prep out data as grid (over -3,3) or the original dataset
        xout,yxout=prep_out_grid(kerngrid,xdata_std,ydata_std)
        self.xin=xdata_std,self.yin=ydata_std
        self.xout=xout;self.yxout=yxout
                    
        #for small data pre-build lists of multi dimensional differences and masks and masks to differences.
        #self.Ndifflist=max_Ndiff_datastacker(xdata_std,xout,max_Ndiff) 
        self.Ndiff_masklist=max_Ndiff_maskstacker(self,nout,nin,max_Ndiff)#do I need to save yxin?
        self.Ndiff=makediffmat_itoj(xout,xdata_std)
                                           
                                        
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
        
        
        args_tuple=(yxin,yxout,xin,xout,modeldict,fixed_paramdict)
                
        return minimize(MY_KDEregMSE,free_params,args=args_tuple,method=method) 


    def makediffmat_itoj(self,xi,xj):
        #return xi[:,None,:]-xj[None,:,:] #replaced with more flexible version
        #below code needs to keep i at first dimension, j,k,.... at nearly last dimension, p at last
        r=xi.ndim
        return np.expand_dims(xi,r)-np.expand_dims(xj,r-1) #check this xj-xi where j varies for each i
            
    

    def doX_KDEsmalln(self,xin,xout,xbw,modeldict):
        """estimate the density of xout using data xin and weights, xbw
        if no data are provided for xgrid, xgrid is set as xin
        return xgrid x 2 array of values and marginal densities.
        doKDE first constructs
        Should I replace dictionaries with actual values? 
        """
        nout,pout=xout.shape;nin,pin=xin.shape
        assert pout==pin,'xout and xin have different numbers of parameters'
        assert np.shape(xin)[-1]==len(w),'len(w) does not match length of last dimension of xin'
        
        
    
    #def doYX_KDEsmalln(self,yin,xin,xout,ybw,xbw,modeldict):
        

    
    def MY_KDEregMSE(self,hyper_params,yxin,yxout,xin,xout,modeldict,fixed_paramdict):
        """moves hyper_params to first position of the obj function, preps data, and then runs MY_KDEreg to fit the model
            then returns MSE of the fit 
        Assumes last p elements of hyper_params are the scale parameters for 'el two' approach to
        columns of x.
        """
        print('starting optimization of hyperparameters')
        #is the masking approach sufficient for leave one out cross validation?
        #kern_grid='no' forces masking of self for predicting self
        if modeldict['Kh_form']=='exp_l2':
            xin_scaled=np.product(xin,hyper_params[:-p]**-1)
            xout_scaled=np.product(xout,hyper_params[:-p]**-1)
            yxin_scaled=np.product(yxin,np.concatenate((np.array([1]),hyper_params[:-p]**-1),axis=0))
            yxout_scaled=np.product(yxout,np.concatenate((np.array([1]),hyper_params[:-p]**-1),axis=0))
            onediffs_scaled=np.product(onediffs,hyper_params[:-p])
            yhat=MY_KDEreg(yxin_scaled,yxout_scaled,xin_scaled,xout_scaled,hyper_params,onediffs_scaled,modeldict,fixed_paramdict)
            
        
                            
        if modeldict['Kh_form']=='product':
            yhat=MY_KDEreg(yxin,yxout,xin,xout,hyper_params,onediffs,modeldict)
        
                            

        #here is the simple MSE objective function. however, I think I need to use
        #the more sophisticated MISE or mean integrated squared error
        y_err=yin-yhat
        return np.sum(y_err*y_err)
        
    
    def MY_KDEreg(self,yxin,yxout,xin,xout,hyper_params,onediffs_scaled,modeldict,fixed_paramdict):
        """returns predited values of y for xpredict based on yin, xin, and modeldict
        """
        #prepare the Ndiff bandwidth weights
        Ndiff_exp_params=
        if modeldict['Kh_form']=='exp_l2':
            onediffs_scaled_l2norm=np.sum(onediffs_scaled*onediffs_scaled,axis=p)
            assert onediffs_scaled_l2norm.shape==[nout,nin],'onediffs_scaled_l2norm has the wrong shape'
            xBWmaker(modeldict['max_Ndiff'],self.Ndiff_masklist,onediffs_scaled_l2norm,fixed_paramdict['Ndiff_exp_params'],free_paramlist,Kh_form)
                            
        prob_yx=doYX_KDEsmalln(yin,xin,xin,ybw,xbw,modeldict)#joint density of y and all of x's
        prox_x=doX_KDEsmalln(xin,xin,xbw,,modeldict)
        
    def MY_KDE_gridprep_smalln(self,n,p,kern_grid):
        """creates a grid with all possible combinations of n evenly spaced values from -3 to 3.
        if I am using the "el two" approach, it seems like the grid is just two items wide, y and ||x||
        """
        agrid=np.linspace(-3,3,n) #assuming variables have been standardized
            for idx in range(-1)#-1 b/c agrid already created; need to figure out how to better vectorize this loop
                agrid=np.concatenate(np.repeat(agrid,n,axis=0),np.repeat(np.linspace(-3,3,n)[:,None],n**(idx+1),axis=0),axis=1)
                #assertions added to check shape of output
        return agrid
           

    def prep_out_grid(self,kerngrid,xdata_std,ydata_std):
        #for small data, pre-create the 'grid'/out data 
        if self.n<10**5 and not (type(kerngrid)==int and kerngrid**self.p>10**8):
            self.data_is_small='yes'
            if type(kerngrid) is int:
                self.nout=kerngrid**self.p             
                xout=MY_KDE_gridprep_smalln(kerngrid,self.p)
                assert xout.shape[1]==self.p,'xout has wrong number of columns'
                assert xout.shape[0]==kerngrid**self.p,'xout has wrong number of rows'

                yxout=MY_KDE_gridprep_smalln(kerngrid,self.p+1)
                assert yxout.shape[1]==self.p+1,'yxout has wrong number of columns'
                assert yxout.shape[0]==kerngrid**(self.p+1),'yxout has wrong number of rows'
                self.outgrid='yes'
            if kerngrid=='no'
                self.nout=self.n
                xout=xdata_std;
                yxout=np.concatenate(ydata_std,xdata_std,axis=1)
                self.outgrid='no'
        return xout,yxout              
        
    def standardize_xy(self,xdata,ydata):
        self.xmean=np.mean(xdata,axis=0)
        self.ymean=np.mean(ydata,axis=0)
        self.xstd=np.std(xdata,axis=0)
        self.ystd=np.std(ydata,axis=0)
        return (xdata-self.xmean)/self.xstd,(ydata-self.ymean)/self.ystd


                             
