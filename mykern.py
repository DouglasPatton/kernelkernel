import numpy as np
from scipy.optimize import minimize

class kNdtool():
    """kNd refers to the fact that there will be kernels in kernels in these estimators

    """

    def xBWmaker(max_bw_Ndiff,self.Ndiff_masklist,onediffs,Ndiff_exponent,Ndiff_bw_kern,normalization):
        """returns an nout X nin np.array of bandwidths
        """
        for depth in range(max_bw_Ndiff,0,-1)#start at deepest Ndiff and work to front
            #axis=depth+1 b/c we want to sum over the last (rbf kern) or 2nd to last (product kern). As can be seen from the
            #tupe construction algorithm in max_Ndiff_datastacker(), there are the first two dimensions that are from the
            #original Ndiff, which is NoutXNin. Then there is a dimension added *depth* times and the last one is what we are
            #collapsing with np.ma.sum. 
            n_depth_mean=np.ma.sum(np.ma.array(max_Ndiff_datastacker(Ndiffs,depth),mask=self.Ndiff_masklist[depth]),axis=depth+1)
                if normalization=='own_n':

                if normalization=='across':
                    np.ma.sum(
            
        
        #normalization options can be implemented after each sum. two obvious options
        #are to divide by the sum across the same level or divide by the number of observations at the same level.
        #perhaps an extension could be to normalize by sums at other levels.


        
    def max_Ndiff_datastacker(self,Ndiffs,depth):
        """After working on two other approaches, I think this approach to replicating the differences with views via
        np.broadcast_to and then masking them using the pre-optimization-start-built lists of progressively deeper masks
        (though it may prove more effective not do have masks of each depth pre-built),
        """
        #prepare tuple indicating shape to broadcast to
        Ndiff_shape=Ndiffs.shape()
        Ndiff_shape_out_tup=Ndiff_shape[0:1]+(Ndiff_shape[1],)*depth#these are tupples, so read as python not numpy
        if len(Ndiff_shape==3):#if parameter dimension hasn't been collapsed yet, keep it at end
            Ndiff_shape_out_tup=Ndiff_shape_out_tup+Ndiff_shape[3]
        return np.brodcast_to(Ndiffs,Ndiff_shape_out_tup)#the tupples tells us how to broadcast nin times over <depth> dimensions          
    
        def max_bw_Ndiff_maskstacker(self,nout,nin,p,max_bw_Ndiff,modeldict_):
        "match the parameter structure of Ndifflist produced by max_bw_Ndiff_datastacker
        ninmask=np.repeat(np.eye(nin)[:,:,None],p,axis=2)
        #change p to 1 if using Ndiff_bw_kern==rbfkern because parameters will be collapsed before mask is applied
        if Ndiff_bw_kern==rbfkern:p=1;
        if self.outgrid=='no':
            masklist=[ninmask]
        if self.outgrid=='yes':
            masklist=[np.zero([nout,nin,p])]
        
        for ii in range(max_bw_Ndiff-1)+1:#since the first mask has already been computed
            maskpartlist=[]
            for iii in range(2,ii+2,1):#does this need to start at 2?
                #use broacast_to instead of repeat?
                #when i insert a new dimension, I'm effectively transposing
                #but not sure if this is quite right. Do I need to broadcast to
                maskpartlist.append(np.repeat(np.expand_dim(ninmask,iii),nin,axis=iii))
                ninmask=np.repeat(np.expand_dim(ninmask,ninmask.dim),nin,axis=ninmask.dim)#need to think about this carefully
            masklist.append[np.ma.mask_or(maskpartlist)#syntax to merge masks
        return masklist                    
        

    
        
    def optimize_hyper_params(self,ydata,xdata,optimizedict):
        """This is the method for iteratively running kernelkernel to optimize hyper parameters
        optimize dict contains starting values for free parameters, hyper-parameter structure(not flexible),
        and a model dict that describes which model to run including how hyper-parameters enter (partiall flexible)
        speed and memory usage is a big goal when writing this. I pre-created masks to exclude the increasing
        list of centered data points. see mykern_core for an example and explanation of dictionaries.
        Flexibility is also a goal. max_bw_Ndiff is the deepest the model goes.
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
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        method=optimize_dict['method']
        param_valdict=optimizedict['hyper_param_dict']
        
        #if model_param_formdict['p_bandwidth']
        free_params=param_valdict['p_bandwidth']#not flexible yet, add exponents later
        free_params=np.concatenate(param_valdict['all_x_bandwidth'],free_params,axis=0)#two more parameters for x and y
        free_params=np.concatenate(param_valdict['all_y_bandwidth'],free_params,axis=0)                            

        Ndiff_exponent=param_valdict['Ndiff_exponent']#fixed right now
        if model_param_formdict['Ndiff_exponent']=='fixed':                 
            fixed_paramdict={'Ndiff_exponent':Ndiff_exponent}#not flexible yet                   
        
        #prep out data as grid (over -3,3) or the original dataset
        xout,yxout=prep_out_grid(kerngrid,xdata_std,ydata_std)
        self.xin=xdata_std,self.yin=ydata_std
        self.xout=xout;self.yxout=yxout
                    
        #for small data pre-build lists of multi dimensional differences and masks and masks to differences.
        #self.Ndifflist=max_bw_Ndiff_datastacker(xdata_std,xout,max_bw_Ndiff) 
        self.Ndiff_masklist=max_bw_Ndiff_maskstacker(self,nout,nin,max_bw_Ndiff)#do I need to save yxin?
        #self.Ndiff=makediffmat_itoj(xout,xdata_std)#xout is already standardized; doing this inside optimization now
                                           
                                        
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


            

    
    def MY_KDEregMSE(self,hyper_params,yxin,yxout,xin,xout,modeldict,fixed_paramdict):
        """moves hyper_params to first position of the obj function, preps data, and then runs MY_KDEreg to fit the model
            then returns MSE of the fit 
        Assumes last p elements of hyper_params are the scale parameters for 'el two' approach to
        columns of x.
        """
        print('starting optimization of hyperparameters')
        #is the masking approach sufficient for leave one out cross validation?
        #kern_grid='no' forces masking of self for predicting self
        p=xin.shape[1]
        if modeldict['Ndiff_bw_kern']=='rbfkern':
            xin_scaled=np.product(xin,hyper_params[:-p])#assuming the last p items of the hyper_params array are the scale parameters
            xout_scaled=np.product(xout,hyper_params[:-p])
            yxin_scaled=np.product(yxin,np.concatenate((np.array([1]),hyper_params[:-p]),axis=0))#insert array of 1's to avoid scaling y
            yxout_scaled=np.product(yxout,np.concatenate((np.array([1]),hyper_params[:-p]),axis=0))
            onediffs_scaled_l2norm=np.sum(np.power(np.product(makediffmat_itoj(xout_scaled,xin_scaled),hyper_params[:-p]),2,axis=p
            assert onediffs_scaled_l2norm.shape==(xout.shape[0],xin.shape[0]),'onediffs_scaled_l2norm does not have shape=(nout,nin)'
            #predict
            yhat=MY_KDEreg(yxin_scaled,yxout_scaled,xin_scaled,xout_scaled,hyper_params,onediffs_scaled_l2norm,modeldict,fixed_paramdict)
            
        
                            
        if modeldict['Ndiff_bw_kern']=='product':
            onediffs=makediffmat_itoj(xout,xin)#scale now?
            #predict
            yhat=MY_KDEreg(yxin,yxout,xin,xout,hyper_params,onediffs,modeldict)
        
                            

        #here is the simple MSE objective function. however, I think I need to use
        #the more sophisticated MISE or mean integrated squared error,
        #either way need to replace with cost function function
        y_err=yin-yhat
        return np.sum(np.power(y_err,2)
        
    
    def MY_KDEreg(self,yxin,yxout,xin,xout,hyper_params,onediffs,modeldict,fixed_paramdict):
        """returns predited values of y for xpredict based on yin, xin, and modeldict
        """
                                                   
        #prepare the Ndiff bandwidth weights
        if modeldict['hyper_param_form_dict']['Ndiff_exponent']=='fixed':
            Ndiff_exponent=fixed_paramdict['Ndiff_exponent']
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        Ndiff_bw_kern=modeldict['Ndiff_bw_kern']
        normalization=modeldict['normalize_Ndiffwtsum']                                                   
        xBWmaker(
            max_bw_Ndiff,self.Ndiff_masklist,onediffs,
            Ndiff_exponent,Ndiff_bw_kern,normalization
            )
                            
        prob_yx=doYX_KDEsmalln(yin,xin,xin,ybw,xbw,modeldict)#joint density of y and all of x's
        prox_x=doX_KDEsmalln(xin,xin,xbw,,modeldict)

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

        
    def MY_KDE_gridprep_smalln(self,n,p,kern_grid):
        """creates a grid with all possible combinations of n evenly spaced values from -3 to 3.
        
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


                             
