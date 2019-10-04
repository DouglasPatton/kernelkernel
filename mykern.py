import numpy as np
from scipy.optimize import minimize

class kNdtool():
    """kNd refers to the fact that there will be kernels in kernels in these estimators

    """
    def normalize_and_sum_bw(self,kernstack,normalization):
        if normalization=='none':
            return np.ma.sum(kernstack,axis=0)
        
        if normalization=='own_n':
            return np.ma.mean(kernstack,axis=0)
        
        # if normalization=='across': #does this make sense? not working now.
        #    this_depth_not_summed=kernstack
        #   one_deeper_summed=np.ma.sum(do_bw_kern(Ndiff_bw_kern,np.ma.array(Ndiff_datastacker(Ndiffs,depth+1,Ndiff_bw_kern),mask=self.Ndiff_masklist[depth+1])),axis=0)
        #  n_depth_total=np.ma.sum(np.ma.divide(this_depth_not_summed,one_deeper_summed),axis=0)
    def xBWmaker(self,max_bw_Ndiff,self.Ndiff_masklist,diffdict,Ndiff_exponent_params,p_bandwidth_params,Ndiff_bw_kern,normalization=None):
        """returns an nout X nin np.array of bandwidths
        """

        #for loop starts at deepest Ndiff and works to front
    #
        #axis=depth+1 b/c we want to sum over the last (rbf kern) or 2nd to last (product kern). As can be seen from the
        #tup construction algorithm in Ndiff_datastacker(), there are the first two dimensions that are from the
        #original Ndiff, which is NoutXNin. Then there is a dimension added *depth* times and the last one is what we are
        #collapsing with np.ma.sum.
        Ndiffs=diffdict['Ndiffs']
        if Ndiff_bw_kern=='rbfkern': #parameter column already collapsed
            
            for depth in range(max_bw_Ndiff,0,-1):#dpeth starts wtih the last mask first
                this_depth_ma_Ndiffstack=np.ma.array(Ndiff_datastacker(Ndiffs,depth,Ndiff_bw_kern),mask=self.Ndiff_masklist[depth])
                if depth==max_bw_Ndiff:
                    the_bw=normalize_and_sum_bw(do_bw_kern(Ndiff_bw_kern,this_depth_ma_Ndiffstack)),normalization)
                else:
                    the_bw=np.ma.multiply(the_bw,np.ma.power(this_depth_ma_Ndiffstack,Ndiff_exponent_params[depth]))
                
                n_depth_total=np.ma.power(n_depth_total,Ndiff_exponent_params[depth])
                
                
                else:
                    the_bw=np.ma.multiply(n_depth_total,the_bw)
        if Ndiff_bw_kern=='product': #onediffs parameter column not yet collapsed
            n_depth_masked_sum_kern=do_bw_kern(Ndiff_bw_kern,n_depth_masked_sum,p_bandwidth_params)
        
        #normalization options can be implemented after each sum. two obvious options
        #are to divide by the sum across the same level or divide by the number of observations at the same level.
        #perhaps an extension could be to normalize by sums at other levels.

    def do_bw_kern(self,kern_choice,maskeddata,p_bandwidth_params=None):
        if kern_choice=="product":
            return np.ma.product(p_bandwidth_params,np.ma.exp(-np.ma.power(maskeddata,2)),axis=maskeddata.ndim-1)
            #axis-1 b/c axes counted from zero but ndim counts from 1
        if kern_choice=='rbfkern':
            return np.ma.exp(-np.ma.power(maskeddata,2))
            
        
    def Ndiff_datastacker(self,Ndiffs,depth,Ndiff_bw_kern):
        """After working on two other approaches, I think this approach to replicating the differences with views via
        np.broadcast_to and then masking them using the pre-optimization-start-built lists of progressively deeper masks
        (though it may prove more effective not do have masks of each depth pre-built)
        """
        #prepare tuple indicating shape to broadcast to
        
        Ndiff_shape=Ndiffs.shape()
        if Ndiff_bw_kern=='rbfkern':
            assert Ndiff_shape==(self.nin,self.nin),"Ndiff shape not nin X nin but bwkern is rbfkern"
        if Ndiff_bw_kern=='product':
            assert Ndiff_shape==(self.nin,self.nin,self.p),"Ndiff shape not nin X nin X p but bwkern is product"
        
        Ndiff_shape_out_tup=(Ndiff_shape[1])*depth+(Ndiff_shape[0])#these are tupples, so read as python not numpy
       if Ndiff_bw_kern=='product':#if parameter dimension hasn't been collapsed yet, 
            Ndiff_shape_out_tup=Ndiff_shape_out_tup+Ndiff_shape[2]#then add it at the end
        return np.broadcast_to(Ndiffs,Ndiff_shape_out_tup)#the tupples tells us how to
            #broadcast nin times over <depth> dimensions added to the left side of np.shape()        
    
    def max_bw_Ndiff_maskstacker(self,nout,nin,p,max_bw_Ndiff,modeldict):
        '''match the parameter structure of Ndifflist produced by Ndiff_datastacker
        notably, mostly differences (and thus masks) will be between the nin (n in the original dataset) obeservations.
        though would be interesting to make this more flexible in the future.
        when i insert a new dimension between two dimensions, I'm effectively transposing
        '''
        ninmask=np.repeat(np.eye(nin)[:,:,None],p,axis=2)#this one will be used extensively to construct masks
        #change p to 1 if using Ndiff_bw_kern==rbfkern because parameters will be collapsed before mask is applied
        if Ndiff_bw_kern=='rbfkern':p=1
        if self.outgrid=='no':
            masklist=[ninmask]
        if self.outgrid=='yes':
            masklist=[np.zero([nin,nout,p])]#reindexed to lkjip
        for ii in range(max_bw_Ndiff-1):#-1since the first mask is in masklist
            basemask=masklist[-1]#copy the last item to basemask
            masklist.append[np.repeat(np.expand_dim(basemask,0),nin,axis=0)]#then use basemask to
            for iii in range(1,ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                #then this loop maxes at 0,1,2from range(1+2)

                #take the last item we're constructing and merge it with another mask
                masklist[-1]=np.ma.mask_or(masklist[-1],np.repeat(np.expand_dim(basemask,iii),nin,axis=iii))
                #reindex:ninmask=np.repeat(np.expand_dim(ninmask,ninmask.dim),nin,axis=ninmask.dim)
            #masklist.append(np.ma.mask_or(maskpartlist))#syntax to merge masks

        return masklist
                            
    def pull_value_from_fixed_or_free(self,free_params,fixed_params,param_name,fixed_or_free_paramdict):
        start,end=fixed_or_free_paramdict[param_name]['location_idx']
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='fixed':
            the_param_values=fixed_params[start:end]#end already includes +1 to make range inclusive of the end value
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='free':
            the_param_values=free_params[start:end]#end already includes +1 to make range inclusive of the end value 
        return the_param_values

    def sort_fixed_or_free(self,model_param_formdict,param_valdict):
        '''takes a dictionary specifying fixed or free and a dictionary specifying starting (if free) or
        fixed (if fixed) values
        returns 2 lists and a dict
            free_params 1 dim np array of the starting parameter values in order
            fixed_params 1 dim nop array of the fixed parameter values in order
            ffixed_or_free_paramdict a dictionary for each parameter (param_name) with the following key:val
                fixed_or_free:'fixed' or 'free'
                location_idx: the start and end location for parameters of param_name in the appropriate array
        '''
        fixed_params=[];free_params=[];fixed_or_free_paramdict={}
        #build fixed and free vectors of hyper-parameters based on hyper_param_formdict
        for param_name,param_form in model_param_formdict.items():
            param_feature_dict={}
            param_val=param_val_dict[param_name]
            assert param_val.ndim==1,"values for {} have not ndim==1".format(param_name)
            if param_form=='fixed':
                param_feature_dict['fixed_or_free']=='fixed'
                param_feature_dict['location_idx']=(len(fixed_params),len(fixed_params)+len(param_val))
                    #start and end indices, with end already including +1 to make python slicing inclusive of end in start:end
                np.concatenate([fixed_params,param_val],axis=0)
                fixed_or_free_paramdict[param_name]=param_feature_dict
            if param_form=='free':
                param_feature_dict['fixed_or_free']=='free'
                param_feature_dict['location_idx']=(len(free_params),len(free_params)+len(param_val))
                    #start and end indices, with end already including +1 to make python slicing inclusive of end in start:end
                np.concatenate([free_params,param_val],axis=0)
                fixed_or_free_paramdict[param_name]=param_feature_dict
        return free_params,fixed_params,fixed_or_free_paramdict
        
    def optimize_free_params(self,ydata,xdata,optimizedict):
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

        #create lists of variables and dictionary for keeping track of them
        free_params,fixed_params,fixed_or_free_paramdict=sort_fixed_or_free(model_param_formdict,param_valdict)
             
        #--------------------------------
        #prep out data as grid (over -3,3) or the original dataset
        xout,yxout=prep_out_grid(kerngrid,xdata_std,ydata_std)
        self.xin=xdata_std,self.yin=ydata_std
        self.xout=xout;self.yxout=yxout
                    
        #for small data pre-build lists of multi dimensional differences and masks and masks to differences.
        #self.Ndifflist=max_bw_Ndiff_datastacker(xdata_std,xout,max_bw_Ndiff) 
        self.Ndiff_masklist=max_bw_Ndiff_maskstacker(self,nout,nin,max_bw_Ndiff)#do I need to save yxin?
        #self.Ndiff=makediffmat_itoj(xout,xdata_std)#xout is already standardized; doing this inside optimization now
                                           
        args_tuple=(fixed_params,yxin,yxout,xin,xout,modeldict,fixed_or_free_paramdict)
        return minimize(MY_KDEregMSE,free_params,args=args_tuple,method=method)

    
    def MY_KDEregMSE(self,free_params,fixed_params,yxin,yxout,xin,xout,modeldict,fixed_or_free_paramdict):
        """moves free_params to first position of the obj function, preps data, and then runs MY_KDEreg to fit the model
            then returns MSE of the fit 
        Assumes last p elements of free_params are the scale parameters for 'el two' approach to
        columns of x.
        """
        print('starting optimization of hyperparameters')
        #is the masking approach sufficient for leave one out cross validation?
        #kern_grid='no' forces masking of self for predicting self

        #pull p_bandwidth parameters from the appropriate location and appropriate vector
        p_bandwidth_params=self.pull_value_from_fixed_or_free(free_params,fixed_params,'p_bandwidth',fixed_or_free_paramdict)
        p=xin.shape[1]
        assert p==len(p_bandwidth_params),"the wrong number of p_bandwidth_params exist"


        if modeldict['Ndiff_bw_kern']=='rbfkern':
            
            xin_scaled=xin*p_bandwidth_params#assuming the last p items of the free_params array are the scale parameters
            xout_scaled=xout*p_bandwidth_params
            yxin_scaled=yxin*np.concatenate([np.array([1]),p_bandwidth_params],axis=0))#insert array of 1's to avoid scaling y. need to think about whether this is correc
            yxout_scaled=yxout*np.concatenate([np.array([1]),p_bandwidth_params],axis=0))#or should I scale y?
            onediffs_scaled_l2norm=np.power(np.sum(np.power(makediffmat_itoj(xout_scaled,xin_scaled),2),axis=p),.5)
            if modeldict['kerngrid']=='no:
                'Ndiffs_scaled_l2norm=onediffs_scaled_l2norm
            else:
                Ndiffs_scaled_l2norm=np.power(np.sum(np.power(makediffmat_itoj(xin_scaled,xin_scaled),2),axis=p),.5)
                    #used for higher differences when kerngrid=yes
            assert onediffs_scaled_l2norm.shape==(xout.shape[0],xin.shape[0]),'onediffs_scaled_l2norm does not have shape=(nout,nin)'
            #predict
            diffdict={}
            diffdict['onediffs']=onediffs_scaled_l2norm
            diffdict['Ndiffs']=Ndiffs_scaled_l2norm
            yhat=MY_KDEreg(yxin_scaled,yxout_scaled,xin_scaled,xout_scaled,free_params,fixed_params,diffdict,modeldict,fixed_or_free_paramdict)
            
        
                            
        if modeldict['Ndiff_bw_kern']=='product':
            onediffs=makediffmat_itoj(xout,xin)#scale now? if so, move if...='rbfkern' down 
            #predict
            yhat=MY_KDEreg(yxin,yxout,xin,xout,free_params,fixed_params,onediffs,modeldict)
        
                            

        #here is the simple MSE objective function. however, I think I need to use
        #the more sophisticated MISE or mean integrated squared error,
        #either way need to replace with cost function function
        y_err=yin-yhat
        return np.sum(np.power(y_err,2)
        
    
    def MY_KDEreg(self,yxin,yxout,xin,xout,free_params,fixed_params,diffdict,modeldict,fixed_or_free_paramdict):
        """returns predited values of y for xpredict based on yin, xin, and modeldict
        """
                                                   
        #prepare the Ndiff bandwidth weights
        Ndiff_exponent_params=pull_value_from_fixed_or_free(free_params,fixed_params,'Ndiff_exponent',fixed_or_free_paramdict)
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        Ndiff_bw_kern=modeldict['Ndiff_bw_kern']
        normalization=modeldict['normalize_Ndiffwtsum']
        p_bandwidth_params=pull_value_from_fixed_or_free(self,free_params,fixed_params,'p_bandwidth',fixed_or_free_paramdict)
        
        xBWmaker(max_bw_Ndiff,self.Ndiff_masklist,diffdict,Ndiff_exponent_params,p_bandwidth_params,Ndiff_bw_kern,normalization)
                            
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


                             
