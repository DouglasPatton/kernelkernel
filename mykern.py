import numpy as np
from scipy.optimize import minimize

class kNdtool():
    """kNd refers to the fact that there will be kernels in kernels in these estimators

    """

    def max_Ndiff_maskstacker(self,nout,nin,p,max_Ndiff):
        "match the parameter structure of Ndifflist produced by max_Ndiff_datastacker
        ninmask=np.repeat(np.eye(nin)[:,:,None],p,axis=2)
        if self.outgrid=='no':
            masklist=[nXnmask]
        if self.outgrid=='yes':
            masklist=[np.zero([nout,nin,p])]
        
        for ii in range(max_Ndiff-1)+1:#since the first mask has already been computed
            maskpartlist=[]
            for iii in range(ii+2):
                maskpartlist.append(np.repeat(np.expand_dim(ninmask,iii),nin,axis=iii))
            masklist.append[np.ma.mask_or(maskpartlist)#syntax to merge masks
        return masklist                    
        

    def max_Ndiff_datastacker(self,xdata_std,xout,max_Ndiff):
        """Broadcasting makes this code unnecessary. pre-Masking is sufficient.
        take 2 arrays expands their last and second to last dimensions,
        respectively, then repeats into that dimension, then takes differences
        between the two and saves those differences to a list.
        xout_i-xin_j is the top level difference (N times) and from their it's always xin_j-xin_k, xin_k-xin_l.
        Generally xin_j-xin_k have even probabilities, based on the sample, but
        xgridi-xgridj or xgrid_j-xgrid_k have probabilities that must be calculated
        later go back and make this more flexible and weight xgrids by probabilities and provide an option
        """
        
        '''below, dim1 is like a child of dim0. for each item in dim0,we need to subtract
        each item which vary along dim1 xout is copied over the new dimension for
        the many times we will subtract xin from each one. later we will sum(or multiply
        for produt kernels) over the rhs dimension (p), then multiply kernels/functions
        of the remaining rhs dimensions until we have an nx1 (i.e., [n,]) array that is our
        kernel for that obervation
        '''
        xoutstack=xout[:,None,:]
        #the None layer of xinstack *should* (I think) be broadcast a
        #different number of times at different uses below
        xinstack=xdata_std[None,:,:]
        xinstackT=xdata_std[:,None,:]
        nout=xout.shape[0]
        Ndifflist=[xoutstack-xinstack]
        for ii in range(max_Ndiff-1)+1:#since the first diff has already been computed
            xinstack=np.repeat(np.expand_dims(xinstack,ii+1),self.n,axis=ii+1)
            xinstackT=np.repeat(np.expand_dims(xinstackT,ii),self.n,axis=ii)
            Ndifflist.append(xinstack-xinstackT)
        return Ndifflist
        
    def optimize_hyper_params(self,ydata,xdata,optimizedict):
        self.n,self.p=xdata.shape
        self.optdict=optimizedict
        
        assert ydata.shape[0]==xdata.shape[0],'x and y do not match in dimension 0'

        #standardize x and y and save their means and std to self
        xdata_std,ydata_std=standardize_xy(xdata,ydata)
        #store the standardized (by column or parameter,p) versions of x and y
        self.xdata_std=xdata_std;self.ydata_std=ydata_std
        #extract optimization information, including modeling information in model dict,
        #parametr structure in model dict, and starting free parameter values from paramdict
        #these options are still not mature
        modeldict=optimizedict['model_dict'] #reserve _dict for names of dicts in *keys* of parent dicts
        model_param_formdict=modeldict['hyper_param_form_dict']
        kerngrid=modeldict['kern_grid']
        max_Ndiff=modeldict['max_Ndiff']
        method=optimize_dict['method']
        param_valdict=optimizedict['hyper_param_dict']
        free_paramlist=param_valdict['p_bandwidth']#not flexible yet, add exponents later
        
        #prep out data as grid (over -3,3) or the original dataset
        xout,xyout=prep_out_grid(kerngrid,xdata_std,ydata_std)
        self.xout=xout;self.xyout=xyout
                    
        #for small data pre-build lists of multi dimensional differences and masks and masks to differences.
        #self.Ndifflist=max_Ndiff_datastacker(xdata_std,xout,max_Ndiff) 
        self.Ndiff_masklist=max_Ndiff_maskstacker(self,nout,nin,max_Ndiff)
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
        
        
        args_tuple=(modeldict)
                
        return minimize(MY_KDEregMISE,free_paramlist,args=args_tuple,method=method) 


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
        #here is the simple MSE objective function. however, I think I need to use
        #the more sophisticated MISE or mean integrated squared error
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
        #for small data, pre-create the 'grid'/out data 
        if self.n<10**5 and not (type(kerngrid)==int and kerngrid**self.p>10**8):
            self.data_is_small='yes'
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


                             
