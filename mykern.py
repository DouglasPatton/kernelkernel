import numpy as np
from scipy.optimize import minimize

class kNdtool( object ):
    """kNd refers to the fact that there will be kernels in kernels in these estimators

    """

    def __init__(self):
        pass
    def sum_then_normalize_bw(self,kernstack,normalization):
        '''3 types of Ndiff normalization so far. could extend to normalize by other levels.
        '''
        if normalization=='none':
            return np.ma.sum(kernstack,axis=0)

        if type(normalization) is int:
            return np.ma.sum(kernstack/int)
        if normalization=='across':
            #return np.ma.sum(kernstack/np.ma.mean(kernstack,axis=0),axis=0)
            this_depth_sum=np.ma.sum(kernstack,axis=0)
            return this_depth_sum/np.ma.sum(this_depth_sum,axis=0)#dividing by sum across the sums at "this_depth"

        # if normalization=='across': #does this make sense? not working now.
        #    this_depth_not_summed=kernstack
        #   one_deeper_summed=np.ma.sum(do_bw_kern(Ndiff_bw_kern,np.ma.array(Ndiff_datastacker(Ndiffs,depth+1,Ndiff_bw_kern),mask=self.Ndiff_list_of_masks[depth+1])),axis=0)
        #  n_depth_total=np.ma.sum(np.ma.divide(this_depth_not_summed,one_deeper_summed),axis=0)

    def recursive_BWmaker(self, max_bw_Ndiff, Ndiff_list_of_masks, fixed_or_free_paramdict, diffdict, modeldict):
        """returns an nin X nout np.array of bandwidths
        """
        Ndiff_exponent_params = self.pull_value_from_fixed_or_free('Ndiff_exponent', fixed_or_free_paramdict)
        Ndiff_depth_bw_params = self.pull_value_from_fixed_or_free('Ndiff_depth_bw', fixed_or_free_paramdict)
        max_bw_Ndiff = modeldict['max_bw_Ndiff']
        Ndiff_bw_kern = modeldict['Ndiff_bw_kern']
        normalization = modeldict['normalize_Ndiffwtsum']
        kern_grid = model_dict['kern_grid']

        p_bandwidth_params = self.pull_value_from_fixed_or_free('p_bandwidth', fixed_or_free_paramdict)
        Ndiffs = diffdict['Ndiffs']
        onediffs = diffdict['onediffs']

        if Ndiff_bw_kern == 'rbfkern':  # parameter column already collapsed
            lower_depth_bw=Ndiff_depth_bw_params[1]#there should only be two for recursive Ndiff

            for depth in range(max_bw_Ndiff, 0, -1):  # depth starts with the last mask first #this loop will be memory
                # intensive since I am creating the lower_depth_bw. perhaps there is a better way to complete this
                # nested summation with broadcasting tools in numpy
                if normalization == 'own_n':
                    normalize = self.nin - depth
                else:
                    normalize = normalization
                this_depth_bw = np.ma.power(
                    self.sum_then_normalize_bw(
                        self.do_bw_kern(
                            Ndiff_bw_kern, np.ma.array(
                                self.Ndiff_datastacker(Ndiffs, depth, Ndiff_bw_kern),
                                mask=self.Ndiff_list_of_masks[depth]
                                ),
                            lower_depth_bw #this is the recursive part
                            ),
                        normalize
                        )
                    ,Ndiff_exponent_params[depth]
                    )

                if depth > 1:
                    lower_depth_bw=this_depth_bw
            last_depth_bw=Ndiff_depth_bw_params[0]*np.ma.power(this_depth_bw,Ndiff_exponent_params[0])
            assert last_depth_bw.shape()==(self.nin, self.nout), 'final bw is not ninXnout with rbfkernel'
            return last_depth_bw
        if Ndiff_bw_kern == 'product':  # onediffs parameter column not yet collapsed
            n_depth_masked_sum_kern = self.do_bw_kern(Ndiff_bw_kern, n_depth_masked_sum, Ndiff_depth_bw_params[depth],
                                                      p_bandwidth_params)
            #not developed yet

    def product_BWmaker(self,max_bw_Ndiff,Ndiff_list_of_masks,fixed_or_free_paramdict,diffdict,modeldict):
        """returns an nin X nout np.array of bandwidths
        """
        #for loop starts at deepest Ndiff and works to front
        #axis=depth+1 b/c we want to sum over the last (rbf kern) or 2nd to last (product kern). As can be seen from the
        #tup construction algorithm in Ndiff_datastacker(), there are the first two dimensions that are from the
        #original Ndiff, which is NoutXNin. Then there is a dimension added *depth* times and the last one is what we are
        #collapsing with np.ma.sum.


        Ndiff_exponent_params = self.pull_value_from_fixed_or_free('Ndiff_exponent', fixed_or_free_paramdict)
        Ndiff_depth_bw_params = self.pull_value_from_fixed_or_free('Ndiff_depth_bw', fixed_or_free_paramdict)
        max_bw_Ndiff = modeldict['max_bw_Ndiff']
        Ndiff_bw_kern = modeldict['Ndiff_bw_kern']
        normalization = modeldict['normalize_Ndiffwtsum']
        kern_grid=model_dict['kern_grid']

        p_bandwidth_params = self.pull_value_from_fixed_or_free('p_bandwidth', fixed_or_free_paramdict)
        Ndiffs=diffdict['Ndiffs']
        onediffs=diffdict['onediffs']

        if Ndiff_bw_kern=='rbfkern': #parameter column already collapsed
            for depth in range(max_bw_Ndiff,1,-1):#depth starts with the last mask first #this loop will be memory
                # intensive since I am creating the lower_depth_bw. perhaps there is a better way to complete this
                # nested summation with broadcasting tools in numpy
                if normalization == 'own_n':normalize=self.nin-depth
                else:normalize=normalization
                this_depth_bw=self.sum_then_normalize_bw(
                    self.do_bw_kern(
                        Ndiff_bw_kern,np.ma.array(
                            self.Ndiff_datastacker(Ndiffs,depth,Ndiff_bw_kern),
                            mask=self.Ndiff_list_of_masks[depth]
                        ),
                        Ndiff_depth_bw_params[depth]
                    ),
                    normalize
                )

                if depth<max_bw_Ndiff:#at max depth, no lower depth exists, so leave it alone
                    this_depth_bw=this_depth_bw*np.ma.power(lower_depth_bw,Ndiff_exponent_params[depth+1])
                if depth>2:lower_depth_bw=this_depth_bw#setup for next iteration
            #now the for loop is over and this_depth_bw
            if normalization == 'own_n':
                if kern_grid == 'no': normalize = self.nin - 1  # first item in stack of masks should match these
                if kern_grid == 'yes': normalize = self.nout
            else:
                normalize = normalization
            last_depth_bw=np.ma.power(
                self.sum_then_normalize_bw(
                    self.do_bw_kern(Ndiff_bw_kern,onediffs,Ndiff_depth_bw_params[0]),
                    normalize
                ),
                Ndiff_exponent_params[0]
            )*np.ma.power(this_depth_bw,Ndiff_exponent_params[1])
            assert last_depth_bw.shape()==(self.nin,self.nout),'final bw is not ninXnout with rbfkernel'
            return last_depth_bw
        if Ndiff_bw_kern=='product': #onediffs parameter column not yet collapsed
            n_depth_masked_sum_kern=self.do_bw_kern(Ndiff_bw_kern,n_depth_masked_sum,Ndiff_depth_bw_params[depth],p_bandwidth_params)



    def do_bw_kern(self,kern_choice,maskeddata,Ndiff_depth_bw_param,p_bandwidth_params=None):
        if kern_choice=="product":
            return np.ma.product(p_bandwidth_params,np.ma.exp(-np.ma.power(maskeddata,2)),axis=maskeddata.ndim-1)/Ndiff_depth_bw_param
            #axis-1 b/c axes counted from zero but ndim counts from 1
        if kern_choice=='rbfkern':
            return self.gkernh(maskeddata, Ndiff_depth_bw_param)#parameters already collapsed, so this will be rbf

    def gkernh(self, x, h):
        "returns the gaussian kernel at x with bandwidth h"
        return 1/((2*np.pi)^.5*h)*np.ma.exp(-np.ma.power(x/h)/2, 2)

    def Ndiff_datastacker(self,Ndiffs,depth,Ndiff_bw_kern):
        """After working on two other approaches, I think this approach to replicating the differences with views via
        np.broadcast_to and then masking them using the pre-optimization-start-built lists of progressively deeper masks
        (though it may prove more effective not to have masks of each depth pre-built)
        """
        #prepare tuple indicating shape to broadcast to
        
        Ndiff_shape=Ndiffs.shape()
        if Ndiff_bw_kern=='rbfkern':
            assert Ndiff_shape==(self.nin,self.nin),"Ndiff shape not nin X nin but bwkern is rbfkern"
        if Ndiff_bw_kern=='product':
            assert Ndiff_shape==(self.nin,self.nin,self.p),"Ndiff shape not nin X nin X p but bwkern is product"
        
        Ndiff_shape_out_tup=(Ndiff_shape[1],)*depth+(Ndiff_shape[0],)#these are tupples, so read as python not numpy
        if Ndiff_bw_kern=='product':#if parameter dimension hasn't been collapsed yet,
            Ndiff_shape_out_tup=Ndiff_shape_out_tup+(Ndiff_shape[2],)#then add parameter dimension
            # at the end of the tupple
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
        Ndiff_bw_kern=modeldict['Ndiff_bw_kern']
        if Ndiff_bw_kern=='rbfkern':p=1
        if self.outgrid=='no':
            list_of_masks=[ninmask]
        if self.outgrid=='yes':
            list_of_masks=[np.zero([nin,nout,p])]#reindexed to lkjip
        for ii in range(max_bw_Ndiff-1):#-1since the first mask is in list_of_masks
            lastmask=list_of_masks[-1]#copy the last item to lastmask
            list_of_masks.append(np.repeat(np.expand_dims(lastmask,axis=0),nin,axis=0))#then use lastmask to
            for iii in range(1,ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                #then this loop maxes at 0,1,2from range(1+2)

                #take the last item we're constructing and merge it with another mask
                list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.repeat(np.expand_dims(lastmask,axis=iii),nin,axis=iii))
                #reindex:ninmask=np.repeat(np.expand_dims(ninmask,ninmask.dim),nin,axis=ninmask.dim)
            #list_of_masks.append(np.ma.mask_or(maskpartlist))#syntax to merge masks
            
        return list_of_masks
                            
    def pull_value_from_fixed_or_free(self,param_name,fixed_or_free_paramdict):
        start,end=fixed_or_free_paramdict[param_name]['location_idx']
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='fixed':
            the_param_values=fixed_or_free_paramdict['fixed_params'][start:end]#end already includes +1 to make range inclusive of the end value
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='free':
            the_param_values=fixed_or_free_paramdict['free_params'][start:end]#end already includes +1 to make range inclusive of the end value 
        return the_param_values

    def setup_fixed_or_free(self,model_param_formdict,param_valdict):
        '''takes a dictionary specifying fixed or free and a dictionary specifying starting (if free) or
        fixed (if fixed) values
        returns 1 array and a dict
            free_params 1 dim np array of the starting parameter values in order,
                outside the dictionary to pass to optimizer
            fixed_params 1 dim np array of the fixed parameter values in order in side the dictionary
            fixed_or_free_paramdict a dictionary for each parameter (param_name) with the following key:val
                fixed_or_free:'fixed' or 'free'
                location_idx: the start and end location for parameters of param_name in the appropriate array,
                    notably end location has 1 added to it, so python indexing will correctly include the last value.
                fixed_params:array of fixed params
                Once inside optimization, the following will be added
                free_params:array of free params or string:'outside' if the array has been removed to pass to the optimizer
        '''
        fixed_params=np.array([]);free_params=np.array([]);fixed_or_free_paramdict={}
        #build fixed and free vectors of hyper-parameters based on hyper_param_formdict
        for param_name,param_form in model_param_formdict.items():
            param_feature_dict={}
            param_val=param_valdict[param_name]
            print('param_val',param_val)
            print('param_form',param_form)
            assert param_val.ndim==1,"values for {} have not ndim==1".format(param_name)
            if param_form=='fixed':
                param_feature_dict['fixed_or_free']='fixed'
                param_feature_dict['location_idx']=(len(fixed_params),len(fixed_params)+len(param_val))
                    #start and end indices, with end already including +1 to make python slicing inclusive of end in start:end
                fixed_params=np.concatenate([fixed_params,param_val],axis=0)
                fixed_or_free_paramdict[param_name]=param_feature_dict
            if param_form=='free':
                param_feature_dict['fixed_or_free']='free'
                param_feature_dict['location_idx']=(len(free_params),len(free_params)+len(param_val))
                    #start and end indices, with end already including +1 to make python slicing inclusive of end in start:end
                free_params=np.concatenate([free_params,param_val],axis=0)
                fixed_or_free_paramdict[param_name]=param_feature_dict
        fixed_or_free_paramdict['free_params']='outside'
        fixed_or_free_paramdict['fixed_params'] = fixed_params

        return free_params,fixed_or_free_paramdict

    
    def MY_KDEpredictMSE (self,free_params,yin,yxout,xin,xout,modeldict,fixed_or_free_paramdict):
        """moves free_params to first position of the obj function, preps data, and then runs MY_KDEreg to fit the model
            then returns MSE of the fit 
        Assumes last p elements of free_params are the scale parameters for 'el two' approach to
        columns of x.
        """
        print('starting optimization of hyperparameters')
        #is the masking approach sufficient for leave one out cross validation?
        #kern_grid='no' forces masking of self for predicting self
        
        #add free_params back into fixed_or_free_paramdict now that inside optimizer
        fixed_or_free_paramdict['free_params']=free_params
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        #pull p_bandwidth parameters from the appropriate location and appropriate vector
        p_bandwidth_params=self.pull_value_from_fixed_or_free('p_bandwidth',fixed_or_free_paramdict)

        #p=xin.shape[1]
        print(p_bandwidth_params)
        p=p_bandwidth_params.shape[0]
        assert self.p==p,\
            "p={} but p_bandwidth_params.shape={}".format(self.p,p_bandwidth_params.shape)


        if modeldict['Ndiff_bw_kern']=='rbfkern':
            
            xin_scaled=xin*p_bandwidth_params
            xout_scaled=xout*p_bandwidth_params
            yxout_scaled=yxout*np.concatenate([np.array([1]),p_bandwidth_params],axis=0)
            y_yxout=yxout_scaled[:,0]
            x_yxout=yxout_scaled[:,1:]
            y_onediffs=self.makediffmat_itoj(yin,y_yxout)
            y_Ndiffs=self.makediffmat_itoj(yin,yin)
            onediffs_scaled_l2norm=np.power(np.sum(np.power(self.makediffmat_itoj(xin_scaled,xout_scaled),2),axis=p),.5)

            if modeldict['kern_grid']=='no':
                Ndiffs_scaled_l2norm=onediffs_scaled_l2norm
            else:
                Ndiffs_scaled_l2norm=np.power(np.sum(np.power(self.makediffmat_itoj(xin_scaled,x_yxout),2),axis=p),.5)
            assert onediffs_scaled_l2norm.shape==(xin.shape[0],xout.shape[0]),'onediffs_scaled_l2norm does not have shape=(nin,nout)'

            diffdict={}
            diffdict['onediffs']=onediffs_scaled_l2norm
            diffdict['Ndiffs']=Ndiffs_scaled_l2norm
            ydiffdict={}
            ydiffdict['onediffs']=y_onediffs
            ydiffdict['Ndiffs']=y_Ndiffs
            diffdict['ydiffdict']=ydiffdict


        if modeldict['Ndiff_bw_kern']=='product':
            onediffs=makediffmat_itoj(xout,xin)#scale now? if so, move if...='rbfkern' down 
            #predict
            yhat=MY_KDEreg(yin,xin_scaled,xout_scaled,y_yxout,x_yxout,fixed_or_free_paramdict,diffdict,modeldict)
            #not developed yet

        # prepare the Ndiff bandwidth weights
        if modeldict['Ndiff_type'] == 'product':
            xbw = self.product_BWmaker(max_bw_Ndiff, self.Ndiff_list_of_masks, fixed_or_free_paramdict, diffdict, modeldict)
            ybw = self.product_BWmaker(max_bw_Ndiff, self.Ndiff_list_of_masks, fixed_or_free_paramdict, diffdict['ydiffdict'],
                modeldict)
        if modeldict['Ndiff_type'] == 'recursive':
            xbw = self.recursive_BWmaker(max_bw_Ndiff, self.Ndiff_list_of_masks, fixed_or_free_paramdict, diffdict, modeldict)
            ybw = self.recursive_BWmaker(max_bw_Ndiff, self.Ndiff_list_of_masks, fixed_or_free_paramdict, diffdict['ydiffdict'],
                                modeldict)
        #extract and multiply ij varying part of bw times non varying part
        hx=self.pull_value_from_fixed_or_free('outer_x_bw', fixed_or_free_paramdict)
        hy=self.pull_value_from_fixed_or_free('outer_y_bw', fixed_or_free_paramdict)

        xbw=xbw*hx#need to make this flexible to blocks of x
        ybw=ybw*hy

        xonediffs=diffdict['onediffs']
        yonediffs=diffdict['ydiffdict']['onediffs']
        yx_onediffs_endstack=np.concatenate([yonediffs[:,:,None],xonedifs[:,:,None]],axis=2)
        yx_bw_endstack=np.concatenate([ybw[:,:,None],xbw[:,:,None]],axis=2)
        prob_x = do_KDEsmalln(one_diffs, xbw, fixed_or_free_paramdict, modeldict)
        prob_yx = do_KDEsmalln(yx_one_diffs_endstack, yx_bw_endstack, fixed_or_free_paramdict, modeldict)

        if modeldict['regression_model']=='NW':
            yhat = my_NW_KDEreg(prob_yx,prob_x,y_yxout)
        #here is the simple MSE objective function. however, I think I need to use
        #the more sophisticated MISE or mean integrated squared error,
        #either way need to replace with cost function function
        yin_un_std=yin*self.ystd+self.ymean
        yhat_un_std=yin*self.ystd+self.ymean
        y_err=yin_un_std=yhat_un_std
        return np.sum(np.power(y_err,2))


    def my_NW_KDEreg(self,prob_yx,prob_x,y_yxout):
        """returns predited values of y for xpredict based on yin, xin, and modeldict
        """
        cdfnorm_prob_yx=prob_yx/np.sum(prob_yx,axis=0)
        cdfnorm_prob_x = prob_x / np.sum(prob_x, axis=0)
        return np.sum(y_yxout*cdfnorm_prob_yx/cdfnorm_prob_x)

    def makediffmat_itoj(self,xin,xout):
        return np.expand_dims(xin, axis=1) - np.expand_dims(xout, axis=0)#should return ninXnoutXp if xin an xout were ninXp and noutXp
            

    def do_KDEsmalln(self,onediffs,xbw,modeldict):
        """estimate the density items in onediffs. collapse via products if dimensionality is greater than 2
        first 2 dimensions of onediffs must be ninXnout
        """
        assert onediffs.shape()==xbw.shape(), "onediffs is shape:{} while xbw is shape:{}".format(onediffs.shape(),xbw.shape())
        allkerns=self.gkernh(onediffs, xbw)
        #collapse by random variables indexed in last axis until allkerns.ndim=2
        normalization=modeldict['product_kern_norm']
        if normalization =='self':
            allkerns=allkerns/np.sum(allkerns,axis=0)    #need to check this logic. should I
            # collapse just nin dim or both lhs dims?
        if allkerns.ndim>2:
            for i in range((allkerns.ndim()-2),0,-1):
                assert allkerns.ndim>2, "allkerns is being collapsed via product on rhs " \
                                        "but has {} dimensions instead of ndim>2".format(allkerns.ndim)
                allkerns=np.product(allkerns,axis=i+2)#collapse right most dimension
        assert allkerns.shape()==(self.nin,self.nout), "allkerns is shaped{} not {} X {}".format(allkerns.shape(),self.nin,self.nout)
        return np.ma.sum(allkerns,axis=0)/self.nin#collapsing across the nin kernels for each of nout


    def MY_KDE_gridprep_smalln(self,n,p,kern_grid):
        """creates a grid with all possible combinations of n evenly spaced values from -3 to 3.
        """
        agrid=np.linspace(-3,3,n) #assuming variables have been standardized
        for idx in range(n-1):#-1 b/c agrid already created; need to figure out how to better vectorize this loop
            agrid=np.concatenate([np.repeat(agrid,n,axis=0),np.repeat(np.linspace(-3,3,n)[:,None],n**(idx+1),axis=0)],axis=1)
            #assertions added to check shape of output
        return agrid

    def prep_out_grid(self,kerngrid,xdata_std,ydata_std):
        '''#for small data, pre-create the 'grid'/out data
        no big data version for now
        '''
        # if self.n<10**5 and not (type(kerngrid)==int and kerngrid**self.p>10**8):
        #    self.data_is_small='yes'
        if type(kerngrid) is int:
            self.nout=kerngrid**self.p
            xout=MY_KDE_gridprep_smalln(kerngrid,self.p)
            assert xout.shape[1]==self.p,'xout has wrong number of columns'
            assert xout.shape[0]==kerngrid**self.p,'xout has wrong number of rows'

            yxout=MY_KDE_gridprep_smalln(kerngrid,self.p+1)
            assert yxout.shape[1]==self.p+1,'yxout has wrong number of columns'
            assert yxout.shape[0]==kerngrid**(self.p+1),'yxout has wrong number of rows'
            self.outgrid='yes'
        if kerngrid=='no':
            self.nout=self.n
            xout=xdata_std;
            yxout=np.concatenate([ydata_std[:,None],xdata_std],axis=1)
            self.outgrid='no'
        return xout,yxout

    def standardize_yx(self,xdata,ydata):
        self.xmean=np.mean(xdata,axis=0)
        self.ymean=np.mean(ydata,axis=0)
        self.xstd=np.std(xdata,axis=0)
        self.ystd=np.std(ydata,axis=0)
        standard_x=(xdata-self.xmean)/self.xstd
        standard_y=(ydata-self.ymean)/self.ystd
        return standard_x,standard_y

class optimize_free_params(kNdtool):
    """"This is the method for iteratively running kernelkernel to optimize hyper parameters
    optimize dict contains starting values for free parameters, hyper-parameter structure(is flexible),
    and a model dict that describes which model to run including how hyper-parameters enter (quite flexible)
    speed and memory usage is a big goal when writing this. I pre-created masks to exclude the increasing
    list of centered data points. see mykern_core for an example and explanation of dictionaries.
    Flexibility is also a goal. max_bw_Ndiff is the deepest the model goes.
    ------------------
    attributes created
    self.n,self.p,self.optdict
    self.xdata_std, self.xmean,self.xstd
    self.ydata_std,self.ymean,self.ystd
    self.Ndiff - the nout X nin X p matrix of first differences of xdata_std
    self.Ndiff_list_of_masks - a list of progressively higher dimension (len=nin)
        masks to broadcast(views) Ndiff to.
    """

    def __init__(self,ydata,xdata,optimizedict):
        kNdtool.__init__(self)

        print(ydata.shape)
        print(xdata.shape)
        self.nin,self.p=xdata.shape
        self.n=self.nin
        self.optdict=optimizedict
        assert ydata.shape[0]==xdata.shape[0],'xdata.shape={} but ydata.shape={}'.format(xdata.shape,ydata.shape)

        #standardize x and y and save their means and std to self
        xdata_std,ydata_std=self.standardize_yx(xdata,ydata)
        #store the standardized (by column or parameter,p) versions of x and y
        self.xdata_std=xdata_std;self.ydata_std=ydata_std
        #extract optimization information, including modeling information in model dict,
        #parameter structure in model dict, and starting free parameter values from paramdict
        #these options are still not fully mature and actually flexible
        modeldict=optimizedict['model_dict'] #reserve _dict for names of dicts in *keys* of parent dicts
        model_param_formdict=modeldict['hyper_param_form_dict']
        kerngrid=modeldict['kern_grid']
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        method=optimizedict['method']
        param_valdict=optimizedict['hyper_param_dict']

        #create a list of free paramters for optimization  and
        # dictionary for keeping track of them and the list of fixed parameters too
        free_params,fixed_or_free_paramdict=self.setup_fixed_or_free(model_param_formdict,param_valdict)
        print('free_params',free_params)
        print('fixed_params',fixed_or_free_paramdict['fixed_params'])
        #--------------------------------
        #prep out data as grid (over -3,3) or the original dataset
        xout,yxout=self.prep_out_grid(kerngrid,xdata_std,ydata_std)
        self.xin=xdata_std;self.yin=ydata_std
        self.xout=xout;self.yxout=yxout
        self.nout=xout.shape[0]


        #pre-build list of masks
        self.Ndiff_list_of_masks=self.max_bw_Ndiff_maskstacker(self.nout,self.nin,self.p,max_bw_Ndiff,modeldict)


        args_tuple=(self.yin,self.yxout,self.xin,self.xout,modeldict,fixed_or_free_paramdict)
        self.mse=minimize(self.MY_KDEpredictMSE,free_params,args=args_tuple,method=method)

if __name__=="_main__":
    pass