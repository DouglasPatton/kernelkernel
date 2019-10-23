
def Ndiff_datastacker(self,Ndiffs,depth):
        """After working on two other approaches, I think this approach to replicating the differences with views via
        np.broadcast_to and then masking them using the pre-optimization-start-built lists of progressively deeper masks
        (though it may prove more effective not to have masks of each depth pre-built)
        """
        #prepare tuple indicating shape to broadcast to
        #print(type(Ndiffs))
        Ndiff_shape=Ndiffs.shape
        if Ndiff_bw_kern=='rbfkern':
            assert Ndiff_shape==(self.nin,self.nin),"Ndiff shape is {}, not nin X nin but bwkern is rbfkern".format(Ndiff_shape)
        if Ndiff_bw_kern=='product':
            assert Ndiff_shape==(self.nin,self.nin,self.p),"Ndiff shape not nin X nin X p but bwkern is product"
        
        #reindex:Ndiff_shape_out_tup=(Ndiff_shape[1],)*depth+(Ndiff_shape[0],)#these are tupples, so read as python not numpy
        Ndiff_shape_out_tup=(self.nin,)*depth+(self.nout,)+(self.npr,)#these are tupples, so read as python not numpy
        if Ndiff_bw_kern=='product':#if parameter dimension hasn't been collapsed yet,
            Ndiff_shape_out_tup=Ndiff_shape_out_tup+(Ndiff_shape[2],)#then add parameter dimension
            # at the end of the tupple
        Ndiff_shape_out_tup=Ndiff_shape_out_tup+(self.npr,)#do all of this for each value that will be predicted
        return np.broadcast_to(Ndiffs,Ndiff_shape_out_tup)#the tupples tells us how to
        #broadcast nin times over <depth> dimensions added to the left side of np.shape      
        '''
        if depth>2:
            Ndiff_shape_out_tup=(self.nin,)*depth+(self.nout,)#these are tupples, so read as python not numpy
            if Ndiff_bw_kern=='product':#if parameter dimension hasn't been collapsed yet,
                Ndiff_shape_out_tup=Ndiff_shape_out_tup+(Ndiff_shape[2],)#then add parameter dimension
            # at the end of the tupple
            return np.broadcast_to(Ndiffs,Ndiff_shape_out_tup)#the tupples tells us how to
            #broadcast nin times over <depth> dimensions added to the left side of np.shape       
        if depth==2:
            Ndiff_shape_out_tup=(self.nin,self.nin,self.nout)
            if Ndiff_bw_kern=='product':#if parameter dimension hasn't been collapsed yet,
                Ndiff_shape_out_tup=Ndiff_shape_out_tup+(Ndiff_shape[2],)#then add parameter dimension
            return np.broadcast_to(np.expand_dims(Ndiffs,2),Ndiff_shape_out_tup)
        '''
        
        
        
        
        
        
        --------------------------------------------------------



                test=self.do_bw_kern(
                    Ndiff_bw_kern,np.ma.array(
                        self.Ndiff_datastacker(Ndiffs,depth+1,Ndiff_bw_kern),#depth+1 b/c depth is in index form
                        mask=self.Ndiff_list_of_masks[depth]
                        ),
                    this_depth_bw_param
                    )
                print('test.shape is {}'.format(test.shape))

"""optimize_free_params( should be totally duplicated now that inheritance is setup, but juts in case, I'm saving it here too
"""
'''   
    def optimize_free_params(self,ydata,xdata,optimizedict):
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
        self.Ndiff_masklist - a list of progressively higher dimension (len=nin) 
            masks to broadcast(views) Ndiff to.
        """
        print(ydata.shape)
        print(xdata.shape)
        self.n,self.p=xdata.shape
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
        method=optimize_dict['method']
        param_valdict=optimizedict['hyper_param_dict']

        #create a list of free paramters for optimization  and
        # dictionary for keeping track of them and the list of fixed parameters too
        free_params,fixed_or_free_paramdict=self.setup_fixed_or_free(model_param_formdict,param_valdict)

        #--------------------------------
        #prep out data as grid (over -3,3) or the original dataset
        xout,yxout=self.prep_out_grid(kerngrid,xdata_std,ydata_std)
        self.xin=xdata_std,self.yin=ydata_std
        self.xout=xout;self.yxout=yxout


        #pre-build list of masks
        self.Ndiff_masklist=self.max_bw_Ndiff_maskstacker(self,nout,nin,max_bw_Ndiff)


        args_tuple=(yin,yxout,xin,xout,modeldict,fixed_or_free_paramdict)
        self.mse=minimize(MY_KDEpredictMSE,free_params,args=args_tuple,method=method)
    '''




old and new mask stacker code before cleanup
for ii in range(max_bw_Ndiff-1):#-1since the first mask has already been computed
            basemask=masklist[-1]#copy the last item to basemask
            masklist.append[np.repeat(np.expand_dim(basemask,0),nin,axis=0)]#then use basemask to
                #start the next item in the masklist with 1 more dimension on the left
            #reindex:for iii in range(2,ii+2,1):
            for iii in range(1,ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                #then this loop maxes at 0,1,2from range(1+2)
                #when i insert a new dimension between two dimensions, I'm effectively transposing
                #take the last item we're constructing and merge it with another mask
                masklist[-1]=np.ma.mask_or(masklist[-1],np.repeat(np.expand_dim(basemask,iii),nin,axis=iii))
                #reindex:ninmask=np.repeat(np.expand_dim(ninmask,ninmask.dim),nin,axis=ninmask.dim)
            #masklist.append(np.ma.mask_or(maskpartlist))#syntax to merge masks


'''old way to assign fixed vs. free, etc.
#if model_param_formdict['p_bandwidth']
free_params=param_valdict['p_bandwidth']#not flexible yet, add exponents later
free_params=np.concatenate(param_valdict['all_x_bandwidth'],free_params,axis=0)#two more parameters for x and y
free_params=np.concatenate(param_valdict['all_y_bandwidth'],free_params,axis=0)                            

Ndiff_exponent=param_valdict['Ndiff_exponent']#fixed right now
if model_param_formdict['Ndiff_exponent']=='fixed':                 
    fixed_or_free_paramdict={'Ndiff_exponent':Ndiff_exponent}#not flexible yet                   
'''
#---------------------------------------------
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



"""max_Ndiff_datastaker is irrelevant. It is built inefficiently and it would be easier
to do the differencing before making the numerous multidimensional copies. 
Also, I would like to run some memory tests on it 
at some point to make sure I understand views vs copies and how that relates to broadcasting.
I believe that this approach using repeat, as with tile, the data is copied. If I switch to broacast_to, then 
python does not copy the data, but just uses views. 
"""
def max_Ndiff_datastacker(self,xdata_std,xout,max_Ndiff):
        """
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
        xinstack=xdata_std[None,:,:]
        xinstackT=xdata_std[:,None,:]
        nout=xout.shape[0]
        Ndifflist=[xoutstack-xinstack]
        for ii in range(max_Ndiff-1)+1:#since the first diff has already been computed
            xinstack=np.repeat(np.expand_dims(xinstack,ii+1),self.n,axis=ii+1)
            xinstackT=np.repeat(np.expand_dims(xinstackT,ii),self.n,axis=ii)
            Ndifflist.append(xinstack-xinstackT)
        return Ndifflist

'''after writing it out, it seems that I do need the vectorized Ndiffstacker and its mask counterpart afterall.
'''
def xBWmaker(self,max_Ndiff,masklist,Ndiffs,Ndiff_exp_params,free_paramlist,Kh_form):
        """using short version of calculating sums of squares with increasing left out observations
        """
        if Kh_form=='exp_l2':
            #Ndiffwt_list=[]
            n=Ndiffs.shape[0]
            if self.outgrid=='no':
                diffsumsq=np.sum(np.square(Ndiffs),axis=1))#don't need to subtract i=j from sum b/c 0 always
                n=n-1
                #Ndiffsumsq=np.sum(np.ma.array(np.square(Ndiffs),mask=np.eye(n),axis=1)
            if self.outgrid=='yes':#smoothing to a grid, so no points dropped
                Ndiffsumsq=np.sum(np.square(Ndiffs),axis=1))
                n_ijk=n
            Ndiffwt_list=[Ndiffsumsq]
            Ndiffwt_subtract=[Ndiffsumsq]
            for ii in range(max_Ndiff):
                Ndiffwt_subtract.append(Ndiffwt_subtract[ii]*n
                Ndiffwt_list.append((Ndiffwt_list[ii]*(n-1))- Ndiffwt_subtract#check syntax on this
                n_ijk-=1
d
