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
