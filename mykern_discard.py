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