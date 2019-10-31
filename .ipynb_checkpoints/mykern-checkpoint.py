import numpy as np
#from numba import jit
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
            return np.ma.sum(kernstack,axis=0)/normalization
        if normalization=='across':
            #return np.ma.sum(kernstack/np.ma.mean(kernstack,axis=0),axis=0)
            this_depth_sum=np.ma.sum(kernstack,axis=0)
            return this_depth_sum/np.ma.sum(this_depth_sum,axis=0)#dividing by sum across the sums at "this_depth"

        # if normalization=='across': #does this make sense? not working now.
        #    this_depth_not_summed=kernstack
        #   one_deeper_summed=np.ma.sum(do_bw_kern(Ndiff_bw_kern,np.ma.array(Ndiff_datastacker(Ndiffs,depth+1,Ndiff_bw_kern),mask=self.Ndiff_list_of_masks[depth+1])),axis=0)
        #  n_depth_total=np.ma.sum(np.ma.divide(this_depth_not_summed,one_deeper_summed),axis=0)

    def recursive_BWmaker(self, max_bw_Ndiff, Ndiff_list_of_masks, fixed_or_free_paramdict, diffdict, modeldict, x_or_y):
        """returns an nin X nout npr np.array of bandwidths if x_or_y=='y'
        or nin X npr if x_or_y=='x'
        """
        Ndiff_exponent_params = self.pull_value_from_fixed_or_free('Ndiff_exponent', fixed_or_free_paramdict)
        Ndiff_depth_bw_params = self.pull_value_from_fixed_or_free('Ndiff_depth_bw', fixed_or_free_paramdict)
        max_bw_Ndiff = modeldict['max_bw_Ndiff']
        Ndiff_bw_kern = modeldict['Ndiff_bw_kern']
        normalization = modeldict['normalize_Ndiffwtsum']
        #kern_grid = model_dict['kern_grid']

        x_bandscale_params = self.pull_value_from_fixed_or_free('x_bandscale', fixed_or_free_paramdict)
        Ndiffs = diffdict['Ndiffs']
        onediffs = diffdict['onediffs']

        if Ndiff_bw_kern == 'rbfkern':  # parameter column already collapsed
            lower_depth_bw=Ndiff_depth_bw_params[1]#there should only be two for recursive Ndiff

            for depth in range(max_bw_Ndiff-1, 0, -1):  # depth starts with the last mask first #this loop will be memory
                # intensive since I am creating the lower_depth_bw. perhaps there is a better way to complete this
                # nested summation with broadcasting tools in numpy
                if normalization == 'own_n':
                    normalize = self.nin - depth - 1
                else:
                    normalize = normalization
                this_depth_bw = np.ma.power(
                    self.sum_then_normalize_bw(
                        self.do_bw_kern(
                            Ndiff_bw_kern, np.ma.array(
                                self.Ndiff_datastacker(Ndiffs, noediffs.shape,depth+1),
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
            assert last_depth_bw.shape==(self.nin, self.nout), 'final bw is not ninXnout with rbfkernel'
            return last_depth_bw
        if Ndiff_bw_kern == 'product':  # onediffs parameter column not yet collapsed
            n_depth_masked_sum_kern = self.do_bw_kern(Ndiff_bw_kern, n_depth_masked_sum, Ndiff_depth_bw_params[depth],
                x_bandscale_params)
            return "n/a"
        pass    

    
    def product_BWmaker(self,max_bw_Ndiff,Ndiff_list_of_masks,fixed_or_free_paramdict,diffdict,modeldict,x_or_y):
        """returns an nin X nout npr np.array of bandwidths if x_or_y=='y'
        ? or nin X npr if x_or_y=='x' ?
        """
        Ndiff_exponent_params = self.pull_value_from_fixed_or_free('Ndiff_exponent', fixed_or_free_paramdict)
        Ndiff_depth_bw_params = self.pull_value_from_fixed_or_free('Ndiff_depth_bw', fixed_or_free_paramdict)
        #print('Ndiff_depth_bw_params',Ndiff_depth_bw_params)
        #max_bw_Ndiff = modeldict['max_bw_Ndiff']
        Ndiff_bw_kern = modeldict['Ndiff_bw_kern']
        normalization = modeldict['normalize_Ndiffwtsum']
        #kern_grid=modeldict['kern_grid']

        x_bandscale_params = self.pull_value_from_fixed_or_free('x_bandscale', fixed_or_free_paramdict)
        Ndiffs=diffdict['Ndiffs']
        onediffs=diffdict['onediffs']
        if x_or_y=='y':
            masklist=self.Ndiff_list_of_masks_y
            #datastacker=self.Ndiff_datastacker_y #this is the function name
        if x_or_y=='x':
            masklist=self.Ndiff_list_of_masks_x
            
        if Ndiff_bw_kern=='rbfkern': #parameter column already collapsed
            if x_or_y=='y':
                this_depth_bw=np.ones([self.nin,self.nout,self.npr])
            if x_or_y=='x':
                this_depth_bw=np.ones([self.nin,1,self.npr])#x doesn't vary over nout like y does, so just 1 for a dimension placeholder.
            deeper_depth_bw=np.array([1])
            for depth in range(max_bw_Ndiff,0,-1):
                #print('depth={}'.format(depth))
                if normalization == 'own_n':normalize=self.nin-(depth)
                else:normalize=normalization
                this_depth_bw_param=Ndiff_depth_bw_params[depth-1]
                this_depth_mask=masklist[depth]
                this_depth_exponent=Ndiff_exponent_params[depth-1]
                this_depth_data=self.Ndiff_datastacker(Ndiffs,onediffs.shape,depth)
                
                #print('this_depth_bw_param',this_depth_bw_param)

                this_depth_bw=np.ma.power(
                    self.sum_then_normalize_bw(
                        self.do_bw_kern(
                            Ndiff_bw_kern,
                            np.ma.array(this_depth_data,mask=this_depth_mask),
                            this_depth_bw_param
                            )*deeper_depth_bw,
                        normalize
                        ),
                    this_depth_exponent 
                    )
                #print('depth:',depth,'this_depth_bw masked count:',np.ma.count_masked(this_depth_bw),'with shape:',this_depth_bw.shape)
                #print('this_depth_bw.shape=',this_depth_bw.shape)
                if depth>1: deeper_depth_bw=this_depth_bw#setup deeper_depth_bw for next iteration if there is another
            
           # this_depth_bw=this_depth_bw*Ndiff_depth_bw_params[0]#simple version that doesn't vary with i, but j only
            
            
            

       
                                    
            #assert this_depth_bw.shape==(self.nin,self.nout),'final bw is {} but expected ninXnout({}X{}) with rbfkernel'.format(this_depth_bw.shape,self.nin,self.nout)
            
            return this_depth_bw
                
        if Ndiff_bw_kern=='product': #onediffs parameter column not yet collapsed
            n_depth_masked_sum_kern=self.do_bw_kern(Ndiff_bw_kern,n_depth_masked_sum,Ndiff_depth_bw_params[depth],x_bandscale_params)


    
    def do_bw_kern(self,kern_choice,maskeddata,Ndiff_depth_bw_param,x_bandscale_params=None):
        if kern_choice=="product":
            return np.ma.product(x_bandscale_params,np.ma.exp(-np.ma.power(maskeddata,2)),axis=maskeddata.ndim-1)/Ndiff_depth_bw_param
            #axis-1 b/c axes counted from zero but ndim counts from 1
        if kern_choice=='rbfkern':
            return self.gkernh(maskeddata, Ndiff_depth_bw_param)#parameters already collapsed, so this will be rbf
    
    def gkernh(self, x, h):
        "returns the gaussian kernel at x with bandwidth h"
        #print("x.shape",x.shape)
        #print('h.shape',h.shape)
        #print('h',h)
        
        denom=1/((2*np.pi)**.5*h)
        #print('gkern_denom:',denom)
        numerator=np.ma.exp(-np.ma.power(x/(h*2), 2))
        #print('excess numerator mask count',np.ma.count_masked(numerator)-np.ma.count_masked(h))    
        #print('excess denom mask count',np.ma.count_masked(denom)-np.ma.count_masked(h))    
        excess_denom_mask=np.ma.count_masked(denom)-np.ma.count_masked(h)
        #if excess_denom_mask>0:
        #    print('h',h)
        #    print('denom',denom)
        
        kern=denom*numerator
        #print('kern',kern)
        hmaskcount=np.ma.count_masked(h)
        kernmaskedcount=np.ma.count_masked(kern)
        xmaskedcount=np.ma.count_masked(x)
        #print('gkernh maskcount',kernmaskedcount,'kernshape',kern.shape,'maskeddata_maskcount',xmaskedcount,'maskeddata_shape',x.shape,'hmaskcount',hmaskcount)
        #if kernmaskedcount==22:
        #    print('denom,numerator',[(i,k) for i,k in zip(denom,numerator)])
        #    print(self.fixed_or_free_paramdict)
        return kern
        #return np.ma.exp(-np.ma.power(x/h/2, 2))

    
    
    def Ndiff_datastacker(self,Ndiffs,onediffs_shape,depth):
        """
        """
        if len(onediffs_shape)==3:#this should only happen if we're working on y
            #assert depth>0,"depth is not greater than zero, depth is:{}".format(depth)
            ytup=(self.nin,)*depth+onediffs_shape#depth-1 b/c Ndiffs starts as ninXninXnpr
            #print('Ndiffs.shape',Ndiffs.shape,'ytup',ytup,'depth',depth)
            return np.broadcast_to(np.expand_dims(Ndiffs,2),ytup)
        if len(onediffs_shape)==2:#this should only happen if we're working on x
            Ndiff_shape_out_tup=(self.nin,)*depth+onediffs_shape
            return np.broadcast_to(Ndiffs,Ndiff_shape_out_tup)#no dim exp b/c only adding to lhs of dim tupple
    
    def max_bw_Ndiff_maskstacker_y(self,npr,nout,nin,p,max_bw_Ndiff,modeldict):
        #print('nout:',nout)
        Ndiff_bw_kern=modeldict['Ndiff_bw_kern']
        ykerngrid=modeldict['ykern_grid']
        #ninmask=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,:,None],(nin,nin,npr))
        
        if not modeldict['predict_self_without_self']=='yes':
            ninmask=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,:,None],(nin,nin,npr))
        if modeldict['predict_self_without_self']=='yes' and nin==npr and ykerngrid=='no':
            
            ninmask3=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,:,None],(nin,nin,npr))
            ninmask2=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,None,:],(nin,nin,nin))
            ninmask1=np.broadcast_to(np.ma.make_mask(np.eye(nin))[None,:,:],(nin,nin,nin))
            ninmask=np.ma.mask_or(ninmask1,np.ma.mask_or(ninmask2,ninmask3))
        
        if modeldict['predict_self_without_self']=='yes' and nin==npr and type(ykerngrid) is int:
            
            #ninmask3=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,:,None],(nin,nin,npr))
            ninmask=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,None,:],(nin,nin,nin))#nin not used to calculate npr
            #ninmask1=np.broadcast_to(np.ma.make_mask(np.eye(nin))[None,:,:],(nin,nin,nin))
            #ninmask=np.ma.mask_or(ninmask1,np.ma.mask_or(ninmask2,ninmask3))
        #print('ninmask',ninmask)
        
                
        list_of_masks=[ninmask]
        if max_bw_Ndiff>0:
            if ykerngrid=='no':
                firstdiffmask=np.ma.mask_or(np.broadcast_to(np.expand_dims(ninmask,0),(nin,nin,nin,npr)),np.broadcast_to(np.expand_dims(ninmask,2),(nin,nin,nin,npr)))
                firstdiffmask=np.ma.mask_or(np.broadcast_to(np.expand_dims(ninmask,1),(nin,nin,nin,npr)),firstdiffmask)#when 0 dim of ninmask is expanded, masked if i=j for all k.
                #when 2 dim of nin mask is expanded, masked if k=j for all i, and when 1 dim of nin mask is expanded, masked if k=i for all j. all are for all ii
            if type(ykerngrid) is int:
                firstdiffmask=np.broadcast_to(np.expand_dims(ninmask,2),(nin,nin,nout,npr))
                #if yout is a grid and nout not equal to nin, 
            list_of_masks.append(firstdiffmask)# this line is indexed (k,j,i,ii)
                
        if max_bw_Ndiff>1:
            
            for ii in range(max_bw_Ndiff-1):#-1 b/c 1diff masks already in second position of list of masks if max_bw_Ndiff>0
                lastmask=list_of_masks[-1] #second masks always based on self.nin
                masktup=(nin,)+lastmask.shape#expand dimensions on lhs
                #print("y:lastmask.shape:",lastmask.shape)
                list_of_masks.append(np.broadcast_to(np.expand_dims(lastmask,0),masktup))
                #print("y mask stack, lastmask.shape:",lastmask.shape)
                #print("list_of_masks[-1].shape:",list_of_masks[-1].shape)
        
                for iii in range(ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                    #then this loop maxes at 0,1 from range(2-1+1)
                    iii+=1 #since 0 dim added before for_loop
                    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.broadcast_to(np.expand_dims(lastmask,axis=iii),masktup))
                if ykerngrid=='no':
                    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.broadcast_to(np.expand_dims(lastmask,axis=ii+3),masktup))#this should mask \
                    #the yout values from the rest since yout==yin
                #if modeldict['predict_self_without_self']=='yes':
                #    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.broadcast_to(np.expand_dims(lastmask,axis=ii+4),(nin,)+masktup))#this should mask \
                    #the yin values when predicting yin

        
        if type(ykerngrid) is int:
            list_of_masks[0]=np.zeros([nin,nout,npr])#overwrite first item in list of masks to remove masking when predicting y using ykerngrid==int
        #[print('shape of mask {}'.format(i),list_of_masks[i].shape) for i in range(max_bw_Ndiff+1)]
        return list_of_masks
        
    def max_bw_Ndiff_maskstacker_x(self,npr,nout,nin,p,max_bw_Ndiff,modeldict):
        '''match the parameter structure of Ndifflist produced by Ndiff_datastacker
        notably, mostly differences (and thus masks) will be between the nin (n in the original dataset) obeservations.
        though would be interesting to make this more flexible in the future.
        when i insert a new dimension between two dimensions, I'm effectively transposing
        '''
        Ndiff_bw_kern=modeldict['Ndiff_bw_kern']
        ykerngrid=modeldict['ykern_grid']
        assert Ndiff_bw_kern=='rbfkern','only rbfkern developed and your kern is {}'.format(Ndiff_bw_kern)
            
        ninmask=np.ma.make_mask(np.eye(nin))
        list_of_masks=[ninmask]
        if not modeldict['predict_self_without_self']=='yes' and max_bw_Ndiff>0:#first mask will be corrected at the bottom
            list_of_masks.append(np.broadcast_to(ninmask[:,:,None],(nin,nin,npr)))
        if modeldict['predict_self_without_self']=='yes' and nin==npr and max_bw_Ndiff>0:
                ninmask3=np.broadcast_to(ninmask[:,:,None],(nin,nin,nin))
                ninmask2=np.broadcast_to(ninmask[:,None,:],(nin,nin,nin))
                ninmask1=np.broadcast_to(ninmask[None,:,:],(nin,nin,nin))
                list_of_masks.append(np.ma.mask_or(ninmask1,np.ma.mask_or(ninmask2,ninmask3)))
        #ninmask=np.ma.make_mask(np.repeat(np.eye(nin)[:,:,None],p,axis=2))#this one will be used extensively to construct masks
                
        if max_bw_Ndiff>1:
            
            for ii in range(max_bw_Ndiff-1):#-1 b/c 1diff masks already in second position of list of masks if max_bw_Ndiff>0
                lastmask=list_of_masks[-1] #second masks always based on self.nin
                masktup=(nin,)+lastmask.shape#expand dimensions on lhs
                #print("x:lastmask.shape:",lastmask.shape)
                list_of_masks.append(np.broadcast_to(np.expand_dims(lastmask,0),masktup))
                #print("x mask stack, lastmask.shape:",lastmask.shape)
                #print("list_of_masks[-1].shape:",list_of_masks[-1].shape)
        
                for iii in range(ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                    #then this loop maxes at 0,1 from range(2-1+1)
                    iii+=1 #since 0 dim added before for_loop
                    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.broadcast_to(np.expand_dims(lastmask,axis=iii),masktup))
                
                    #the yout values from the rest since yout==yin
                #if modeldict['predict_self_without_self']=='yes':
                #    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.broadcast_to(np.expand_dims(lastmask,axis=ii+4),(nin,)+masktup))#this should mask \
                    #the yin values when predicting yin
                
            lastmask=list_of_masks[-1]#copy the last item to lastmask
        #[print('shape of mask {}'.format(i),list_of_masks[i].shape) for i in range(max_bw_Ndiff)]
        if not modeldict['predict_self_without_self']=='yes':
            masklist[0]=np.ma.make_mask(np.zeros(nin,npr))

        return list_of_masks
    
    def return_param_name_and_value(self,fixed_or_free_paramdict,modeldict):
        params={}
        paramlist=[key for key,val in modeldict['hyper_param_form_dict'].items()]
        for param in paramlist:
            paramdict=fixed_or_free_paramdict[param]
            form=paramdict['fixed_or_free']
            const=paramdict['const']
            start,end=paramdict['location_idx']
            
            value=fixed_or_free_paramdict[f'{form}_params'][start:end]
            if const=='non-neg':
                const=f'{const}'+':'+f'{np.exp(value)}'
            params[param]={'value':value,'const':const}
        return params
    
    
    def pull_value_from_fixed_or_free(self,param_name,fixed_or_free_paramdict):
        start,end=fixed_or_free_paramdict[param_name]['location_idx']
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='fixed':
            the_param_values=fixed_or_free_paramdict['fixed_params'][start:end]#end already includes +1 to make range inclusive of the end value
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='free':
            the_param_values=fixed_or_free_paramdict['free_params'][start:end]#end already includes +1 to make range inclusive of the end value 
        if fixed_or_free_paramdict[param_name]['const']=='non-neg':#transform variable with e^(.) if there is a non-negative constraint
            the_param_values=np.exp(the_param_values)
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
            #print('param_val',param_val)
            #print('param_form',param_form)
            assert param_val.ndim==1,"values for {} have not ndim==1".format(param_name)
            if param_form=='fixed':
                param_feature_dict['fixed_or_free']='fixed'
                param_feature_dict['const']='fixed'
                param_feature_dict['location_idx']=(len(fixed_params),len(fixed_params)+len(param_val))
                    #start and end indices, with end already including +1 to make python slicing inclusive of end in start:end
                fixed_params=np.concatenate([fixed_params,param_val],axis=0)
                fixed_or_free_paramdict[param_name]=param_feature_dict
            elif param_form=='free':
                param_feature_dict['fixed_or_free']='free'
                param_feature_dict['const']='free'
                param_feature_dict['location_idx']=(len(free_params),len(free_params)+len(param_val))
                    #start and end indices, with end already including +1 to make python slicing inclusive of end in start:end
                free_params=np.concatenate([free_params,param_val],axis=0)
                fixed_or_free_paramdict[param_name]=param_feature_dict
            else:
                param_feature_dict['fixed_or_free']='free'
                param_feature_dict['const']=param_form
                param_feature_dict['location_idx']=(len(free_params),len(free_params)+len(param_val))
                    #start and end indices, with end already including +1 to make python slicing inclusive of end in start:end
                free_params=np.concatenate([free_params,param_val],axis=0)
                fixed_or_free_paramdict[param_name]=param_feature_dict
        fixed_or_free_paramdict['free_params']='outside'
        fixed_or_free_paramdict['fixed_params'] = fixed_params

        return free_params,fixed_or_free_paramdict

    
    def makediffmat_itoj(self,xin,xpr):
        diffs= np.expand_dims(xin, axis=1) - np.expand_dims(xpr, axis=0)#should return ninXnoutXp if xin an xpr were ninXp and noutXp
        #print('type(diffs)=',type(diffs))
        return diffs



    def MY_KDE_gridprep_smalln(self,m,p):
        """creates a grid with all possible combinations of m=n^p (kerngrid not nin or nout) evenly spaced values from -3 to 3.
        """
        agrid=np.linspace(-3,3,m)[:,None] #assuming variables have been standardized
        pgrid=agrid.copy()
        for idx in range(p-1):
        #for idx in range(n-1):#-1 b/c agrid already created; need to figure out how to better vectorize this loop
            #pgrid=np.concatenate([np.repeat(pgrid,m,axis=0),np.repeat(agrid,m**(idx+1),axis=0)],axis=1)
            
            
            #agridtupple=(m**(idx+1),m)#starting with var=0, agrid tupple will be shape(2,n) which has 
            #pgridtupple=(m*m**(idx+1),pgrid.shape[1])
            #pgrid=np.concatenate(np.broadcast_to(agrid,agridtupple).ravel()[:,None],np.broadcast_to(pgrid,pgridtupple),axis=1)
            
            pgrid=np.concatenate([np.repeat(agrid,m**(idx+1),axis=0),np.tile(pgrid,[m,1])],axis=1)
            

        return pgrid

    def prep_out_grid(self,xkerngrid,ykerngrid,xdata_std,ydata_std,modeldict,xpr=None):
        '''#for small data, pre-create the 'grid'/out data
        no big data version for now
        '''
        # if self.n<10**5 and not (type(kerngrid)==int and kerngrid**self.p>10**8):
        #    self.data_is_small='yes'
        if xpr==None:
            xpr=xdata_std
            modeldict['predict_self_without_self']='yes'
        if not np.allclose(xpr,xdata_std):
            modeldict['predict_self_without_self']='n/a'
        if type(ykerngrid) is int and xkerngrid=="no":
            #yout=np.broadcast_to(np.linspace(-3,3,ykerngrid),(xdata_std.shape[0],ykerngrid))
            yout=np.linspace(-3,3,ykerngrid)#will broadcast later
            self.nout=ykerngrid
            #xpr=np.(np.tile(y_out,xdata_std.shape[0],axis=0))
            
        if type(xkerngrid) is int:#this maybe doesn't work yet
            self.nout=ykerngrid
            xpr=self.MY_KDE_gridprep_smalln(kerngrid,self.p)
            assert xpr.shape[1]==self.p,'xpr has wrong number of columns'
            assert xpr.shape[0]==kerngrid**self.p,'xpr has wrong number of rows'

            yxpr=self.MY_KDE_gridprep_smalln(kerngrid,self.p+1)
            assert yxpr.shape[1]==self.p+1,'yxpr has wrong number of columns'
            assert yxpr.shape[0]==kerngrid**(self.p+1),'yxpr has {} rows not {}'.format(yxpr.shape[0],kerngrid**(self.p+1))
            
        if xkerngrid=='no'and ykerngrid=='no':
            
            self.nout=self.nin
            yout=ydata_std
            #print('xprshape and yxprs.shape',xpr.shape,yxpr.shape)
            #yxpr=np.concatenate([ydata_std[None,:],xdata_std],axis=1)
            
        return xpr,yout

    def standardize_yx(self,xdata,ydata):
        self.xmean=np.mean(xdata,axis=0)
        self.ymean=np.mean(ydata,axis=0)
        self.xstd=np.std(xdata,axis=0)
        self.ystd=np.std(ydata,axis=0)
        standard_x=(xdata-self.xmean)/self.xstd
        standard_y=(ydata-self.ymean)/self.ystd
        return standard_x,standard_y


    def do_KDEsmalln(self,diffs,bw,modeldict):
        """estimate the density items in onediffs. collapse via products if dimensionality is greater than 2
        first 2 dimensions of onediffs must be ninXnout
        """
        assert diffs.shape==bw.shape, "diffs is shape:{} while bw is shape:{}".format(diffs.shape,bw.shape)
        #print('diffs.shape',diffs.shape,'diffs',diffs)
        #print('bw',bw)
        allkerns=self.gkernh(diffs, bw)
        #print(np.ma.count_masked(allkerns),'alkerns masked count with shape',allkerns.shape)
        #print('allkerns:',allkerns)
        #collapse by random variables indexed in last axis until allkerns.ndim=2
        second_to_last_axis=allkerns.ndim-2
        normalization=modeldict['product_kern_norm']
        if normalization =='self':
            allkerns_sum=np.ma.sum(allkerns,axis=second_to_last_axis)#this should be the nout axis
            #print('allkerns_sum.shape:',allkerns_sum.shape,allkerns_sum.max())               
            #assert allkerns.ndim==3,"allkerns has ndim:{} but expecting 3 by do_KDEsmalln".format(allkerns.ndim)
            allkerns=allkerns/np.broadcast_to(np.expand_dims(allkerns_sum,second_to_last_axis),allkerns.shape)
            # collapse just nin dim or both lhs dims?
        if normalization =="own_n":
            allkerns=allkerns/allkerns.count(axis=second_to_last_axis)#1 should be the nout axis
        if allkerns.ndim>3:
            #print('diffs',diffs)
            #print('bw',bw) 
            for i in range((allkerns.ndim-3),0,-1):
                assert allkerns.ndim>3, "allkerns is being collapsed via product on rhs " \
                                        "but has {} dimensions instead of ndim>3".format(allkerns.ndim)
                allkerns=np.ma.product(allkerns,axis=allkerns.ndim-1)#collapse right most dimension, so if the two items in the 3rd dimension\\
                #are kernels of x and y, we are creating the product kernel of x and y
        #assert allkerns.shape==(self.nin,self.nout), "allkerns is shaped{} not {} X {}".format(allkerns.shape,self.nin,self.nout)
        #above assert no longer relevant since x doesn't have nout as it's shape till compared to y
        #return allkerns
        #print('allkerns:',allkerns)
        #print('np.ma.count_masked(allkerns)',np.ma.count_masked(allkerns),'allkerns.shape',allkerns.shape)
        return np.ma.sum(allkerns,axis=0)/self.nin#collapsing across the nin kernels for each of nout    
        
    def MY_KDEpredictMSE(self,free_params,yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict):
        if not type(fixed_or_free_paramdict['free_params']) is dict: #it would be the string "outside" otherwise
            self.call_iter+=1#then it must be a new call during optimization
            if self.call_iter>1:
                print(f'iter:{self.call_iter},mse:{self.mselist[-1][0]}')
            
        fixed_or_free_paramdict['free_params']=free_params
        
        
        yhat_un_std=self.MY_KDEpredict(yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict)
        y_err=self.ydata-yhat_un_std
        mse= np.mean(np.power(y_err,2))
        self.mselist.append((mse,self.return_param_name_and_value(fixed_or_free_paramdict,modeldict)))
        
        #assert np.ma.count_masked(yhat_un_std)==0,"{}are masked in yhat of yhatshape:{}".format(np.ma.count_masked(yhat_un_std),yhat_un_std.shape)
        if not np.ma.count_masked(yhat_un_std)==0:
            mse=np.ma.count_masked(yhat_un_std)*10**199
        return mse
            
    def MY_KDEpredict(self,yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict):
        """moves free_params to first position of the obj function, preps data, and then runs MY_KDEreg to fit the model
            then returns MSE of the fit 
        Assumes last p elements of free_params are the scale parameters for 'el two' approach to
        columns of x.
        """
        #print('starting optimization of hyperparameters')
 
        #add/or_refresh free_params back into fixed_or_free_paramdict now that inside optimizer
        
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        #pull x_bandscale parameters from the appropriate location and appropriate vector
        x_bandscale_params=self.pull_value_from_fixed_or_free('x_bandscale',fixed_or_free_paramdict)
        y_bandscale_params=self.pull_value_from_fixed_or_free('y_bandscale',fixed_or_free_paramdict)
        p=x_bandscale_params.shape[0]
        assert self.p==p,\
            "p={} but x_bandscale_params.shape={}".format(self.p,x_bandscale_params.shape)


        if modeldict['Ndiff_bw_kern']=='rbfkern':
            
            xin_scaled=xin*x_bandscale_params
            xpr_scaled=xpr*x_bandscale_params
            xpr_scaled=xpr*x_bandscale_params
            #yxpr_scaled=yxpr*np.concatenate([np.array([1]),x_bandscale_params],axis=0)
            yin_scaled=yin*y_bandscale_params
            yout_scaled=yout*y_bandscale_params
            y_onediffs=self.makediffmat_itoj(yin_scaled,yout_scaled)
            y_Ndiffs=self.makediffmat_itoj(yin_scaled,yin_scaled)
            #print('y_Ndiffs_shape:{} and self.nout:{}'.format(y_Ndiffs.shape,self.nout))
            onediffs_scaled_l2norm=np.power(np.sum(np.power(self.makediffmat_itoj(xin_scaled,xpr_scaled),2),axis=2),.5)
            Ndiffs_scaled_l2norm=np.power(np.sum(np.power(self.makediffmat_itoj(xin_scaled,xin_scaled),2),axis=2),.5)
            assert onediffs_scaled_l2norm.shape==(xin.shape[0],xpr.shape[0]),'onediffs_scaled_l2norm does not have shape=(nin,nout)'

            diffdict={}
            diffdict['onediffs']=onediffs_scaled_l2norm
            diffdict['Ndiffs']=Ndiffs_scaled_l2norm
            ydiffdict={}
            ydiffdict['onediffs']=np.broadcast_to(y_onediffs[:,:,None],y_onediffs.shape+(self.npr,))
            ydiffdict['Ndiffs']=np.broadcast_to(y_Ndiffs[:,:,None],y_Ndiffs.shape+(self.npr,))
            diffdict['ydiffdict']=ydiffdict


        if modeldict['Ndiff_bw_kern']=='product':
            onediffs=makediffmat_itoj(xin,xpr)#scale now? if so, move if...='rbfkern' down 
            #predict
            yhat=MY_NW_KDEreg(yin_scaled,xin_scaled,xpr_scaled,yout_scaled,fixed_or_free_paramdict,diffdict,modeldict)
            #not developed yet

        # prepare the Ndiff bandwidth weights
        if modeldict['Ndiff_type'] == 'product':
            xbw = self.product_BWmaker(max_bw_Ndiff, self.Ndiff_list_of_masks_x, fixed_or_free_paramdict, diffdict, modeldict,'x')
            ybw = self.product_BWmaker(max_bw_Ndiff, self.Ndiff_list_of_masks_y, fixed_or_free_paramdict, diffdict['ydiffdict'],modeldict,'y')
        if modeldict['Ndiff_type'] == 'recursive':
            xbw = self.recursive_BWmaker(max_bw_Ndiff, self.Ndiff_list_of_masks_x, fixed_or_free_paramdict, diffdict, modeldict,'x')
            ybw = self.recursive_BWmaker(max_bw_Ndiff, self.Ndiff_list_of_masks_y, fixed_or_free_paramdict, diffdict['ydiffdict'],modeldict,'y')
        #extract and multiply ij varying part of bw times non varying part
        hx=self.pull_value_from_fixed_or_free('outer_x_bw', fixed_or_free_paramdict)
        hy=self.pull_value_from_fixed_or_free('outer_y_bw', fixed_or_free_paramdict)

                
        xbw=xbw*hx#need to make this flexible to blocks of x
        ybw=ybw*hy
        #print('xbw masked count:',np.ma.count_masked(xbw),'with shape:',xbw.shape)
        #print('ybw masked count:',np.ma.count_masked(ybw),'with shape:',ybw.shape)
        #print('xbw.shape',xbw.shape)
        #print('xbw:',xbw)
        #print('ybw.shape',ybw.shape)
        
        xonediffs=diffdict['onediffs']
        yonediffs=diffdict['ydiffdict']['onediffs']
        assert xonediffs.ndim==2, "xonediffs have ndim={} not 2".format(xonediffs.ndim)
        print(yonediffs.shape,xonediffs.shape)
        ykern_grid=modeldict['ykern_grid'];xkern_grid=modeldict['xkern_grid']
        #if ykern_grid=='no' and xkern_grid=='no':
        #    yx_onediffs_endstack=np.concatenate([yonediffs[:,:,None],xonediffs[:,:,None]],axis=2)
        #    yx_bw_endstack=np.ma.concatenate([ybw[:,:,None],xbw[:,:,None]],axis=2)
        if True:#type(ykern_grid) is int and xkern_grid=='no':
            xonedifftup=xonediffs.shape[:-1]+(self.nout,)+(xonediffs.shape[-1],)
            xonediffs_stack=np.broadcast_to(np.expand_dims(xonediffs,len(xonediffs.shape)-1),xonedifftup)
            xbw_stack=np.broadcast_to(np.expand_dims(xbw,len(xonediffs.shape)-1),xonedifftup)
        newaxis=len(yonediffs.shape)
        yx_onediffs_endstack=np.ma.concatenate((np.expand_dims(xonediffs_stack,newaxis),np.expand_dims(yonediffs,newaxis)),axis=newaxis)
        yx_bw_endstack=np.ma.concatenate((np.expand_dims(xbw_stack,newaxis),np.expand_dims(ybw,newaxis)),axis=newaxis)
        
        #print('xbw',xbw,'np.ma.count_masked(xbw)',np.ma.count_masked(xbw))
        #print('np.ma.count_masked(xonediffs)',np.ma.count_masked(xonediffs))
                        
        prob_x = self.do_KDEsmalln(xonediffs, xbw, modeldict)
        
        prob_yx = self.do_KDEsmalln(yx_onediffs_endstack, yx_bw_endstack,modeldict)#do_KDEsmalln implements product \\
            #kernel across axis=2, the 3rd dimension after the 2 diensions of onediffs. endstack refers to the fact \\
            #that y and x data are stacked in dimension 2 and do_kdesmall_n collapses them via the product of their kernels.
        
        #print('prob_x.shape:',prob_x.shape,'np.ma.count_masked(prob_x):',np.ma.count_masked(prob_x))
    
        
        #print('prob_yx.shape:',prob_yx.shape,'np.ma.count_masked(prob_yx)',np.ma.count_masked(prob_yx))
        #print('prob_x',prob_x)
        np.ma.count_masked(prob_x)
        
        
        if modeldict['regression_model']=='NW':
            yhat_raw = self.my_NW_KDEreg(prob_yx,prob_x,yout_scaled)
        self.yhat_std=yhat_raw*y_bandscale_params**-1#remove the effect of any parameters applied prior to using y.
        #here is the simple MSE objective function. however, I think I need to use
        #the more sophisticated MISE or mean integrated squared error,
        #either way need to replace with cost function function
        self.yhat_un_std=self.yhat_std*self.ystd+self.ymean
        return self.yhat_un_std


    def my_NW_KDEreg(self,prob_yx,prob_x,yout):
        """returns predited values of y for xpredict based on yin, xin, and modeldict
        """
        yout_axis=len(prob_yx.shape)-2#-2 b/c -1 for index form vs len count form and -1 b/c second to last dimensio is what we seek.
        
        prob_yx_sum=np.broadcast_to(np.expand_dims(np.ma.sum(prob_yx,axis=yout_axis),yout_axis),prob_yx.shape)
        cdfnorm_prob_yx=prob_yx/prob_yx_sum
        #cdfnorm_prob_yx=prob_yx#dropped normalization
        prob_x_sum=np.broadcast_to(np.expand_dims(np.ma.sum(prob_x, axis=yout_axis),yout_axis),prob_x.shape)
        cdfnorm_prob_x = prob_x / prob_x_sum
        
        #cdfnorm_prob_x = prob_x#dropped normalization
        #print(np.ma.count_masked(cdfnorm_prob_yx),'are masked in cdfnorm_prob_yx of shape:',cdfnorm_prob_yx.shape)
        #print(np.ma.count_masked(cdfnorm_prob_x),'are masked in cdfnorm_prob_x of shape:',cdfnorm_prob_x.shape)
        yout_stack=np.broadcast_to(np.expand_dims(yout,1),(self.nout,self.npr))
        prob_x_stack_tup=prob_x.shape[:-1]+(self.nout,)+(prob_x.shape[-1],)
        prob_x_stack=np.broadcast_to(np.expand_dims(cdfnorm_prob_x,yout_axis),prob_x_stack_tup)
        
        yhat= np.ma.sum(yout_stack*cdfnorm_prob_yx/prob_x_stack,axis=yout_axis)#sum over axis=0 collapses across nin for each nout
        #print(y_yxpr,cdfnorm_prob_yx/cdfnorm_prob_x)
        return yhat
    
    def predict_tool(self,xpr,fixed_or_free_paramdict,modeldict):
        """
        """
        xpr=(xpr-self.xmean)/self.xstd
        self.prediction=self.MY_KDEpredictMSE(fixed_or_free_paramdict['free_params'],self.yin,self.yout,self.xin,xpr,modeldict,fixed_or_free_paramdict)
        return self.prediciton.yhat  
    
class optimize_free_params(kNdtool):
    """"This is the method for iteratively running kernelkernel to optimize hyper parameters
    optimize dict contains starting values for free parameters, hyper-parameter structure(is flexible),
    and a model dict that describes which model to run including how hyper-parameters enter (quite flexible)
    speed and memory usage is a big goal when writing this. I pre-created masks to exclude the increasing
    list of centered data points. see mykern_core for an example and explanation of dictionaries.
    Flexibility is also a goal. max_bw_Ndiff is the deepest the model goes.
    ------------------
    attributes created
    self.n,self.p
    self.xdata,self.ydata contain the original data
    self.xdata_std, self.xmean,self.xstd
    self.ydata_std,self.ymean,self.ystd
    self.Ndiff - the nout X nin X p matrix of first differences of xdata_std
    self.Ndiff_list_of_masks - a list of progressively higher dimension (len=nin)
        masks to broadcast(views) Ndiff to.
    """

    def __init__(self,ydata,xdata,optimizedict):
        kNdtool.__init__(self)
        self.call_iter=0#one will be added to this each time the outer MSE function is called by scipy.minimize
        self.mselist=[]#will contain a tuple of  (mse, fixed_or_free_paramdict) at each call
        
        #extract optimization information, including modeling information in model dict,
        #parameter structure in model dict, and starting free parameter values from paramdict
        #these options are still not fully mature and actually flexible
        modeldict=optimizedict['model_dict'] #reserve _dict for names of dicts in *keys* of parent dicts
        model_param_formdict=modeldict['hyper_param_form_dict']
        xkerngrid=modeldict['xkern_grid']
        ykerngrid=modeldict['ykern_grid']
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        method=optimizedict['method']
        param_valdict=optimizedict['hyper_param_dict']
        #self.optdict=optimizedict
        
        #save and transform the data
        self.xdata=xdata;self.ydata=ydata
        self.nin,self.p=xdata.shape
        assert ydata.shape[0]==xdata.shape[0],'xdata.shape={} but ydata.shape={}'.format(xdata.shape,ydata.shape)

        #standardize x and y and save their means and std to self
        xdata_std,ydata_std=self.standardize_yx(xdata,ydata)
        #store the standardized (by column or parameter,p) versions of x and y
        self.xdata_std=xdata_std;self.ydata_std=ydata_std
        

        #create a list of free paramters for optimization  and
        # dictionary for keeping track of them and the list of fixed parameters too
        free_params,fixed_or_free_paramdict=self.setup_fixed_or_free(model_param_formdict,param_valdict)
        self.fixed_or_free_paramdict=fixed_or_free_paramdict
                                 
        xpr,yout=self.prep_out_grid(xkerngrid,ykerngrid,xdata_std,ydata_std,modeldict)
        self.xin=xdata_std;self.yin=ydata_std
        self.xpr=self.xin.copy()#xpr is x values used for prediction, which is the original data since we are optimizing.
        
        self.npr=xpr.shape[0]#probably redundant
        self.yout=yout


        #pre-build list of masks
        self.Ndiff_list_of_masks_y=self.max_bw_Ndiff_maskstacker_y(self.npr,self.nout,self.nin,self.p,max_bw_Ndiff,modeldict)
        self.Ndiff_list_of_masks_x=self.max_bw_Ndiff_maskstacker_x(self.npr,self.nout,self.nin,self.p,max_bw_Ndiff,modeldict)


        args_tuple=(self.yin,self.yout,self.xin,self.xpr,modeldict,fixed_or_free_paramdict)
        optiondict={
            'xatol':0.1,
            'fatol':0.1,
            'adaptive':True
        }
        self.mse=minimize(self.MY_KDEpredictMSE,free_params,args=args_tuple,method=method, options=optiondict)
        
        

if __name__=="_main__":
    pass