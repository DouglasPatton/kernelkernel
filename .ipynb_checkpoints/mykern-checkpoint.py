from copy import deepcopy
from typing import List
import os
from time import strftime
import datetime
import pickle
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
        if normalization=='none' or normalization==None:
            return np.ma.sum(kernstack,axis=0)

        if type(normalization) is int:
            return np.ma.sum(kernstack,axis=0)/normalization
        if normalization=='across':
            #return np.ma.sum(kernstack/np.ma.mean(kernstack,axis=0),axis=0)
            this_depth_sum=np.ma.sum(kernstack,axis=0)
            return this_depth_sum/np.ma.sum(this_depth_sum,axis=0)#dividing by sum across the sums at "this_depth"

    def recursive_BWmaker(self, max_bw_Ndiff, Ndiff_list_of_masks, fixed_or_free_paramdict, diffdict, modeldict, x_or_y):
        """returns an nin X nout npr np.array of bandwidths if x_or_y=='y'
        or nin X npr if x_or_y=='x'
        """
        Ndiff_exponent_params = self.pull_value_from_fixed_or_free('Ndiff_exponent', fixed_or_free_paramdict)
        Ndiff_depth_bw_params = self.pull_value_from_fixed_or_free('Ndiff_depth_bw', fixed_or_free_paramdict)
        max_bw_Ndiff = modeldict['max_bw_Ndiff']
        Ndiff_bw_kern = modeldict['Ndiff_bw_kern']
        normalization = modeldict['normalize_Ndiffwtsum']
        ykern_grid = modeldict['ykern_grid']

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
    
    def Ndiff_recursive(self,masked_data,deeper_bw,Ndiff_exp,Ndiff_bw,Ndiff_bw_kern,normalize):
        return np.ma.power(
                    self.sum_then_normalize_bw(
                        self.do_bw_kern(Ndiff_bw_kern, masked_data,deeper_bw),normalize),Ndiff_exp                   )
    
    def Ndiff_product(self,masked_data,deeper_bw,Ndiff_exp,Ndiff_bw,Ndiff_bw_kern,normalize):
        return np.ma.power(
            self.sum_then_normalize_bw(
                self.do_bw_kern(Ndiff_bw_kern,masked_data,Ndiff_bw)*deeper_bw,normalize),Ndiff_exp)
    
    def BWmaker(self,max_bw_Ndiff,fixed_or_free_paramdict,diffdict,modeldict,x_or_y):
        """returns an nin X nout npr np.array of bandwidths if x_or_y=='y'
        ? or nin X npr if x_or_y=='x' ?
        """
        Ndiff_exponent_params = self.pull_value_from_fixed_or_free('Ndiff_exponent', fixed_or_free_paramdict)
        Ndiff_depth_bw_params = self.pull_value_from_fixed_or_free('Ndiff_depth_bw', fixed_or_free_paramdict)
        Ndiff_bw_kern = modeldict['Ndiff_bw_kern']
        normalization = modeldict['normalize_Ndiffwtsum']
        Ndiff_start=modeldict['Ndiff_start']      
        Ndiffs=diffdict['Ndiffs']
        onediffs=diffdict['onediffs']
        ykern_grid = modeldict['ykern_grid']
        Ndiff_type=modeldict['Ndiff_type']
        if x_or_y=='y':
            masklist=self.Ndiff_list_of_masks_y
        if x_or_y=='x':
            masklist=self.Ndiff_list_of_masks_x
            
        if Ndiff_bw_kern=='rbfkern': #parameter column of x already collapsed
            #if type(ykern_grid) is int and Ndiff_start>1:
                #max_bw_Ndiff-=1
                #missing_i_dimension=1
                
            #else: missing_i_dimension=0
            if x_or_y=='y':
                this_depth_bw=np.ones([self.nin,self.nout,self.npr])
            if x_or_y=='x':
                this_depth_bw=np.ones([self.nin,1,self.npr])#x doesn't vary over nout like y does, so just 1 for a dimension placeholder.
            
            deeper_depth_bw=np.array([1])
            for depth in range(max_bw_Ndiff,0,-1):
                if normalization == 'own_n':
                    normalize=self.nin-(depth)
                else:normalize=normalization
                param_idx=depth-1-(Ndiff_start-1)#if Ndiff_start>1, there are fewer parameters than depths
                if Ndiff_type=='product':this_depth_bw_param=Ndiff_depth_bw_params[param_idx]
                if Ndiff_type=='recursive':this_depth_bw_param=None
                this_depth_exponent=Ndiff_exponent_params[param_idx]
                this_depth_data=self.Ndiff_datastacker(Ndiffs,onediffs.shape,depth)
                this_depth_mask = masklist[depth]
                if depth % 2 == 0:  # every other set of Ndiffs is transposed
                    #print(f'this_depth_data.ndim:{this_depth_data.ndim}')
                    dimcount=this_depth_data.ndim
                    transposelist=[i for i in range(dimcount)]
                    transposelist[dimcount-3]=dimcount-4#make 3rd to last dimension have the dimension number of 4th to last
                    transposelist[dimcount - 4] = dimcount - 3#and make 4th to last dimension have dimension number of 3rd to last
                    this_depth_data = np.transpose(this_depth_data, transposelist)#implement the tranpose of 3rd to last and 2nd to last dimensions
                    this_depth_mask = np.transpose(this_depth_mask,transposelist)

                
                if Ndiff_start>1:# the next few lines collapse the length of Ndiff dimensions before Ndiff start down to lenght 1, but preserves the dimension
                    select_dims=list((slice(None),)*this_depth_mask.ndim)#slice(None) is effectively a colon when the list is turned into a tuple of dimensions
                    for dim in range(Ndiff_start-1,0,-1):
                        shrinkdim=max_bw_Ndiff-dim
                        select_dims[shrinkdim]=[0,]
                    dim_select_tup=tuple(select_dims)
                    this_depth_mask=this_depth_mask[dim_select_tup]
                    this_depth_data=this_depth_data[dim_select_tup]
                this_depth_masked_data=np.ma.array(this_depth_data,mask=this_depth_mask)
                if depth<Ndiff_start:
                    this_depth_bw_param=1
                    this_depth_exponent=1
                    normalize=1
                if Ndiff_type=='product':
                    this_depth_bw=self.Ndiff_product(this_depth_masked_data,deeper_depth_bw,this_depth_exponent,this_depth_bw_param,Ndiff_bw_kern,normalize)
                if Ndiff_type=='recursive':
                    if depth==max_bw_Ndiff:deeper_depth_bw=Ndiff_depth_bw_params[0]
                    
                    this_depth_bw=self.Ndiff_recursive(this_depth_masked_data,deeper_depth_bw,this_depth_exponent,this_depth_bw_param,Ndiff_bw_kern,normalize)
                    
                if depth>1: deeper_depth_bw=this_depth_bw#setup deeper_depth_bw for next iteration if there is another
            '''if missing_i_dimension==1:
                dimcount=len(this_depth_bw.shape)
                print(f'this_depth_bw.shape:{this_depth_bw.shape}')
                full_tuple=tuple(list(this_depth_bw.shape).insert(dimcount-3,self.nin))#
                full_tuple
                np.broadcast_to(this_depth_bw,full_tuple)
            '''
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
        denom=1/((2*np.pi)**.5*h)
        numerator=np.ma.exp(-np.ma.power(x/(h*2), 2))
        kern=denom*numerator
        return kern
        

    
    
    def Ndiff_datastacker(self,Ndiffs,onediffs_shape,depth):
        """
        """
        if len(onediffs_shape)==3:#this should only happen if we're working on y
            ytup=(self.nin,)*depth+onediffs_shape#depth-1 b/c Ndiffs starts as ninXninXnpr
            if depth==0:return Ndiffs
            return np.broadcast_to(np.expand_dims(Ndiffs,2),ytup)
        if len(onediffs_shape)==2:#this should only happen if we're working on x
            Ndiff_shape_out_tup=(self.nin,)*depth+onediffs_shape
            return np.broadcast_to(Ndiffs,Ndiff_shape_out_tup)#no dim exp b/c only adding to lhs of dim tuple
    
    def max_bw_Ndiff_maskstacker_y(self,npr,nout,nin,p,max_bw_Ndiff,modeldict):
        #print('nout:',nout)
        Ndiff_bw_kern=modeldict['Ndiff_bw_kern']
        ykerngrid=modeldict['ykern_grid']
        #ninmask=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,:,None],(nin,nin,npr))
        
        if not self.predict_self_without_self=='yes':
            ninmask=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,:,None],(nin,nin,npr))
        if self.predict_self_without_self=='yes' and nin==npr and ykerngrid=='no':
            ninmask3=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,:,None],(nin,nin,npr))
            ninmask2=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,None,:],(nin,nin,nin))
            ninmask1=np.broadcast_to(np.ma.make_mask(np.eye(nin))[None,:,:],(nin,nin,nin))
            ninmask=np.ma.mask_or(ninmask1,np.ma.mask_or(ninmask2,ninmask3))
        if self.predict_self_without_self=='yes' and nin==npr and type(ykerngrid) is int:
            ninmask=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,None,:],(nin,nin,nin))#nin not used to calculate npr
        list_of_masks=[ninmask]
        if max_bw_Ndiff>0:
            if ykerngrid=='no':
                firstdiffmask=np.ma.mask_or(np.broadcast_to(np.expand_dims(ninmask,0),(nin,nin,nin,npr)),np.broadcast_to(np.expand_dims(ninmask,2),(nin,nin,nin,npr)))
                firstdiffmask=np.ma.mask_or(np.broadcast_to(np.expand_dims(ninmask,1),(nin,nin,nin,npr)),firstdiffmask)
                #when 0 dim of ninmask is expanded, masked if i=j for all k.
                #when 2 dim of nin mask is expanded, masked if k=j for all i, and when 1 dim of nin mask is expanded, masked if k=i for all j. all are for all ii
            if type(ykerngrid) is int:
                firstdiffmask=np.broadcast_to(np.expand_dims(ninmask,2),(nin,nin,nout,npr))
                #if yout is a grid and nout not equal to nin, 
            list_of_masks.append(firstdiffmask)# this line is indexed (k,j,i,ii)
                
        if max_bw_Ndiff>1:
            
            for ii in range(max_bw_Ndiff-1):#-1 b/c 1diff masks already in second position of list of masks if max_bw_Ndiff>0
                lastmask=list_of_masks[-1] #second masks always based on self.nin
                masktup=(nin,)+lastmask.shape#expand dimensions on lhs
                list_of_masks.append(np.broadcast_to(np.expand_dims(lastmask,0),masktup))
                for iii in range(ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                    #then this loop maxes at 0,1 from range(2-1+1)
                    iii+=1 #since 0 dim added before for_loop
                    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.broadcast_to(np.expand_dims(lastmask,axis=iii),masktup))
                if ykerngrid=='no':
                    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.broadcast_to(np.expand_dims(lastmask,axis=ii+3),masktup))#this should mask \
                    #the yout values from the rest since yout==yin        
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
        if not self.predict_self_without_self=='yes' and max_bw_Ndiff>0:#first mask will be corrected at the bottom
            list_of_masks.append(np.broadcast_to(ninmask[:,:,None],(nin,nin,npr)))
        if self.predict_self_without_self=='yes' and nin==npr and max_bw_Ndiff>0:
                ninmask3=np.broadcast_to(ninmask[:,:,None],(nin,nin,nin))
                ninmask2=np.broadcast_to(ninmask[:,None,:],(nin,nin,nin))
                ninmask1=np.broadcast_to(ninmask[None,:,:],(nin,nin,nin))
                list_of_masks.append(np.ma.mask_or(ninmask1,np.ma.mask_or(ninmask2,ninmask3)))
                        
        if max_bw_Ndiff>1:
            for ii in range(max_bw_Ndiff-1):#-1 b/c 1diff masks already in second position of list of masks if max_bw_Ndiff>0
                lastmask=list_of_masks[-1] #second masks always based on self.nin
                masktup=(nin,)+lastmask.shape#expand dimensions on lhs
                list_of_masks.append(np.broadcast_to(np.expand_dims(lastmask,0),masktup))
                for iii in range(ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                    #then this loop maxes at 0,1 from range(2-1+1)
                    iii+=1 #since 0 dim added before for_loop
                    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.broadcast_to(np.expand_dims(lastmask,axis=iii),masktup))
            lastmask=list_of_masks[-1]#copy the last item to lastmask
        if not self.predict_self_without_self=='yes':
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
    
    
    def pull_value_from_fixed_or_free(self,param_name,fixed_or_free_paramdict,transform=None):
        if transform==None:
            transform=1
        if transform=='no':
            transform=0
        start,end=fixed_or_free_paramdict[param_name]['location_idx']
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='fixed':
            the_param_values=fixed_or_free_paramdict['fixed_params'][start:end]#end already includes +1 to make range inclusive of the end value
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='free':
            the_param_values=fixed_or_free_paramdict['free_params'][start:end]#end already includes +1 to make range inclusive of the end value
        if transform==1:
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
        print(f'param_valdict:{param_valdict}')
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
        
        print(f'setup_fixed_or_free_paramdict:{fixed_or_free_paramdict}')
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
            self.predict_self_without_self='yes'
        if not np.allclose(xpr,xdata_std):
            self.predict_self_without_self='n/a'
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
        allkerns=self.gkernh(diffs, bw)
        second_to_last_axis=allkerns.ndim-2
        normalization=modeldict['product_kern_norm']
        if normalization =='self':
            allkerns_sum=np.ma.sum(allkerns,axis=second_to_last_axis)#this should be the nout axis
            allkerns=allkerns/np.broadcast_to(np.expand_dims(allkerns_sum,second_to_last_axis),allkerns.shape)
            # collapse just nin dim or both lhs dims?
        if normalization =="own_n":
            allkerns=allkerns/allkerns.count(axis=second_to_last_axis)#1 should be the nout axis
        if allkerns.ndim>3:
            for i in range((allkerns.ndim-3),0,-1):
                assert allkerns.ndim>3, "allkerns is being collapsed via product on rhs " \
                                        "but has {} dimensions instead of ndim>3".format(allkerns.ndim)
                allkerns=np.ma.product(allkerns,axis=allkerns.ndim-1)#collapse right most dimension, so if the two items in the 3rd dimension\\
        return np.ma.sum(allkerns,axis=0)/self.nin#collapsing across the nin kernels for each of nout    
        
    def MY_KDEpredictMSE(self,free_params,yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict):
        
            
        if not type(fixed_or_free_paramdict['free_params']) is list: #it would be the string "outside" otherwise
            self.call_iter+=1#then it must be a new call during optimization
            #if self.call_iter>1 and self.call_iter%5==0:
            #    print(f'iter:{self.call_iter},mse:{self.mse_param_list[-1]}')
            #if self.call_iter>1:# and self.call_iter%5>0:
            #    print(f'iter:{self.call_iter} mse:{self.mse_param_list[-1][0]}',end=',')
            
            
        fixed_or_free_paramdict['free_params']=free_params
        #print(f'free_params added to dict. free_params:{free_params}')
        
        
        yhat_un_std=self.MY_KDEpredict(yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict)
        y_err=self.ydata-yhat_un_std
        mse= np.mean(np.power(y_err,2))
        self.mse_param_list.append((mse,deepcopy(fixed_or_free_paramdict)))
        #self.return_param_name_and_value(fixed_or_free_paramdict,modeldict)
        self.fixed_or_free_paramdict=fixed_or_free_paramdict
        t_format="%Y%m%d-%H%M%S"
        self.iter_start_time_list.append(strftime(t_format))
        
        if self.call_iter==3:
            
            tdiff=np.abs(datetime.datetime.strptime(self.iter_start_time_list[-1],t_format)-datetime.datetime.strptime(self.iter_start_time_list[-2],t_format))
            self.save_interval= int(max([15-np.round(np.log(tdiff.total_seconds()+1)**3,0),1]))#+1 to avoid negative and max to make sure save_interval doesn't go below 1
            print(f'save_interval changed to {self.save_interval}')
            
        
        if self.call_iter%self.save_interval==0:
            self.sort_then_saveit(self.mse_param_list[-self.save_interval*2:],modeldict,'model_save')
                
        #assert np.ma.count_masked(yhat_un_std)==0,"{}are masked in yhat of yhatshape:{}".format(np.ma.count_masked(yhat_un_std),yhat_un_std.shape)
        if not np.ma.count_masked(yhat_un_std)==0:
            mse=np.ma.count_masked(yhat_un_std)*10**199
        
        return mse
            
    def sort_then_saveit(self,mse_param_list,modeldict,filename):
        
        mse_list=[i[0] for i in mse_param_list]
        minmse=min(mse_list)
        fof_param_dict_list=[i[1] for i in mse_param_list]
        bestparams=fof_param_dict_list[mse_list.index(minmse)]
        savedict={}
        savedict['mse']=minmse
        savedict['xdata']=self.xdata #inherit data description?
        savedict['ydata']=self.ydata
        savedict['params']=bestparams
        savedict['modeldict']=modeldict
        savedict['when_saved']=strftime("%Y%m%d-%H%M%S")
        savedict['datagen_dict']=self.datagen_dict
        try:
            savedict['minimize_obj']=self.minimize_obj
        except:
            pass
        try:
            with open(filename,'rb') as modelfile:
                modellist=pickle.load(modelfile)
                #print('---------------success----------')
        except:
            modellist=[]
        modellist.append(savedict)
        with open(filename,'wb') as thefile:
            pickle.dump(modellist,thefile)
        print(f'saved to {filename} at about {strftime("%Y%m%d-%H%M%S")} with mse={minmse}')
    
    
    def MY_KDEpredict(self,yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict):
        """moves free_params to first position of the obj function, preps data, and then runs MY_KDEreg to fit the model
            then returns MSE of the fit 
        Assumes last p elements of free_params are the scale parameters for 'el two' approach to
        columns of x.
        """

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
            yin_scaled=yin*y_bandscale_params
            yout_scaled=yout*y_bandscale_params
            y_onediffs=self.makediffmat_itoj(yin_scaled,yout_scaled)
            y_Ndiffs=self.makediffmat_itoj(yin_scaled,yin_scaled)
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

        xbw = self.BWmaker(max_bw_Ndiff, fixed_or_free_paramdict, diffdict, modeldict,'x')
        ybw = self.BWmaker(max_bw_Ndiff, fixed_or_free_paramdict, diffdict['ydiffdict'],modeldict,'y')

        hx=self.pull_value_from_fixed_or_free('outer_x_bw', fixed_or_free_paramdict)
        hy=self.pull_value_from_fixed_or_free('outer_y_bw', fixed_or_free_paramdict)

                
        xbw=xbw*hx#need to make this flexible to blocks of x
        ybw=ybw*hy
        xonediffs=diffdict['onediffs']
        yonediffs=diffdict['ydiffdict']['onediffs']
        assert xonediffs.ndim==2, "xonediffs have ndim={} not 2".format(xonediffs.ndim)
        ykern_grid=modeldict['ykern_grid'];xkern_grid=modeldict['xkern_grid']
        if True:#type(ykern_grid) is int and xkern_grid=='no':
            xonedifftup=xonediffs.shape[:-1]+(self.nout,)+(xonediffs.shape[-1],)
            xonediffs_stack=np.broadcast_to(np.expand_dims(xonediffs,len(xonediffs.shape)-1),xonedifftup)
            xbw_stack=np.broadcast_to(np.expand_dims(xbw,len(xonediffs.shape)-1),xonedifftup)
        newaxis=len(yonediffs.shape)
        yx_onediffs_endstack=np.ma.concatenate((np.expand_dims(xonediffs_stack,newaxis),np.expand_dims(yonediffs,newaxis)),axis=newaxis)
        yx_bw_endstack=np.ma.concatenate((np.expand_dims(xbw_stack,newaxis),np.expand_dims(ybw,newaxis)),axis=newaxis)
        prob_x = self.do_KDEsmalln(xonediffs, xbw, modeldict)
        prob_yx = self.do_KDEsmalln(yx_onediffs_endstack, yx_bw_endstack,modeldict)#do_KDEsmalln implements product \\
            #kernel across axis=2, the 3rd dimension after the 2 diensions of onediffs. endstack refers to the fact \\
            #that y and x data are stacked in dimension 2 and do_kdesmall_n collapses them via the product of their kernels.
            
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
        yout_stack=np.broadcast_to(np.expand_dims(yout,1),(self.nout,self.npr))
        prob_x_stack_tup=prob_x.shape[:-1]+(self.nout,)+(prob_x.shape[-1],)
        prob_x_stack=np.broadcast_to(np.expand_dims(cdfnorm_prob_x,yout_axis),prob_x_stack_tup)
        
        yhat= np.ma.sum(yout_stack*cdfnorm_prob_yx/prob_x_stack,axis=yout_axis)#sum over axis=0 collapses across nin for each nout
        return yhat
    
    def predict_tool(self,xpr,fixed_or_free_paramdict,modeldict):
        """
        """
        xpr=(xpr-self.xmean)/self.xstd
        self.prediction=self.MY_KDEpredictMSE(fixed_or_free_paramdict['free_params'],self.yin,self.yout,self.xin,xpr,modeldict,fixed_or_free_paramdict)
        return self.prediction.yhat  
    
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
    self.Ndiff_list_of_masks - a list of progressively higher dimension (len=nin)
        masks to broadcast(views) Ndiff to.
    """

    def __init__(self,ydata,xdata,optimizedict):
        
        kNdtool.__init__(self)
        self.call_iter=0#one will be added to this each time the outer MSE function is called by scipy.minimize
        self.mse_param_list=[]#will contain a tuple of  (mse, fixed_or_free_paramdict) at each call
        self.iter_start_time_list=[]
        self.save_interval=1
        self.datagen_dict=optimizedict['datagen_dict']
        #Extract from outer optimizedict
        modeldict=optimizedict['modeldict'] 
        opt_settings_dict=optimizedict['opt_settings_dict']
        param_valdict=optimizedict['hyper_param_dict']
        
        method=opt_settings_dict['method']
        opt_method_options=opt_settings_dict['options']
        '''mse_threshold=opt_settings_dict['mse_threshold']
        inherited_mse=optimizedict['mse']
        if inherited_mse<mse_threshold:
            print(f'optimization halted because inherited mse:{inherited_mse}<mse_threshold:{mse_threshold}')
            return'''
        
        model_param_formdict=modeldict['hyper_param_form_dict']
        xkerngrid=modeldict['xkern_grid']
        ykerngrid=modeldict['ykern_grid']
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        
        #build dictionary for parameters
        free_params,fixed_or_free_paramdict=self.setup_fixed_or_free(model_param_formdict,param_valdict)
        self.fixed_or_free_paramdict=fixed_or_free_paramdict
                
        #save and transform the data
        self.xdata=xdata;self.ydata=ydata
        self.nin,self.p=xdata.shape
        assert ydata.shape[0]==xdata.shape[0],'xdata.shape={} but ydata.shape={}'.format(xdata.shape,ydata.shape)

        #standardize x and y and save their means and std to self
        xdata_std,ydata_std=self.standardize_yx(xdata,ydata)
        #store the standardized (by column or parameter,p) versions of x and y
        self.xdata_std=xdata_std;self.ydata_std=ydata_std
                                 
        xpr,yout=self.prep_out_grid(xkerngrid,ykerngrid,xdata_std,ydata_std,modeldict)
        self.xin=xdata_std;self.yin=ydata_std
        self.xpr=self.xin.copy()#xpr is x values used for prediction, which is the original data since we are optimizing.
        
        self.npr=xpr.shape[0]#probably redundant
        self.yout=yout

        #pre-build list of masks
        self.Ndiff_list_of_masks_y=self.max_bw_Ndiff_maskstacker_y(self.npr,self.nout,self.nin,self.p,max_bw_Ndiff,modeldict)
        self.Ndiff_list_of_masks_x=self.max_bw_Ndiff_maskstacker_x(self.npr,self.nout,self.nin,self.p,max_bw_Ndiff,modeldict)
        
        #setup and run scipy minimize
        args_tuple=(self.yin, self.yout, self.xin, self.xpr, modeldict, fixed_or_free_paramdict)
        print(f'modeldict:{modeldict}')
        self.minimize_obj=minimize(self.MY_KDEpredictMSE, free_params, args=args_tuple, method=method, options=opt_method_options)
        
        lastmse=self.mse_param_list[-1][0]
        lastparamdict=self.mse_param_list[-1][1]
        self.sort_then_saveit([[lastmse,lastparamdict]],modeldict,'model_save')
        #self.sort_then_saveit(self.mse_param_list[-self.save_interval*3:],modeldict,'final_model_save')
        self.sort_then_saveit(self.mse_param_list,modeldict,'final_model_save')
        print(f'lastparamdict:{lastparamdict}')
        
        
        

if __name__=="_main__":
    pass