import multiprocessing
#import traceback
from copy import deepcopy
from typing import List
import os
from time import strftime, sleep
import datetime
import pickle
import numpy as np
#from numba import jit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import logging

#import logging.config
import yaml
import psutil

class kNdtool:
    """kNd refers to the fact that there will be kernels in kernels in these estimators

    """

    def __init__(self,savedir=None,myname=None):
        self.name=myname
        self.cores=int(psutil.cpu_count(logical=False)-1)
        #with open(os.path.join(os.getcwd(),'logconfig.yaml'),'rt') as f:
        #    configfile=yaml.safe_load(f.read())
        logging.basicConfig(level=logging.INFO)
        
        handlername=f'mykern-{self.name}.log'
        print(f'handlername:{handlername}')
        #below assumes it is a node if it has a name, so saving the node's log to the main cluster directory not the node's save directory
        if not self.name==None:
            handler=logging.FileHandler(os.path.join(savedir,'..',handlername))
        else:
            handler=logging.FileHandler(os.path.join(savedir,handlername))
        
        #self.logger = logging.getLogger('mkLogger')
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        
        if savedir==None:
            savedir=os.getcwd()
        self.savedirectory=savedir

    def sum_then_normalize_bw(self,kernstack,normalization):
        '''3 types of Ndiff normalization so far. could extend to normalize by other levels.
        '''
        if normalization=='none' or normalization==None:
            return np.ma.sum(kernstack,axis=0)

        if type(normalization) is int:
            
            return np.ma.sum(kernstack,axis=0)/float(normalization)
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
            #return np.ma.product(x_bandscale_params,np.ma.exp(-np.ma.power(maskeddata,2)),axis=maskeddata.ndim-1)/Ndiff_depth_bw_param
            shapetup=maskeddata.shape
            #the number of params should be equal to the length of the rightmost dimension, so broadcasting should match all the dimensions to the left according to shapetup.
            return self.gkernh(maskeddata,np.broadcast_to(x_bandscale_params,shapetup))
            
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
            ninmask=np.broadcast_to(np.eye(nin))[:,:,None],(nin,nin,npr)
        if self.predict_self_without_self=='yes' and nin==npr and ykerngrid=='no':
            ninmask3=np.broadcast_to(np.eye(nin)[:,:,None],(nin,nin,npr))
            ninmask2=np.broadcast_to(np.eye(nin)[:,None,:],(nin,nin,nin))
            ninmask1=np.broadcast_to(np.eye(nin)[None,:,:],(nin,nin,nin))
            ninmask1=np.ma.make_mask(ninmask1)
            ninmask2=np.ma.make_mask(ninmask2)
            ninmask3=np.ma.make_mask(ninmask3)
            ninmask=np.ma.mask_or(ninmask1,np.ma.mask_or(ninmask2,ninmask3))
        if self.predict_self_without_self=='yes' and nin==npr and type(ykerngrid) is int:
            ninmask=np.broadcast_to(np.eye(nin)[:,None,:],(nin,nin,nin))#nin not used to calculate npr
            ninmask=np.ma.make_mask(ninmask)
        list_of_masks=[ninmask]
        if max_bw_Ndiff>0:
            if ykerngrid=='no':
                firstdiffmask=np.ma.mask_or(np.ma.make_mask(np.broadcast_to(np.expand_dims(ninmask,0),(nin,nin,nin,npr))),np.ma.make_mask(np.broadcast_to(np.expand_dims(ninmask,2),(nin,nin,nin,npr))))
                firstdiffmask=np.ma.mask_or(np.ma.make_mask(np.broadcast_to(np.expand_dims(ninmask,1),(nin,nin,nin,npr))),firstdiffmask)
                #when 0 dim of ninmask is expanded, masked if i=j for all k.
                #when 2 dim of nin mask is expanded, masked if k=j for all i, and when 1 dim of nin mask is expanded, masked if k=i for all j. all are for all ii
            if type(ykerngrid) is int:
                firstdiffmask=np.ma.make_mask(np.broadcast_to(np.expand_dims(ninmask,2),(nin,nin,nout,npr)))
                #if yout is a grid and nout not equal to nin, 
            list_of_masks.append(firstdiffmask)# this line is indexed (k,j,i,ii)
                
        if max_bw_Ndiff>1:
            
            for ii in range(max_bw_Ndiff-1):#-1 b/c 1diff masks already in second position of list of masks if max_bw_Ndiff>0
                lastmask=list_of_masks[-1] #second masks always based on self.nin
                masktup=(nin,)+lastmask.shape#expand dimensions on lhs
                list_of_masks.append(np.ma.make_mask(np.broadcast_to(np.expand_dims(lastmask,0),masktup)))
                for iii in range(ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                    #then this loop maxes at 0,1 from range(2-1+1)
                    iii+=1 #since 0 dim added before for_loop
                    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.ma.make_mask(np.broadcast_to(np.expand_dims(lastmask,axis=iii),masktup)))
                if ykerngrid=='no':
                    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.ma.make_mask(np.broadcast_to(np.expand_dims(lastmask,axis=ii+3),masktup)))#this should mask \
                    #the yout values from the rest since yout==yin        
        if type(ykerngrid) is int:
            list_of_masks[0]=np.ma.make_mask(np.zeros([nin,nout,npr]))#overwrite first item in list of masks to remove masking when predicting y using ykerngrid==int
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
            list_of_masks.append(np.ma.make_mask(np.broadcast_to(ninmask[:,:,None],(nin,nin,npr))))
        if self.predict_self_without_self=='yes' and nin==npr and max_bw_Ndiff>0:
                ninmask3=np.broadcast_to(ninmask[:,:,None],(nin,nin,nin))
                ninmask2=np.broadcast_to(ninmask[:,None,:],(nin,nin,nin))
                ninmask1=np.broadcast_to(ninmask[None,:,:],(nin,nin,nin))
                ninmask1=np.ma.make_mask(ninmask1)
                ninmask2=np.ma.make_mask(ninmask2)
                ninmask3=np.ma.make_mask(ninmask3)
                list_of_masks.append(np.ma.mask_or(ninmask1,np.ma.mask_or(ninmask2,ninmask3)))
                        
        if max_bw_Ndiff>1:
            for ii in range(max_bw_Ndiff-1):#-1 b/c 1diff masks already in second position of list of masks if max_bw_Ndiff>0
                lastmask=list_of_masks[-1] #second masks always based on self.nin
                masktup=(nin,)+lastmask.shape#expand dimensions on lhs
                list_of_masks.append(np.ma.make_mask(np.broadcast_to(np.expand_dims(lastmask,0),masktup)))
                for iii in range(ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                    #then this loop maxes at 0,1 from range(2-1+1)
                    iii+=1 #since 0 dim added before for_loop
                    list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.ma.make_mask(np.broadcast_to(np.expand_dims(lastmask,axis=iii),masktup)))
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
        #pgrid=agrid.copy()
        for idx in range(p-1):
            pgrid=np.concatenate([np.repeat(agrid,m**(idx+1),axis=0),np.tile(pgrid,[m,1])],axis=1)
            #outtup=()
            #pgrid=np.broadcast_to(np.linspace(-3,3,m),)
        return pgrid

    def prep_out_grid(self,xkerngrid,ykerngrid,xdata_std,ydata_std,modeldict,xpr=None):
        '''#for small data, pre-create the 'grid'/out data
        no big data version for now
        '''
        if modeldict['regression_model']=='logistic':
            if type(ykerngrid) is int:
                print(f'overriding modeldict:ykerngrid:{ykerngrid} to {"no"} b/c logisitic regression')
                ykerngrid='no'
        ykerngrid_form=modeldict['ykerngrid_form']
        if xpr==None:
            xpr=xdata_std
            self.predict_self_without_self='yes'
        if not np.allclose(xpr,xdata_std):
            self.predict_self_without_self='n/a'
        if type(ykerngrid) is int and xkerngrid=="no":
            yout=self.generate_grid(ykerngrid_form,ykerngrid)#will broadcast later
            self.nout=ykerngrid
        if type(xkerngrid) is int:#this maybe doesn't work yet
            self.logger.warning("xkerngrid is not fully developed")
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
    
    def generate_grid(self,form,count):
        if form[0]=='even':
            gridrange=form[1]
            return np.linspace(-gridrange,gridrange,count)
        if form[0]=='exp':
            assert count%2==1,f'ykerngrid(={count}) must be odd for ykerngrid_form:exp'
            gridrange=form[1]
            log_gridrange=np.log(gridrange+1)
            log_grid=np.linspace(0,log_gridrange,(count+2)//2)
            halfgrid=np.exp(log_grid[1:])-1
            return np.concatenate((-halfgrid[::-1],np.array([0]),halfgrid),axis=0)
            
    
    def standardize_yx(self,xdata,ydata):
        self.xmean=np.mean(xdata,axis=0)
        self.ymean=np.mean(ydata,axis=0)
        self.xstd=np.std(xdata,axis=0)
        self.ystd=np.std(ydata,axis=0)
        standard_x=(xdata-self.xmean)/self.xstd
        standard_y=(ydata-self.ymean)/self.ystd
        return standard_x,standard_y

    def standardize_yxtup(self,yxtup_list,val_yxtup_list=None):
        #yxtup_list=deepcopy(yxtup_list_unstd)
        all_y=[ii for i in yxtup_list for ii in i[0]]
        all_x=[ii for i in yxtup_list for ii in i[1]]
        self.xmean=np.mean(all_x,axis=0)
        self.ymean=np.mean(all_y,axis=0)
        self.xstd=np.std(all_x,axis=0)
        self.ystd=np.std(all_y,axis=0)
        tupcount=len(yxtup_list)#should be same as batchcount
        yxtup_list_std=[]
        for i in range(tupcount):
            ystd=(yxtup_list[i][0] - self.ymean) / self.ystd
            xstd=(yxtup_list[i][1] - self.xmean) / self.xstd
            yxtup_list_std.append((ystd,xstd))
        if not val_yxtup_list==None:
            val_yxtup_list_std=[]
            val_tupcount=len(val_yxtup_list)
            for i in range(val_tupcount):
                val_ystd=(val_yxtup_list[i][0] - self.ymean) / self.ystd
                val_xstd=(val_yxtup_list[i][1] - self.xmean) / self.xstd
                val_yxtup_list_std.append((val_ystd,val_xstd))
        else: 
            val_yxtup_list_std=None
        return yxtup_list_std,val_yxtup_list_std


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
            allkerns=allkerns/self.ma_broadcast_to(np.ma.expand_dims(allkerns_sum,second_to_last_axis),allkerns.shape)
            
            # collapse just nin dim or both lhs dims?
        if normalization =="own_n":
            allkerns=allkerns/np.ma.expand_dims(np.ma.count(allkerns,axis=second_to_last_axis),second_to_last_axis)#1 should be the nout axis
        if modeldict['regression_model']=='NW-rbf' or modeldict['regression_model']=='NW-rbf2':
            if allkerns.ndim>3:
                for i in range((allkerns.ndim-3),0,-1):
                    assert allkerns.ndim>3, "allkerns is being collapsed via rbf on rhs " \
                                            "but has {} dimensions instead of ndim>3".format(allkerns.ndim)
                    allkerns=np.ma.power(np.ma.sum(np.ma.power(allkerns,2),axis=allkerns.ndim-1),0.5)#collapse right most dimension, so if the two items in the 3rd dimension\\
        if modeldict['regression_model']=='NW':
            if allkerns.ndim>3:
                for i in range((allkerns.ndim-3),0,-1):
                    assert allkerns.ndim>3, "allkerns is being collapsed via product on rhs " \
                                            "but has {} dimensions instead of ndim>3".format(allkerns.ndim)
                    allkerns=np.ma.product(allkerns,axis=allkerns.ndim-1)#collapse right most dimension, so if the two items in the 3rd dimension\\
        return np.ma.sum(allkerns,axis=0)/self.nin#collapsing across the nin kernels for each of nout    
        
    def ma_broadcast_to(self, maskedarray,tup):
            initial_mask=np.ma.getmask(maskedarray)
            broadcasted_mask=np.broadcast_to(initial_mask,tup)
            broadcasted_array=np.broadcast_to(maskedarray,tup)
            return np.ma.array(broadcasted_array, mask=broadcasted_mask)
            
    def sort_then_saveit(self,mse_param_list,modeldict,filename):
        
        fullpath_filename=os.path.join(self.savedirectory,filename)
        mse_list=[i[0] for i in mse_param_list]
        minmse=min(mse_list)
        fof_param_dict_list=[i[1] for i in mse_param_list]
        bestparams=fof_param_dict_list[mse_list.index(minmse)]
        savedict={}
        savedict['mse']=minmse
        #savedict['xdata']=self.xdata
        #savedict['ydata']=self.ydata
        savedict['params']=bestparams
        savedict['modeldict']=modeldict
        now=strftime("%Y%m%d-%H%M%S")
        savedict['when_saved']=now
        savedict['datagen_dict']=self.datagen_dict
        try:#this is only relevant after optimization completes
            savedict['minimize_obj']=self.minimize_obj
        except:
            pass
        for i in range(10):
            try: 
                with open(fullpath_filename,'rb') as modelfile:
                    modellist=pickle.load(modelfile)
                break
            except FileNotFoundError:
                modellist=[]
                break
            except:
                sleep(0.1)
                if i==9:
                    self.logger.exception(f'error in {__name__}')
                    modellist=[]
        #print('---------------success----------')
        if len(modellist)>0:
            lastsavetime=modellist[-1]['when_saved']
            runtime=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(lastsavetime,"%Y%m%d-%H%M%S")
            print(f'time between saves for {self.name} is {runtime}')
        modellist.append(savedict)
        
        for i in range(10):
            try:
                with open(fullpath_filename,'wb') as thefile:
                    pickle.dump(modellist,thefile)
                print(f'saved to {fullpath_filename} at about {strftime("%Y%m%d-%H%M%S")} with mse={minmse}')
                break
            except:
                if i==9:
                    print(f'mykern.py could not save to {fullpath_filename} after {i+1} tries')
        return
    
    
    def MY_KDEpredict(self,yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict):
        """moves free_params to first position of the obj function, preps data, and then runs MY_KDEreg to fit the model
            then returns MSE of the fit 
        Assumes last p elements of free_params are the scale parameters for 'el two' approach to
        columns of x.
        """
        
        try:
            lossfn=modeldict['loss_function']
        except KeyError:
            lossfn='mse'
        iscrossmse=lossfn[0:8]=='crossmse'
        
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
            yhat=self.MY_NW_KDEreg(yin_scaled,xin_scaled,xpr_scaled,yout_scaled,fixed_or_free_paramdict,diffdict,modeldict)[0]
            #not developed yet
        
        xbw = self.BWmaker(max_bw_Ndiff, fixed_or_free_paramdict, diffdict, modeldict,'x')
        ybw = self.BWmaker(max_bw_Ndiff, fixed_or_free_paramdict, diffdict['ydiffdict'],modeldict,'y')
        '''xbwmaskcount=np.ma.count_masked(xbw)
        print('xbwmaskcount',xbwmaskcount)
        print('np.ma.getmask(xbw)',np.ma.getmask(xbw))'''
        
        '''ybwmaskcount=np.ma.count_masked(ybw)
        print('ybwmaskcount',ybwmaskcount)
        print('np.ma.getmask(ybw)',np.ma.getmask(ybw))'''

        hx=self.pull_value_from_fixed_or_free('outer_x_bw', fixed_or_free_paramdict)
        hy=self.pull_value_from_fixed_or_free('outer_y_bw', fixed_or_free_paramdict)

                
        xbw=xbw*hx#need to make this flexible to blocks of x
        if modeldict['regression_model']=='logistic':
            xonediffs=diffdict['onediffs']
            prob_x = self.do_KDEsmalln(xonediffs, xbw, modeldict)
            
            yhat_tup=self.kernel_logistic(prob_x,xin,yin)
            yhat_std=yhat_tup[0]
            cross_errors=yhat_tup[1]
            
        if modeldict['regression_model'][0:2]=='NW':
            ybw=ybw*hy
            xonediffs=diffdict['onediffs']
            yonediffs=diffdict['ydiffdict']['onediffs']
            assert xonediffs.ndim==2, "xonediffs have ndim={} not 2".format(xonediffs.ndim)
            ykern_grid=modeldict['ykern_grid'];xkern_grid=modeldict['xkern_grid']
            if True:#type(ykern_grid) is int and xkern_grid=='no':
                xonedifftup=xonediffs.shape[:-1]+(self.nout,)+(xonediffs.shape[-1],)
                xonediffs_stack=np.broadcast_to(np.expand_dims(xonediffs,len(xonediffs.shape)-1),xonedifftup)
                xbw_stack=np.broadcast_to(np.ma.expand_dims(xbw,len(xonediffs.shape)-1),xonedifftup)
            newaxis=len(yonediffs.shape)
            yx_onediffs_endstack=np.ma.concatenate((np.expand_dims(xonediffs_stack,newaxis),np.expand_dims(yonediffs,newaxis)),axis=newaxis)
            yx_bw_endstack=np.ma.concatenate((np.ma.expand_dims(xbw_stack,newaxis),np.ma.expand_dims(ybw,newaxis)),axis=newaxis)
            prob_x = self.do_KDEsmalln(xonediffs, xbw, modeldict)
            prob_yx = self.do_KDEsmalln(yx_onediffs_endstack, yx_bw_endstack,modeldict)#do_KDEsmalln implements product \\
                #kernel across axis=2, the 3rd dimension after the 2 diensions of onediffs. endstack refers to the fact \\
                #that y and x data are stacked in dimension 2 and do_kdesmall_n collapses them via the product of their kernels.
        
            KDEregtup = self.my_NW_KDEreg(prob_yx,prob_x,yout_scaled,modeldict)
            yhat_raw=KDEregtup[0]
            cross_errors=KDEregtup[1]
            yhat_std=yhat_raw*y_bandscale_params**-1#remove the effect of any parameters applied prior to using y.
        #here is the simple MSE objective function. however, I think I need to use
        #the more sophisticated MISE or mean integrated squared error,
        #either way need to replace with cost function function
        yhat_un_std=yhat_std*self.ystd+self.ymean
        
        #print(f'yhat_un_std:{yhat_un_std}')
        if lossfn=='mse':
            return yhat_un_std
        if iscrossmse:
            return yhat_un_std,cross_errors*self.ystd
        
    def kernel_logistic(self,prob_x,xin,yin):
        lossfn=modeldict['loss_function']
        iscrossmse=lossfn[0:8]=='crossmse'
                      
        for i in range(prob_x.shape[-1]):
            xin_const=np.concatenate(np.ones((xin.shape[0],1),xin,axis=1))
            yhat_i=LogisticRegression().fit(xin_const,yin,prob_x[...,i]).predict(xin)
            yhat_std.extend(yhat_i[i])
            cross_errors.extend(yhat_i)#list with ii on dim0
        cross_errors=np.masked_array(cross_errors,mask=np.eye(yin.shape[0])).T#to put ii back on dim 1
        yhat=np.array(yhat_std)                             
        if not iscrossmse:
            return (yhat,None)
        if iscrossmse:
            if len(lossfn)>8:
                cross_exp=float(lossfn[8:])
                wt_stack=prob_x**cross_exp
            
            cross_errors=(yhat[None,:]-yout[:,None])#this makes dim0=nout,dim1=nin
            crosswt_stack=wt_stack/np.ma.expand_dims(np.ma.sum(wt_stack,axis=1),axis=1)
            wt_cross_errors=np.ma.sum(crosswt_stack*cross_errors,axis=1)#weights normalized to sum to 1, then errors summed to 1 per nin
            return (yhat,wt_cross_errors)

    def my_NW_KDEreg(self,prob_yx,prob_x,yout,modeldict):
        """returns predited values of y for xpredict based on yin, xin, and modeldict
        """
        lossfn=modeldict['loss_function']
        iscrossmse=lossfn[0:8]=='crossmse'
            
        yout_axis=len(prob_yx.shape)-2#-2 b/c -1 for index form vs len count form and -1 b/c second to last dimensio is what we seek.
        
        #prob_yx_sum=np.broadcast_to(np.ma.expand_dims(np.ma.sum(prob_yx,axis=yout_axis),yout_axis),prob_yx.shape)
        #cdfnorm_prob_yx=prob_yx/prob_yx_sum
        #cdfnorm_prob_yx=prob_yx#dropped normalization
        #prob_x_sum=np.broadcast_to(np.ma.expand_dims(np.ma.sum(prob_x, axis=yout_axis),yout_axis),prob_x.shape)
        #cdfnorm_prob_x = prob_x / prob_x_sum
        #cdfnorm_prob_x = prob_x#dropped normalization
        
        yout_stack=self.ma_broadcast_to(np.ma.expand_dims(yout,1),(self.nout,self.npr))
        prob_x_stack_tup=prob_x.shape[:-1]+(self.nout,)+(prob_x.shape[-1],)
        prob_x_stack=self.ma_broadcast_to(np.ma.expand_dims(prob_x,yout_axis),prob_x_stack_tup)
        NWnorm=modeldict['NWnorm']
                
        if modeldict['regression_model']=='NW-rbf2':
            wt_stack=np.ma.power(np.ma.power(prob_yx,2)-np.ma.power(prob_x_stack,2),0.5)
            if NWnorm=='across':
                wt_stack=wt_stack/np.ma.expand_dims(np.ma.sum(wt_stack,axis=1),axis=1)
            yhat=np.ma.sum(yout_stack*wt_stack,axis=yout_axis)
        else:
            wt_stack=prob_yx/prob_x_stack
            if NWnorm=='across':
                wt_stack=wt_stack/np.ma.expand_dims(np.ma.sum(wt_stack,axis=1),axis=1)
            yhat=np.ma.sum(yout_stack*wt_stack,axis=yout_axis)#sum over axis=0 collapses across nin for each nout
            yhatmaskscount=np.ma.count_masked(yhat)
            if yhatmaskscount>0:print('in my_NW_KDEreg, yhatmaskscount:',yhatmaskscount)
        #print(f'yhat:{yhat}')
        
        if not iscrossmse:
            return (yhat,None)
        if iscrossmse:
            if len(lossfn)>8:
                cross_exp=float(lossfn[8:])
                wt_stack=wt_stack**cross_exp
            
            cross_errors=(yhat[None,:]-yout[:,None])#this makes dim0=nout,dim1=nin
            crosswt_stack=wt_stack/np.ma.expand_dims(np.ma.sum(wt_stack,axis=1),axis=1)
            wt_cross_errors=np.ma.sum(crosswt_stack*cross_errors,axis=1)#weights normalized to sum to 1, then errors summed to 1 per nin
            return (yhat,wt_cross_errors)
    
    def predict_tool(self,xpr,fixed_or_free_paramdict,modeldict):
        """
        """
        xpr=(xpr-self.xmean)/self.xstd
        self.prediction=self.MY_KDEpredictMSE(fixed_or_free_paramdict['free_params'],self.yin,self.yout,self.xin,xpr,modeldict,fixed_or_free_paramdict)
        return self.prediction.yhat

    def MY_KDEpredictMSE(self, free_params, batchdata_dict, modeldict, fixed_or_free_paramdict):

        if not type(fixed_or_free_paramdict['free_params']) is list:  # it would be the string "outside" otherwise
            self.call_iter += 1  # then it must be a new call during optimization

        batchcount = self.datagen_dict['batchcount']
        #print(f'batchcount:{batchcount}')
        fixed_or_free_paramdict['free_params'] = free_params
        # print(f'free_params added to dict. free_params:{free_params}')

        try:
            lossfn=modeldict['loss_function']
        except KeyError:
            print(f'loss_function not found in modeldict')
            lossfn='mse'
        iscrossmse=lossfn[0:8]=='crossmse'

        y_err_tup = ()

        arglistlist=[]
        for batch_i in range(batchcount):
            arglist=[]
            arglist.append(batchdata_dict['yintup'][batch_i])
            arglist.append(batchdata_dict['youttup'][batch_i])
            arglist.append(batchdata_dict['xintup'][batch_i])
            arglist.append(batchdata_dict['xprtup'][batch_i])

            arglist.append(modeldict)
            arglist.append(fixed_or_free_paramdict)
            arglistlist.append(arglist)

        process_count=1#self.cores
        if process_count>1 and batchcount>1:
            with multiprocessing.Pool(processes=process_count) as pool:
                yhat_unstd=pool.map(self.MPwrapperKDEpredict,arglistlist)
                sleep(2)
                pool.close()
                pool.join()
        else:
            yhat_unstd=[]
            for i in range(batchcount):
                yhat_unstd.append(self.MPwrapperKDEpredict(arglistlist[i]))
        if iscrossmse:
            yhat_unstd,crosserrors=zip(*yhat_unstd)
        #print(f'after mp.pool,yhat_unstd has shape:{np.shape(yhat_unstd)}')
        for batch_i in range(batchcount):
            y_batch_i=self.datagen_obj.yxtup_list[batch_i][0]#the original y data is a list of tupples
            y_err = y_batch_i - yhat_unstd[batch_i]
            y_err_tup = y_err_tup + (y_err,)

        all_y_err = np.ma.concatenate(y_err_tup,axis=0)
        
        #print('all_y_err',all_y_err)
        if iscrossmse:
            all_y_err=np.ma.concatenate([all_y_err,np.ravel(crosserrors)],axis=0)
        mse = np.ma.mean(np.ma.power(all_y_err, 2))
        maskcount=np.ma.count_masked(all_y_err)
        assert maskcount==0,print(f'{maskcount} masked values found in all_y_err')
            #mse = np.ma.count_masked(all_y_err) * 10000*maskcount
        self.mse_param_list.append((mse, deepcopy(fixed_or_free_paramdict)))
        # self.return_param_name_and_value(fixed_or_free_paramdict,modeldict)
        self.fixed_or_free_paramdict = fixed_or_free_paramdict
        t_format = "%Y%m%d-%H%M%S"
        self.iter_start_time_list.append(strftime(t_format))

        if self.call_iter == 3:
            tdiff = np.abs(
                datetime.datetime.strptime(self.iter_start_time_list[-1], t_format) - datetime.datetime.strptime(
                    self.iter_start_time_list[-2], t_format))
            self.save_interval = int(max([15 - np.round(np.log(tdiff.total_seconds() + 1) ** 3, 0),
                                          1]))  # +1 to avoid negative and max to make sure save_interval doesn't go below 1
            print(f'save_interval changed to {self.save_interval}')

        if self.call_iter % self.save_interval == 0:
            self.sort_then_saveit(self.mse_param_list[-self.save_interval * 2:], modeldict, 'model_save')

        # assert np.ma.count_masked(yhat_un_std)==0,"{}are masked in yhat of yhatshape:{}".format(np.ma.count_masked(yhat_un_std),yhat_un_std.shape)

        return mse

    def MPwrapperKDEpredict(self,arglist):
        #print(f'arglist inside wrapper is:::::::{arglist}')
        yin=arglist[0]
        yout=arglist[1]
        xin=arglist[2]
        xpr=arglist[3]
        modeldict=arglist[4]
        fixed_or_free_paramdict=arglist[5]
        return self.MY_KDEpredict(yin, yout, xin, xpr, modeldict, fixed_or_free_paramdict)
        


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

    def __init__(self,datagen_obj,optimizedict,savedir=None,myname=None):
        self.name=myname
        if savedir==None:
              mydir=os.getcwd()
        kNdtool.__init__(self,savedir=savedir,myname=myname)
        self.datagen_obj=datagen_obj
        self.call_iter=0#one will be added to this each time the outer MSE function is called by scipy.minimize
        self.mse_param_list=[]#will contain a tuple of  (mse, fixed_or_free_paramdict) at each call
        self.iter_start_time_list=[]
        self.save_interval=1
        self.datagen_dict=optimizedict['datagen_dict']
        
        #Extract from optimizedict
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
        #self.xdata=datagen_obj.x;self.ydata=datagen_obj.y#this is just the first of the batches, if batchcount>1
        self.batchcount=datagen_obj.batchcount
        self.nin=datagen_obj.batch_n
        self.p=datagen_obj.param_count#p should work too
        #assert self.ydata.shape[0]==self.xdata.shape[0],'xdata.shape={} but ydata.shape={}'.format(xdata.shape,ydata.shape)

        #standardize x and y and save their means and std to self
        yxtup_list_std,val_yxtup_list_std = self.standardize_yxtup(datagen_obj.yxtup_list,datagen_obj.val_yxtup_list)
        
        #store the standardized (by column or parameter,p) versions of x and y
        #self.xdata_std=xdata_std;self.ydata_std=ydata_std
                                 
        #xpr,yout=self.prep_out_grid(xkerngrid,ykerngrid,xdata_std,ydata_std,modeldict)
        #self.xin=xdata_std;self.yin=ydata_std
        #self.xpr=self.xin.copy()#xpr is x values used for prediction, which is the original data since we are optimizing.
        self.npr=self.nin#since we are optimizing within our sample

        #load up the data for each batch into a dictionary full of tuples
        # with each tuple item containing data for a batch from 0 to batchcount-1
        xintup = ()
        yintup = ()
        xprtup = ()
        youttup = ()
        for i in range(self.batchcount):
            xdata_std=yxtup_list_std[i][1]
            ydata_std=yxtup_list_std[i][0]
            xpri,youti=self.prep_out_grid(xkerngrid,ykerngrid,xdata_std,ydata_std,modeldict)
            xintup=xintup+(xdata_std,)
            yintup=yintup+(ydata_std,)
            xprtup=xprtup+(xpri,)
            youttup=youttup+(youti,)

        batchdata_dict={'xintup':xintup,'yintup':yintup,'xprtup':xprtup,'youttup':youttup}
        #print('=======================')
        #print(f'batchdata_dict{batchdata_dict}')
        #print('=======================')
        #self.npr=xpr.shape[0]#probably redundant
        #self.yout=yout

        #pre-build list of masks
        self.Ndiff_list_of_masks_y=self.max_bw_Ndiff_maskstacker_y(self.npr,self.nout,self.nin,self.p,max_bw_Ndiff,modeldict)
        self.Ndiff_list_of_masks_x=self.max_bw_Ndiff_maskstacker_x(self.npr,self.nout,self.nin,self.p,max_bw_Ndiff,modeldict)
        
        #setup and run scipy minimize
        args_tuple=(batchdata_dict, modeldict, fixed_or_free_paramdict)
        print(f'mykern modeldict:{modeldict}')
        self.minimize_obj=minimize(self.MY_KDEpredictMSE, free_params, args=args_tuple, method=method, options=opt_method_options)
        
        lastmse=self.mse_param_list[-1][0]
        lastparamdict=self.mse_param_list[-1][1]
        self.sort_then_saveit([[lastmse,lastparamdict]],modeldict,'model_save')
        #self.sort_then_saveit(self.mse_param_list[-self.save_interval*3:],modeldict,'final_model_save')
        self.sort_then_saveit(self.mse_param_list,modeldict,'final_model_save')
        print(f'lastparamdict:{lastparamdict}')
        

if __name__ == "__main__":

    import os
    import kernelcompare as kc
    import traceback
    import mykern

    # from importlib import reload
    networkdir = 'o:/public/dpatton/kernel'
    mydir = os.getcwd()
    test = kc.KernelCompare(directory=mydir)

    Ndiff_type_variations = ('modeldict:Ndiff_type', ['recursive', 'product'])
    max_bw_Ndiff_variations = ('modeldict:max_bw_Ndiff', [2])
    Ndiff_start_variations = ('modeldict:Ndiff_start', [1, 2])
    ykern_grid_variations = ('ykern_grid', [49])
    # product_kern_norm_variations=('modeldict:product_kern_norm',['self','own_n'])#include None too?
    # normalize_Ndiffwtsum_variations=('modeldict:normalize_Ndiffwtsum',['own_n','across'])
    optdict_variation_list = [Ndiff_type_variations, max_bw_Ndiff_variations,
                              Ndiff_start_variations]  # ,product_kern_norm_variations,normalize_Ndiffwtsum_variations]

    # the default datagen_dict as of 11/25/2019
    # datagen_dict={'batch_n':32,'batchcount':10, 'param_count':param_count,'seed':1, 'ftype':'linear', 'evar':1, 'source':'monte'}
    batch_n_variations = ('batch_n', [32])
    batchcount_variations = ('batchcount', [8])
    ftype_variations = ('ftype', ['linear', 'quadratic'])
    param_count_variations = ('param_count', [1, 2])
    datagen_variation_list = [batch_n_variations, batchcount_variations, ftype_variations, param_count_variations]
    testrun = test.prep_model_list(optdict_variation_list=optdict_variation_list,
                                   datagen_variation_list=datagen_variation_list, verbose=1)

    from random import shuffle

    #shuffle(testrun)
    # a_rundict=testrun[100]#this produced the Ndiff_exponent error for recursive Ndiff
    for idx in range(len(testrun)):
        print(f'~~~~~~~run number:{idx}`~~~~~~~')
        a_rundict = testrun[idx]
        print(f'a_rundict{a_rundict}')
        optimizedict = a_rundict['optimizedict']
        datagen_dict = a_rundict['datagen_dict']

        try:
            test.do_monte_opt(optimizedict, datagen_dict, force_start_params=0)
            test.open_condense_resave('model_save', verbose=0)
            test.merge_and_condense_saved_models(merge_directory=None, save_directory=None, condense=1, verbose=0)
        except:
            print('traceback for run', idx)
            self.logger.exception(f'error in {__name__}')
