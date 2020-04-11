import os
#os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import logging,logging.config

import psutil
import yaml

class Ndiff:
    def __init__(self,):
        pass
            
        
    def Ndiff_datastacker(self,indiffs,outdiffs,depth):
        """
            for x: (x has it's nout dimension (-2) added later on)
            before:
            outdiffs is #ninXnpr(Xbatchcount)
            indiffs is #ninXnin(Xbatchcount)
            after:
            both are {depth*nin}XninXnprv
            
            for y:
            before:
            outdiffs is broadcast to dim2 npr times #ninXnoutXnpr(Xbatchcount)?
            indiffs is broadcast to dim2 npr times #ninXninXnpr(Xbatchcount)?
            after:
            outdiffs becomes {depth*nin}XninXnoutXnpr(Xbatchcount)
        """
        
        
        
        
        if len(outdiffs.shape)==4:#3:#(ninXnoutXnpr(Xbatchcount))this should only happen if we're working on y
            #
            shape_out_tup=tuple([self.nin for _ in range(depth)])+outdiffs.shape
            #expand_tup=tuple(range(depth)+2)
            if depth>1:
                if depth%2==0:
                    indiffs=np.transpose(indiffs,[1,0,2,3])
                expanded_indiffs=indiffs
                for _ in range(depth): 
                    expanded_indiffs=np.expand_dims(expanded_indiffs,axis=2)    # there is variation over dims 0,1 for all dims to the rhs of them
                    
                #shape_out_tup=tuple([self.nin for _ in range(depth)])+indiffs.shape
                return np.broadcast_to(expanded_indiffs,shape_out_tup)#indiffs starts as ninxninxnpr, expand_dims adds a dimension for nout
            else:
                
                
                
                return np.broadcast_to(outdiffs,shape_out_tup)
        if len(outdiffs.shape)==3:#2:#(ninXnpr(Xbatchcount))this should only happen if we're working on x
            shape_out_tup=tuple([self.nin for _ in range(depth)])+outdiffs.shape
            if depth>1:
                if depth%2==0:
                    indiffs=np.transpose(indiffs,[1,0,2])
                expanded_indiffs=indiffs
                for _ in range(depth): 
                    expanded_indiffs=np.expand_dims(expanded_indiffs,axis=2)   
                #shape_out_tup=tuple([self.nin for _ in range(depth)])+indiffs.shape
                return np.broadcast_to(expanded_indiffs,shape_out_tup)#indiffs starts as ninxnin, expand_dims adds a dimension for npr
            else:
                
                return np.broadcast_to(outdiffs,shape_out_tup)
            
        

    def Ndiffsum_then_normalize_bw(self,kernstack,normalization,depth,x_or_y):
        '''3 types of Ndiff normalization so far. could extend to normalize by other levels.
        '''
        # the next 7 lines are the alternative to having transposition in the datastacker.
        #for sum_axis:
        # if depth is 1, y:-3 and x:-2 # depth 1 is outdiffs
        # if depth is 2, y:-4  and x:-3 # these are the applied to indiffs
        # if depth is 3, y:-3 and x:-2
        # if depth is 4, y:-4 and x:-3
        """dropping these because moving transposition back to the datastacker to keep variation in dims 0,1 where we will sum
        This approach would have worked, but it seems would have required a de transpose
        if x_or_y=='y':
            if depth==1:
                sum_axis=-3
            elif  depth%2==0:
                sum_axis=-4
            else:
                sum_axis=-3
        else: 
            if depth==1:
                sum_axis=-2
            elif  depth%2==0:
                sum_axis=-3
            else:
                sum_axis=-2
        """
        try:
            sum_axis=0    
            #self.logger.info(f'kernstack.shape:{kernstack.shape}, depth: {depth}, x_or_y:{x_or_y}, sum_axis:{sum_axis}')
            if normalization=='none' or normalization==None:
                return np.sum(kernstack,axis=sum_axis)

            if normalization=='own_n':

                return np.mean(kernstack,axis=sum_axis)
            if normalization=='across':
                #return np.sum(kernstack/np.mean(kernstack,axis=0),axis=0)
                this_depth_sum=np.sum(kernstack,axis=sum_axis)
                return this_depth_sum/np.sum(this_depth_sum,axis=0)#dividing by sum across the sums at "this_depth"
                #need to think about the axis of this normalization
        except FloatingPointError:
            self.nperror=1
        except:
            self.logger.exception('')
            assert False,'unexpected error'

        
    def Ndiff_recursive(self,masked_data,deeper_bw,Ndiff_exp,Ndiff_bw,Ndiff_bw_kern,normalize,depth,x_or_y):
        try:
            result=np.power(
                        self.Ndiffsum_then_normalize_bw(
                            self.do_Ndiffbw_kern(Ndiff_bw_kern, masked_data,deeper_bw),normalize,depth,x_or_y),Ndiff_exp)
            return result
        except FloatingPointError:
            self.nperror=1
            return
        except:
            self.logger.exception('')
            assert False,'unexpected error'
    
    def Ndiff_product(self,masked_data,deeper_bw,Ndiff_exp,Ndiff_bw,Ndiff_bw_kern,normalize,depth,x_or_y):
        try:
            result=np.power(
                self.Ndiffsum_then_normalize_bw(
                    self.do_Ndiffbw_kern(Ndiff_bw_kern,masked_data,Ndiff_bw)*deeper_bw,normalize,depth,x_or_y),Ndiff_exp)
        except FloatingPointError:
            self.nperror=1
            return
        except:
            self.logger.exception('')
            assert False,'unexpected error'
    
    
    def NdiffBWmaker(self,max_bw_Ndiff,fixed_or_free_paramdict,diffdict,modeldict,x_or_y):
        """returns an nin X nout npr np.array of bandwidths if x_or_y=='y'
        ? or nin X npr if x_or_y=='x' ?
        """
        try:
            Ndiff_exponent_params = self.pull_value_from_fixed_or_free('Ndiff_exponent', fixed_or_free_paramdict)
            Ndiff_depth_bw_params = self.pull_value_from_fixed_or_free('Ndiff_depth_bw', fixed_or_free_paramdict)
            Ndiff_bw_kern = modeldict['Ndiff_bw_kern']
            normalization = modeldict['normalize_Ndiffwtsum']
            Ndiff_start=modeldict['Ndiff_start']      
            indiffs=diffdict['indiffs']
            outdiffs=diffdict['outdiffs']
            ykern_grid = modeldict['ykern_grid']
            Ndiff_type=modeldict['Ndiff_type']

            if Ndiff_bw_kern=='rbfkern': #parameter column of x already collapsed
                #if type(ykern_grid) is int and Ndiff_start>1:
                    #max_bw_Ndiff-=1
                    #missing_i_dimension=1

                #else: missing_i_dimension=0
                if x_or_y=='y':
                    this_depth_bw=np.ones([self.nin,self.nout,self.npr,self.batchcount])
                if x_or_y=='x':
                    this_depth_bw=np.ones([self.nin,1,self.npr,self.batchcount])#x doesn't vary over nout like y does, so just 1 for a dimension placeholder.

                deeper_depth_bw=np.array([1])
                for depth in range(max_bw_Ndiff,0,-1):
                    if normalization == 'own_n':
                        normalize='own_n'
                    else:normalize=normalization
                    if depth<Ndiff_start:
                        this_depth_bw_param=1
                        this_depth_exponent=1
                        normalize=1
                    else:
                        param_idx=depth-1-(Ndiff_start-1)#if Ndiff_start>1, there are fewer parameters than depths
                        if Ndiff_type=='product':this_depth_bw_param=Ndiff_depth_bw_params[param_idx]
                        this_depth_exponent=Ndiff_exponent_params[param_idx]
                    if Ndiff_type=='recursive':this_depth_bw_param=None
                    this_depth_data=self.Ndiff_datastacker(indiffs,outdiffs,depth)


                    '''if Ndiff_start>1:# the next few lines collapse the length of Ndiff dimensions before Ndiff start down to lenght 1, but preserves the dimension
                        select_dims=list((slice(None),)*this_depth_mask.ndim)#slice(None) is effectively a colon when the list is turned into a tuple of dimensions
                        for dim in range(Ndiff_start-1,0,-1):
                            shrinkdim=max_bw_Ndiff-dim
                            select_dims[shrinkdim]=[0,]
                        dim_select_tup=tuple(select_dims)
                        this_depth_mask=this_depth_mask[dim_select_tup]
                        this_depth_data=this_depth_data[dim_select_tup]'''

                    if Ndiff_type=='product':
                        this_depth_bw=self.Ndiff_product(this_depth_data,deeper_depth_bw,this_depth_exponent,this_depth_bw_param,Ndiff_bw_kern,normalize,depth,x_or_y)
                    if Ndiff_type=='recursive':
                        if depth==max_bw_Ndiff:deeper_depth_bw=Ndiff_depth_bw_params[0]
                        this_depth_bw=self.Ndiff_recursive(this_depth_data,deeper_depth_bw,this_depth_exponent,this_depth_bw_param,Ndiff_bw_kern,normalize,depth,x_or_y)

                    if depth>0: deeper_depth_bw=this_depth_bw#setup deeper_depth_bw for next iteration if there is another

                return this_depth_bw

            if Ndiff_bw_kern=='product': #outdiffs parameter column not yet collapsed
                n_depth_masked_sum_kern=self.do_Ndiffbw_kern(Ndiff_bw_kern,n_depth_masked_sum,Ndiff_depth_bw_params[depth],x_bandscale_params)
        except FloatingPointError:
            self.nperror=1
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
                
    
    
    
    def do_Ndiffbw_kern(self,kern_choice,maskeddata,Ndiff_depth_bw_param,x_bandscale_params=None):
        try:
            if kern_choice=="product":
                #return np.ma.product(x_bandscale_params,np.ma.exp(-np.ma.power(maskeddata,2)),axis=maskeddata.ndim-1)/Ndiff_depth_bw_param
                shapetup=maskeddata.shape
                #the number of params should be equal to the length of the rightmost dimension, so broadcasting should match all the dimensions to the left according to shapetup.
                return self.gkernh(maskeddata,np.broadcast_to(x_bandscale_params,shapetup))

            if kern_choice=='rbfkern':
                return self.gkernh(maskeddata, Ndiff_depth_bw_param)#parameters already collapsed, so this will be rbf
        except FloatingPointError:
            self.nperror=1
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
    
    def gkernh(self, x, h):
        "returns the rbf/gaussian kernel at x with bandwidth h"
        try:
            #amask=np.ma.getmask(x)
            kern=np.exp(-np.power(x,2)/(np.power(h,2)*2))
            #kern=np.ma.masked_array(np.nan_to_num(kern,copy=False),mask=amask)
            return kern#
        except FloatingPointError:
            self.nperror=1
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
    
    
    
    
    
    
    def Ndiffdo_KDEsmalln(self,diffs,bw,modeldict):
        """estimate the density items in outdiffs. collapse via products if dimensionality is greater than 2
        first 2 dimensions of outdiffs must be ninXnout
        """ 
        try:
            assert diffs.shape==bw.shape, "diffs is shape:{} while bw is shape:{}".format(diffs.shape,bw.shape)
            allkerns=self.gkernh(diffs, bw)
            normalization=modeldict['product_kern_norm']
            if normalization =='self':
                allkerns_sum=np.sum(allkerns,axis=1)#from lhs, axes are nin,nout,npr,batchcount and an extra if y and x are stacked
                allkerns=allkerns/np.broadcast_to(np.expand_dims(allkerns_sum,1),allkerns.shape)
                #allkerns_sum=np.sum(allkerns,axis=-3)#-2)#this should be the nout axis
                #allkerns=allkerns/np.broadcast_to(np.expand_dims(allkerns_sum,-3),allkerns.shape) # -2 to 3 since new rhs batchcount dim

                # collapse just nin dim or both lhs dims?
            if normalization =="own_n":
                allkerns=allkerns/self.nout#1 should be the nout axis
            if modeldict['regression_model']=='NW-rbf' or modeldict['regression_model']=='NW-rbf2':
                if allkerns.ndim>4:# was 3 now 4 with batchcount
                    for i in range((allkerns.ndim-4),0,-1):
                        assert allkerns.ndim>4, "allkerns is being collapsed via rbf on rhs " \
                                                "but has {} dimensions instead of ndim>4".format(allkerns.ndim)
                        allkerns=np.power(np.sum(np.power(allkerns,2),axis=-1),0.5)#collapse right most dimension, so if the two items in the 3rd dimension\\
            if modeldict['regression_model']=='NW':
                if allkerns.ndim>4:#was 3 now 4 with batchcount
                    for i in range((allkerns.ndim-4),0,-1):
                        assert allkerns.ndim>4, "allkerns is being collapsed via product on rhs " \
                                                "but has {} dimensions instead of ndim>4".format(allkerns.ndim)
                        allkerns=np.product(allkerns,axis=-1)#collapse right most dimension, so if the two items in the 3rd dimension\\
            return np.sum(allkerns,axis=0)/self.nin#collapsing across the nin kernels for each of nout    
        except FloatingPointError:
            self.nperror=1
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
    
    
    
    
    
    
    
    
    
