import numpy as np
import logging,logging.config
import os
import psutil
import yaml

class Ndiff:
    def __init__(self,):
        pass
            
        
    def Ndiff_datastacker(self,indiffs,outdiffs,depth):
        """
            for x: (x has it's nout dimension (-2) added later on)
            before:
            outdiffs is #ninXnpr
            indiffs is #ninXnin
            after:
            both are {depth*nin}XninXnpr
            
            for y:
            before:
            outdiffs is broadcast to dim2 npr times #ninXnoutXnpr?
            indiffs is broadcast to dim2 npr times #ninXninXnpr?
            after:
            outdiffs becomes {depth*nin}XninXnoutXnpr
        """
        
        
        
        
        if len(outdiffs.shape)==3:#(ninXnoutXnpr)this should only happen if we're working on y
            #
            shape_out_tup=tuple([self.nin for _ in range(depth)])+outdiffs.shape
            #expand_tup=tuple(range(depth)+2)
            if depth>1:
                if depth%2==0:
                    indiffs=np.transpose(indiffs,[1,0,2])
                expanded_indiffs=indiffs
                for _ in range(depth): 
                    expanded_indiffs=np.expand_dims(expanded_indiffs,axis=2)    # there is variation over dims 0,1 for all dims to the rhs of them
                    
                #shape_out_tup=tuple([self.nin for _ in range(depth)])+indiffs.shape
                return np.broadcast_to(expanded_indiffs,shape_out_tup)#indiffs starts as ninxninxnpr, expand_dims adds a dimension for nout
            else:
                
                
                
                return np.broadcast_to(outdiffs,shape_out_tup)
        if len(outdiffs.shape)==2:#(ninXnpr)this should only happen if we're working on x
            shape_out_tup=tuple([self.nin for _ in range(depth)])+outdiffs.shape
            if depth>1:
                if depth%2==0:
                    indiffs=np.transpose(indiffs,[1,0])
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
        This approach would have worked, but it seems would have required an additional transpose at the end
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
        sum_axis=0    
        #self.logger.info(f'kernstack.shape:{kernstack.shape}, depth: {depth}, x_or_y:{x_or_y}, sum_axis:{sum_axis}')
        if normalization=='none' or normalization==None:
            return np.ma.sum(kernstack,axis=sum_axis)

        if normalization=='own_n':
            
            return np.ma.mean(kernstack,axis=sum_axis)
        if normalization=='across':
            #return np.ma.sum(kernstack/np.ma.mean(kernstack,axis=0),axis=0)
            this_depth_sum=np.ma.sum(kernstack,axis=sum_axis)
            return this_depth_sum/np.ma.sum(this_depth_sum,axis=0)#dividing by sum across the sums at "this_depth"
            #need to think about the axis of this normalization

        
    def Ndiff_recursive(self,masked_data,deeper_bw,Ndiff_exp,Ndiff_bw,Ndiff_bw_kern,normalize,depth,x_or_y):
        return np.ma.power(
                    self.Ndiffsum_then_normalize_bw(
                        self.do_Ndiffbw_kern(Ndiff_bw_kern, masked_data,deeper_bw),normalize,depth,x_or_y),Ndiff_exp)
    
    def Ndiff_product(self,masked_data,deeper_bw,Ndiff_exp,Ndiff_bw,Ndiff_bw_kern,normalize,depth,x_or_y):
        return np.ma.power(
            self.Ndiffsum_then_normalize_bw(
                self.do_Ndiffbw_kern(Ndiff_bw_kern,masked_data,Ndiff_bw)*deeper_bw,normalize,depth,x_or_y),Ndiff_exp)
    
    def NdiffBWmaker(self,max_bw_Ndiff,fixed_or_free_paramdict,diffdict,modeldict,x_or_y):
        """returns an nin X nout npr np.array of bandwidths if x_or_y=='y'
        ? or nin X npr if x_or_y=='x' ?
        """
        Ndiff_exponent_params = self.pull_value_from_fixed_or_free('Ndiff_exponent', fixed_or_free_paramdict)
        Ndiff_depth_bw_params = self.pull_value_from_fixed_or_free('Ndiff_depth_bw', fixed_or_free_paramdict)
        Ndiff_bw_kern = modeldict['Ndiff_bw_kern']
        normalization = modeldict['normalize_Ndiffwtsum']
        Ndiff_start=modeldict['Ndiff_start']      
        indiffs=diffdict['indiffs']
        outdiffs=diffdict['outdiffs']
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
                this_depth_mask = masklist[depth]

                
                '''if Ndiff_start>1:# the next few lines collapse the length of Ndiff dimensions before Ndiff start down to lenght 1, but preserves the dimension
                    select_dims=list((slice(None),)*this_depth_mask.ndim)#slice(None) is effectively a colon when the list is turned into a tuple of dimensions
                    for dim in range(Ndiff_start-1,0,-1):
                        shrinkdim=max_bw_Ndiff-dim
                        select_dims[shrinkdim]=[0,]
                    dim_select_tup=tuple(select_dims)
                    this_depth_mask=this_depth_mask[dim_select_tup]
                    this_depth_data=this_depth_data[dim_select_tup]'''
                this_depth_masked_data=np.ma.array(this_depth_data,mask=this_depth_mask,keep_mask=False)
                sumtolastdim_tup=tuple(range(this_depth_masked_data.ndim))[:-1]
                #self.logger.debug(f'before- depth:{depth}, x_or_y:{x_or_y}, this_depth_masked_data maskcount(axis=upto-1):{np.ma.count_masked(this_depth_masked_data,axis=sumtolastdim_tup)} vs. np.sum(this_depth_mask,axis=upto-1):{np.sum(this_depth_mask,axis=sumtolastdim_tup)}, with shape:{this_depth_masked_data.shape}')
                #if depth==1:
                #if x_or_y=='x':
                #    self.logger.debug(f'depth:{depth}, x_or_y:{x_or_y},this_depth_mask:{this_depth_mask}')
                
                if Ndiff_type=='product':
                    this_depth_bw=self.Ndiff_product(this_depth_masked_data,deeper_depth_bw,this_depth_exponent,this_depth_bw_param,Ndiff_bw_kern,normalize,depth,x_or_y)
                if Ndiff_type=='recursive':
                    if depth==max_bw_Ndiff:deeper_depth_bw=Ndiff_depth_bw_params[0]
                    this_depth_bw=self.Ndiff_recursive(this_depth_masked_data,deeper_depth_bw,this_depth_exponent,this_depth_bw_param,Ndiff_bw_kern,normalize,depth,x_or_y)
                sumtolastdim_tup=tuple(range(this_depth_bw.ndim))[:-1]
                #self.logger.debug(f'after- depth:{depth}, x_or_y:{x_or_y}, this_depth_bw maskcount(axis=upto-1)/shape:{np.ma.count_masked(this_depth_bw,axis=sumtolastdim_tup)} of {this_depth_bw.shape}')    
                
                if depth>0: deeper_depth_bw=this_depth_bw#setup deeper_depth_bw for next iteration if there is another
            '''if missing_i_dimension==1:
                dimcount=len(this_depth_bw.shape)
                print(f'this_depth_bw.shape:{this_depth_bw.shape}')
                full_tuple=tuple(list(this_depth_bw.shape).insert(dimcount-3,self.nin))#
                full_tuple
                np.broadcast_to(this_depth_bw,full_tuple)
            '''
            
            return this_depth_bw
                
        if Ndiff_bw_kern=='product': #outdiffs parameter column not yet collapsed
            n_depth_masked_sum_kern=self.do_Ndiffbw_kern(Ndiff_bw_kern,n_depth_masked_sum,Ndiff_depth_bw_params[depth],x_bandscale_params)

    
    
    
    def do_Ndiffbw_kern(self,kern_choice,maskeddata,Ndiff_depth_bw_param,x_bandscale_params=None):
        if kern_choice=="product":
            #return np.ma.product(x_bandscale_params,np.ma.exp(-np.ma.power(maskeddata,2)),axis=maskeddata.ndim-1)/Ndiff_depth_bw_param
            shapetup=maskeddata.shape
            #the number of params should be equal to the length of the rightmost dimension, so broadcasting should match all the dimensions to the left according to shapetup.
            return self.gkernh(maskeddata,np.broadcast_to(x_bandscale_params,shapetup))
            
        if kern_choice=='rbfkern':
            return self.gkernh(maskeddata, Ndiff_depth_bw_param)#parameters already collapsed, so this will be rbf
    
    def gkernh(self, x, h):
        "returns the rbf/gaussian kernel at x with bandwidth h"
        amask=np.ma.getmask(x)
        kern=np.ma.exp(-np.ma.power(x,2)/(np.ma.power(h,2)*2))
        kern=np.ma.masked_array(np.nan_to_num(kern,copy=False),mask=amask)
        return kern#
        

    def max_bw_Ndiff_maskstacker_y(self,npr,nout,nin,p,max_bw_Ndiff,ykerngrid):
        try:
            self.predict_self_without_self
        except:
            self.logger.exception('setting self.predict_self_without_self to "n/a"')
            self.predict_self_without_self='n/a'
        #print('nout:',nout)
        #ninmask=np.broadcast_to(np.ma.make_mask(np.eye(nin))[:,:,None],(nin,nin,npr))
        
        if not self.predict_self_without_self=='yes':
            ninmask=np.broadcast_to(np.eye(nin,dtype=np.bool)[:,:,None],(nin,nin,npr))
        if self.predict_self_without_self=='yes' and nin==npr and ykerngrid=='no':
            assert False, 'needs logic rewrite'
            ninmask3=np.broadcast_to(np.eye(nin,dtype=np.bool)[:,:,None],(nin,nin,npr))
            ninmask2=np.broadcast_to(np.eye(nin,dtype=np.bool)[:,None,:],(nin,nin,nin))
            ninmask1=np.broadcast_to(np.eye(nin,dtype=np.bool)[None,:,:],(nin,nin,nin))
            ninmask1=np.ma.make_mask(ninmask1)
            ninmask2=np.ma.make_mask(ninmask2)
            ninmask3=np.ma.make_mask(ninmask3)
            ninmask=np.ma.mask_or(ninmask1,np.ma.mask_or(ninmask2,ninmask3))
        if self.predict_self_without_self=='yes' and nin==npr and type(ykerngrid) is int:
            assert False, 'needs logic rewrite'
            ninmask0=np.ma.make_mask(np.broadcast_to(np.eye(nin,dtype=np.bool)[:,:,None],(nin,nin,npr)))
            ninmask=np.ma.make_mask(np.broadcast_to(np.eye(nin,dtype=np.bool)[:,None,:],(nin,nin,nin)))
            ninmask=np.ma.mask_or(ninmask0,ninmask)
            
            
        list_of_masks=[ninmask] # nout dimensionality fixed at end of needed
        if max_bw_Ndiff>0:
            if ykerngrid=='no':
                assert False, 'needs reconfigure to bool and logic rewrite'
                firstdiffmask=np.ma.mask_or(np.ma.make_mask(np.broadcast_to(np.expand_dims(ninmask,0),(nin,nin,nin,npr))),np.ma.make_mask(np.broadcast_to(np.expand_dims(ninmask,2),(nin,nin,nin,npr))))
                firstdiffmask=np.ma.mask_or(np.ma.make_mask(np.broadcast_to(np.expand_dims(ninmask,1),(nin,nin,nin,npr))),firstdiffmask)
                #when 0 dim of ninmask is expanded, masked if i=j for all k.
                #when 2 dim of nin mask is expanded, masked if k=j for all i, and when 1 dim of nin mask is expanded, masked if k=i for all j. all are for all ii
            if type(ykerngrid) is int:
                #firstdiffmask=np.ma.make_mask(np.broadcast_to(np.expand_dims(ninmask,2),(nin,nin,nout,npr)))
                firstdiffmask=np.broadcast_to(np.expand_dims(ninmask,2),(nin,nin,nout,npr))#k!=j Vi,ii (V : `for all')
                #if yout is a grid and nout not equal to nin, 
            list_of_masks.append(firstdiffmask)# this line is indexed (k,j,i,ii)
                
                
                
        '''if max_bw_Ndiff>1:
            for ii in range(max_bw_Ndiff-1):
                ninmask_3d=list_of_masks[1]
                preshape=ii*(nin,)+ninmask_3d.shape # if max_bw_ndiff=2, at depth=2, ii is [0]
                nextshape=(nin,)+preshape
                ninmask_pre=np.broadcast_to(ninmask_3d,preshape)
                nextblankmask=np.zeros(nextshape,dtype=np.bool)
                list_of_masks.append(nextblankmask)
                for iii in range(ii+2):
                    list_of_masks[-1]=list_of_masks[-1]+np.expand_dims(ninmask_pre,axis=iii) '''
                    
                    
        if max_bw_Ndiff>1:
            for ii in range(max_bw_Ndiff-1):
                ninmask_4d=list_of_masks[1]
                nextshape=(ii+1)*(nin,)+ninmask_4d.shape
                nextblankmask=np.zeros(nextshape,dtype=np.bool)
                list_of_masks.append(nextblankmask)
                for iii in range(ii+1):
                    a_condition_mask=ninmask_4d.copy()
                    exp_dim_list=[1]*(iii+1)+[-3]*(ii-iii) 
                    for dim in exp_dim_list:
                        a_condition_mask=np.expand_dims(a_condition_mask,axis=dim)
                    list_of_masks[-1]+=a_condition_mask             
                    
                    '''
        if max_bw_Ndiff>1:
            for ii in range(max_bw_Ndiff-1):#-1 b/c 1diff masks already in second position of list of masks if max_bw_Ndiff>0
                lastmask=list_of_masks[-1].copy() #second masks always based on self.nin
                masktup=(nin,)+lastmask.shape#expand dimensions on lhs
                #list_of_masks.append(np.ma.make_mask(np.broadcast_to(np.expand_dims(lastmask,0),masktup)))
                list_of_masks.append(np.broadcast_to(np.expand_dims(lastmask,0),masktup))
                for iii in range(ii+2):#if Ndiff is 2, above for loop maxes out at 1,
                    #then this loop maxes at 0,1 from range(2-1+1)
                    iii+=1 #since 0 dim added before for_loop
                    #list_of_masks[-1]=np.ma.mask_or(list_of_masks[-1],np.ma.make_mask(np.broadcast_to(np.expand_dims(lastmask,axis=iii),masktup)))
                    list_of_masks[-1]=list_of_masks[-1]+\
                        np.broadcast_to(
                            np.expand_dims(
                                lastmask,axis=iii),masktup)'''
                if ykerngrid=='no':
                    assert False, 'needs logic rewrite'
                    list_of_masks[-1]=np.ma.mask_or(
                        list_of_masks[-1],np.ma.make_mask(
                            np.broadcast_to(
                                np.expand_dims(lastmask,axis=ii+3),masktup)))#this should mask \
                    #the yout values from the rest since yout==yin        
        if type(ykerngrid) is int:
            #list_of_masks[0]=np.ma.make_mask(np.zeros([nin,nout,npr]))#overwrite first item in list of masks to remove masking when predicting y using ykerngrid==int
            list_of_masks[0]=np.zeros([nin,nout,npr],dtype=np.bool)
        #[print('shape of mask {}'.format(i),list_of_masks[i].shape) for i in range(max_bw_Ndiff+1)]
        return list_of_masks
        
    def max_bw_Ndiff_maskstacker_x(self,npr,nout,nin,p,max_bw_Ndiff):
        '''match the parameter structure of data produced by Ndiff_datastacker at each depth
        notably, mostly differences (and thus masks) will be between the nin (n in the original dataset) obeservations.
        though would be interesting to make this more flexible in the future.
        when i insert a new dimension between two dimensions, I'm effectively transposing
        -Ndiff_datastacker does not have the 'out dimension at position, -1 added yet.'
        '''
        try:
            self.predict_self_without_self
        except:
            self.logger.exception('setting self.predict_self_without_self to "n/a"')
            self.predict_self_without_self='n/a'
        
            
        ninmask=np.ma.make_mask(np.eye(nin,dtype=np.bool))
        list_of_masks=[ninmask]
        if not self.predict_self_without_self=='yes' and max_bw_Ndiff>0:#first mask will be corrected at the bottom
            list_of_masks.append((np.broadcast_to(ninmask[:,:,None],(nin,nin,npr))))
        '''if self.predict_self_without_self=='yes' and nin==npr and max_bw_Ndiff>0:
                assert False, 'needs logic rewrite'
                ninmask3=np.broadcast_to(ninmask[:,:,None],(nin,nin,nin))
                ninmask2=np.broadcast_to(ninmask[:,None,:],(nin,nin,nin))
                ninmask1=np.broadcast_to(ninmask[None,:,:],(nin,nin,nin))
            

                list_of_masks.append(ninmask1+ninmask2+ninmask3)'''
        if max_bw_Ndiff>1:
            for ii in range(max_bw_Ndiff-1):
                ninmask_3d=list_of_masks[1]
                nextshape=(ii+1)*(nin,)+ninmask_3d.shape
                nextblankmask=np.zeros(nextshape,dtype=np.bool)
                list_of_masks.append(nextblankmask)
                for iii in range(ii+1):
                    a_condition_mask=ninmask_3d.copy()
                    exp_dim_list=[1]*(iii+1)+[-2]*(ii-iii) # if depth=2, ii=1, dimensio added to ninmask_3d at dim1 for l!=jVi,i,j and at 
                    for dim in exp_dim_list:
                        a_condition_mask=np.expand_dims(a_condition_mask,axis=dim)
                    list_of_masks[-1]+=a_condition_mask
                    
                    
                    
                    
        '''if max_bw_Ndiff>1:
            for ii in range(max_bw_Ndiff-1):#-1 b/c 1diff masks already in second position of list of masks if max_bw_Ndiff>0
                lastmask=list_of_masks[-1].copy() #second masks always based on self.nin
                masktup=(nin,)+lastmask.shape#expand dimensions on lhs
                list_of_masks.append(np.broadcast_to(np.expand_dims(lastmask,0),masktup))
                for iii in range(ii+2):#if Ndiff is 2, above for loop maxes out at ii=0,
                    #then this loop has iii=  0,1 from range(0+2)
                    iii+=1 #since 0 dim added before for_loop
                    list_of_masks[-1]=list_of_masks[-1]+np.broadcast_to(np.expand_dims(lastmask,axis=iii),masktup)'''
            #lastmask=list_of_masks[-1]#copy the last item to lastmask
        if not self.predict_self_without_self=='yes':
            list_of_masks[0]=np.zeros([nin,npr],dtype=np.bool)
        return list_of_masks
    

    
    
    
    
    
    
    def Ndiffdo_KDEsmalln(self,diffs,bw,modeldict):
        """estimate the density items in outdiffs. collapse via products if dimensionality is greater than 2
        first 2 dimensions of outdiffs must be ninXnout
        """ 
        assert diffs.shape==bw.shape, "diffs is shape:{} while bw is shape:{}".format(diffs.shape,bw.shape)
        allkerns=self.gkernh(diffs, bw)
        normalization=modeldict['product_kern_norm']
        if normalization =='self':
            allkerns_sum=np.ma.sum(allkerns,axis=-2)#this should be the nout axis
            allkerns=allkerns/self.ma_broadcast_to(np.ma.expand_dims(allkerns_sum,-2),allkerns.shape)
            
            # collapse just nin dim or both lhs dims?
        if normalization =="own_n":
            allkerns=allkerns/np.ma.expand_dims(np.ma.count(allkerns,axis=-2),-2)#1 should be the nout axis
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

    
    
    
    
    
    
    
    
    
