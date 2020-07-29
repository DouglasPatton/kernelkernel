from Ndiff_nomask import Ndiff
from mykernhelper import MyKernHelper

import re
import multiprocessing
#import traceback
from copy import deepcopy
from typing import List
import os
from time import strftime, sleep
import datetime
import pickle
#https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
#from numba import jit
from scipy.optimize import minimize
from sklearn import metrics
#from sklearn.linear_model import LogisticRegression
import logging

#import logging.config
import yaml


class kNdtool(Ndiff,MyKernHelper):
    """kNd refers to the fact that there will be kernels in kernels in these estimators

    """

    def __init__(self,savedir=None,myname=None):
        if savedir==None:
            savedir=os.getcwd()
        #logdir=os.path.join(savedir,'log')
        #if not os.path.exists(logdir):os.mkdir(logdir)
        try:
            self.logger
        except:
            self.logger=logging.getLogger(__name__)
        self.savedir=savedir
        self.pname=myname

        Ndiff.__init__(self,)
        MyKernHelper.__init__(self,)
        

        
    def BWmaker(self, fixed_or_free_paramdict, diffdict, modeldict,xory):
        if self.Ndiff:
            if modeldict['max_bw_Ndiff']==0:
                if xory=='x':
                    bwshape=(self.nin,self.npr,self.batchcount)
                if xory=='y':
                    bwshape=(self.nin,self.nout,self.npr,self.batchcount)
                bw=np.array([1])
                np.broadcast_to(bw,bwshape)    
                return np.array([1])
            else:
                return self.NdiffBWmaker(modeldict['max_bw_Ndiff'], fixed_or_free_paramdict, diffdict, modeldict,xory)
    
    def batchKDEpredict(self,yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict):
        """moves free_params to first position of the obj function, preps data, and then runs MY_KDEreg to fit the model
            then returns loss of the fit 
        Assumes last p elements of free_params are the scale parameters for 'el two' approach to
        columns of x.
        """

        try:
            residual_treatment=modeldict['residual_treatment']
        except KeyError:
            residual_treatment=None
        iscrossloss=residual_treatment[0:8]=='crossloss'
        
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        #pull x_bandscale parameters from the appropriate location and appropriate vector
        x_bandscale_params=self.pull_value_from_fixed_or_free('x_bandscale',fixed_or_free_paramdict)
        y_bandscale_params=self.pull_value_from_fixed_or_free('y_bandscale',fixed_or_free_paramdict)
        p=x_bandscale_params.shape[0]
        assert self.p==p,\
            "p={} but x_bandscale_params.shape={}".format(self.p,x_bandscale_params.shape)
        try :
            spatial=self.datagen_obj.spatial
        except:
            spatial=0
        try: spatialtransform=modeldict['spatialtransform']
        except: spatialtransform=None
        if modeldict['Ndiff_bw_kern']=='rbfkern':

            #xin_scaled=xin*x_bandscale_params
            #p#rint('xin_scaled.shape',xin_scaled.shape)
            #xpr_scaled=xpr*x_bandscale_params
            #p#rint('xpr_scaled.shape',xpr_scaled.shape)

            yin_scaled=yin*y_bandscale_params
            yout_scaled=yout*y_bandscale_params
            y_outdiffs=self.makediffmat_itoj(yin_scaled,yout_scaled) # updated for batch
            y_indiffs=self.makediffmat_itoj(yin_scaled,yin_scaled)
            
            diffmat=self.makediffmat_itoj(xin,xpr,spatial=spatial,spatialtransform=spatialtransform)
            #self.logger.debug(f'diffmat.shape:{diffmat.shape}, diffmat:{diffmat}')
            diffmat_scaled=diffmat*x_bandscale_params[:,None]
            outdiffs_scaled_l2norm=np.power(np.sum(np.power(diffmat_scaled,2),axis=2),.5)
            indiffs_scaled_l2norm=np.power(np.sum(np.power(self.makediffmat_itoj(xin,xin,spatial=spatial,spatialtransform=spatialtransform)*x_bandscale_params[:,None],2),axis=2),.5) # [:,None] for broadcasting to batch dim at -1
            assert outdiffs_scaled_l2norm.shape==(xin.shape[0],xpr.shape[0],self.batchcount),f'outdiffs_scaled_l2norm has shape:{outdiffs_scaled_l2norm.shape} not shape:({self.nin},{self.npr},{self.batchcount})'

            diffdict={}
            diffdict['outdiffs']=outdiffs_scaled_l2norm#ninXnpr?
            diffdict['indiffs']=indiffs_scaled_l2norm#ninXnin?
            ydiffdict={}
            ydiffdict['outdiffs']=np.broadcast_to(y_outdiffs[:,:,None,:],y_outdiffs.shape[:-1]+(self.npr,self.batchcount))#ninXnoutXnprXbatchcount
            ydiffdict['indiffs']=np.broadcast_to(y_indiffs[:,:,None,:],y_indiffs.shape[:-1]+(self.npr,self.batchcount))#ninXninXnprXbatchcount
            diffdict['ydiffdict']=ydiffdict


        """if modeldict['Ndiff_bw_kern']=='product':
            outdiffs=makediffmat_itoj(xin,xpr,spatial=spatial)#scale now? if so, move if...='rbfkern' down 
            #predict
            NWtup=self.MY_NW_KDEreg(yin_scaled,xin_scaled,xpr_scaled,yout_scaled,fixed_or_free_paramdict,diffdict,modeldict)[0]
            #not developed yet"""
        try:
            xbw = self.BWmaker(fixed_or_free_paramdict, diffdict, modeldict,'x')
            ybw = self.BWmaker(fixed_or_free_paramdict, diffdict['ydiffdict'],modeldict,'y')

            #p#rint('xbw',xbw)
            #p#rint('ybw',ybw)


            hx=self.pull_value_from_fixed_or_free('outer_x_bw', fixed_or_free_paramdict)
            hy=self.pull_value_from_fixed_or_free('outer_y_bw', fixed_or_free_paramdict)

            xbw=xbw*hx


            if modeldict['regression_model']=='logistic':
                xoutdiffs=diffdict['outdiffs']
                prob_x = self.do_KDEsmalln(xoutdiffs, xbw, modeldict)

                yhat_tup=self.kernel_logistic(prob_x,xin,yin)
                yhat_std=yhat_tup[0]
                cross_errors=yhat_tup[1]

            if modeldict['regression_model'][0:2]=='NW':
                ybw=ybw*hy
                xoutdiffs=diffdict['outdiffs']
                youtdiffs=diffdict['ydiffdict']['outdiffs']
                ykern_grid=modeldict['ykern_grid'];xkern_grid=modeldict['xkern_grid']
                if True:#type(ykern_grid) is int and xkern_grid=='no':
                    xoutdifftup=xoutdiffs.shape[:-2]+(self.nout,)+xoutdiffs.shape[-2:]# self.nout dimension inserted third from rhs not 2nd b/c batchcount.
                    #p#rint('xoutdiffs.shape',xoutdiffs.shape,'xbw.shape',xbw.shape)
                    xoutdiffs_stack=np.broadcast_to(np.expand_dims(xoutdiffs,axis=-3),xoutdifftup)#from -2 to -3
                    xbw_stack=np.broadcast_to(np.expand_dims(xbw,axis=-3),xoutdifftup)#fro -2 to -3
                newaxis=-1
                yx_outdiffs_endstack=np.concatenate(
                    (np.expand_dims(xoutdiffs_stack,axis=newaxis),np.expand_dims(youtdiffs,newaxis)),axis=newaxis)
                yx_bw_endstack=np.concatenate([
                    np.expand_dims(xbw_stack,newaxis),
                    np.expand_dims(ybw,newaxis)]
                    ,axis=newaxis)
                #p#rint('type(xoutdiffs)',type(xoutdiffs),'type(xbw)',type(xbw),'type(modeldict)',type(modeldict))

                prob_x = self.do_KDEsmalln(xoutdiffs, xbw, modeldict)
                prob_yx = self.do_KDEsmalln(yx_outdiffs_endstack, yx_bw_endstack,modeldict)#

                KDEregtup = self.my_NW_KDEreg(prob_yx,prob_x,yout,modeldict)
                if modeldict['residual_treatment']=='batchnorm_crossval':
                    return KDEregtup
                else:
                    yhat_raw=KDEregtup[0]
                    cross_errors=KDEregtup[1]

                yhat_std=yhat_raw#no longer needed. *y_bandscale_params**-1#remove the effect of any parameters applied prior to using y.

            std_data=modeldict['std_data']

            if type(std_data) is str:
                if std_data=='all':
                    yhat_un_std=yhat_std*self.ystd+self.ymean
            elif type(std_data) is tuple:
                if not std_data[0]==[]:
                    yhat_un_std=yhat_std*self.ystd+self.ymean
                    if iscrossloss:
                        cross_error=cross_errors*self.ystd
                else: yhat_un_std=yhat_std

            #p#rint(f'yhat_un_std:{yhat_un_std}')



            return (yhat_un_std,cross_errors)
        except FloatingPointError:
            self.nperror=1
            self.logger.exception('nperror set to 1 to trigger error and big loss')
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
        
        
    def slicetup(self,dimcount,dimselect,dimval):
        
        if not type(dimselect) is list:
            dimselect=[dimselect]
            dimval=[dimval]
        for dim in dimselect:
            if dim<0:
                dim+=dimcount#so all dims are from lhs, 0...dimcount-1
        slicelist=[slice(None) for _ in range(dimcount)]
        for i in range(len(dimselect)):
            slicelist[dimselect[i]]=dimval[i]
        return tuple(slicelist)
        
    def do_batchnorm_crossval(self,KDEregtup,fixed_or_free_paramdict,modeldict):
        '''
        i is indexing the batchcount chunks of npr that show up batchcount-1 times in crossvalidation. 
        if i is 1, @j=0, j<i istart:iend::0-nin from batch 0, which does not have its own data in 0-nin, since that was skipped for cross-val (not masked either).
        if i is 1, @j=2, j>i istart:iend::nin-2nin from batch 2
        if i is 1, @j=3, "                       " from batch 3
        '''
        try:
            #modifying to index numpy arrays instead of lists, requiring slices of all dims
            batchcount=self.batchcount
            yout,wt_stack,cross_errors=KDEregtup# dims:(nout,nin,batch)???
            nin=self.nin;
            #ybatch=[]
            wtbatch=[]
            youtbatch=[]
            #trueybatch=[]
            
            for i in range(batchcount):
                
                #ybatchlist=[]
                wtbatchlist=[]
                youtbatchlist=[]
                #trueybatchlist=[]
                for j in range(batchcount):
                    if j>i or self.validate:
                        istart=(i)*nin
                        iend=istart+nin
                    elif j<i:
                        istart=(i-1)*nin
                        iend=istart+nin
                    if j!=i or self.validate:
                        batchslicer=self.slicetup(wt_stack.ndim,[-2,-1],[slice(istart,iend),j])
                        wt_i_from_batch_j=wt_stack[batchslicer]
                        yout_batchj=np.broadcast_to(np.expand_dims(yout[:,j],axis=-1),(self.nout,self.nin)) # 
                        wtbatchlist.append(wt_i_from_batch_j)
                        youtbatchlist.append(yout_batchj)

                dimcount=np.ndim(wtbatchlist[0])
                wtbatch.append(np.concatenate([wtbatchj[None,:,:]for wtbatchj in wtbatchlist],axis=0)) # 2/20b adding lhs axis for batchcoun-1 predictions
                youtbatch.append(np.concatenate([youtbatchj[None,:,:] for youtbatchj in youtbatchlist],axis=0)) #2/20b each i has batch_n values to predict 2/20a same as above,
                #   leaving rhs dim as nin*(batchcoun-1)=npr

            wtstack=np.concatenate([wtbatchi[:,:,:,None] for wtbatchi in wtbatch],axis=-1)#adding new rhs axis for stacking batches(i)
            #self.logger.debug(f'wtstack:{wtstack}')
            youtstack=np.concatenate([youtbatchi[:,:,:,None] for youtbatchi in youtbatch],axis=-1)
            wtstacksum=np.sum(wtstack,axis=0)#summed over batchj axis for each batchi
            wtstacksumsum=np.sum(wtstacksum,axis=0)# summed over the yout axis for each batchi
            wtstacksumsum=np.expand_dims(wtstacksumsum,axis=0)# add back in the two lhs collapsed axes
            wtstacksumsum=np.expand_dims(wtstacksumsum,axis=0)
            wtstacksumsum=np.broadcast_to(wtstacksumsum,wtstack.shape) # return to the original dimensions
            #self.logger.info(f'wtstacksumsum.shape:{wtstacksumsum.shape}')
            #self.logger.info(f'wtstacksumsum:{wtstacksumsum}')
            wtstacknorm=np.zeros(wtstack.shape,dtype=np.float64)
            wtstacknorm[wtstacksumsum>0]=wtstack[wtstacksumsum>0]/wtstacksumsum[wtstacksumsum>0]
            yhat_raw=np.sum(np.sum(wtstacknorm*youtstack,axis=0),axis=0)
            #print(f'yhat_raw.shape:{yhat_raw.shape}, expected:(nin,batchcount):{(nin,batchcount)}')
            yhat_raw=yhat_raw.flatten(order='F')

            #y_bandscale_params=self.pull_value_from_fixed_or_free('y_bandscale',fixed_or_free_paramdict) #  removed on 5/2 since NW takes yout not yout_scaled as arg

            yhat_std=yhat_raw# no longer needed*y_bandscale_params**-1#remove the effect of any parameters applied prior to using y.

            std_data=modeldict['std_data']
            if type(std_data) is str:
                if std_data=='all':
                    yhat_un_std=yhat_std*self.ystd+self.ymean
            elif type(std_data) is tuple:
                if not std_data[0]==[]:
                    yhat_un_std=yhat_std*self.ystd+self.ymean
                    try:
                        cross_error=cross_errors*self.ystd
                    except:pass
                else: yhat_un_std=yhat_std

            return yhat_un_std,cross_errors
        except FloatingPointError:
            self.nperror=1
            self.logger.exception('nperror set to 1 to trigger error and big loss')
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
        
    
    def do_KDEsmalln(self,diffs,bw,modeldict):
        try:
            if self.Ndiff:
                return self.Ndiffdo_KDEsmalln(diffs, bw, modeldict)
        except FloatingPointError:
            self.nperror=1
            self.logger.exception('')
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
        
    
    def kernel_logistic(self,prob_x,xin,yin):
        residual_treatment=modeldict['residual_treatment']
        iscrossloss=residual_treatment[0:8]=='crossloss'
                      
        for i in range(prob_x.shape[-1]):
            xin_const=np.concatenate(np.ones((xin.shape[0],1),xin,axis=1))
            yhat_i=LogisticRegression().fit(xin_const,yin,prob_x[...,i]).predict(xin)
            yhat_std.extend(yhat_i[i])
            cross_errors.extend(yhat_i)#list with ii on dim0
        #cross_errors=np.masked_array(cross_errors,mask=np.eye(yin.shape[0])).T#to put ii back on dim 1
        cross_errors=cross_errors.T#no mask version
        yhat=np.array(yhat_std)                             
        if not iscrossloss:
            return (yhat,'no_cross_errors')
        if iscrossloss:
            if len(residual_treatment)>8:
                cross_exp=float(residual_treatment[8:])
                wt_stack=prob_x**cross_exp
            
            cross_errors=(yhat[None,:]-yout[:,None])#this makes dim0=nout,dim1=nin
            crosswt_stack=wt_stack/np.expand_dims(np.sum(wt_stack,axis=1),axis=1)
            wt_cross_errors=np.sum(crosswt_stack*cross_errors,axis=1)#weights normalized to sum to 1, then errors summed to 1 per nin
            return (yhat,wt_cross_errors)

    def my_NW_KDEreg(self,prob_yx,prob_x,yout,modeldict):
        """returns predited values of y for xpredict based on yin, xin, and modeldict
        """
        try:
            residual_treatment=modeldict['residual_treatment']
            try:
                iscrossloss=residual_treatment[0:8]=='crossloss'
            except: iscrossloss=0

            yout_axis=-3 # from -2 to -3 b/c batchcount#len(prob_yx.shape)-2#-2 b/c -1 for index form vs len count form and -1 b/c second to last dimensio is what we seek.
            '''print('yout_axis(expected 0): ',yout_axis)
            print('prob_yx.shape',prob_yx.shape)
            print('prob_x.shape',prob_x.shape)'''
            #prob_yx_sum=np.broadcast_to(np.ma.expand_dims(np.ma.sum(prob_yx,axis=yout_axis),yout_axis),prob_yx.shape)
            #cdfnorm_prob_yx=prob_yx/prob_yx_sum
            #cdfnorm_prob_yx=prob_yx#dropped normalization
            #prob_x_sum=np.broadcast_to(np.ma.expand_dims(np.ma.sum(prob_x, axis=yout_axis),yout_axis),prob_x.shape)
            #cdfnorm_prob_x = prob_x / prob_x_sum
            #cdfnorm_prob_x = prob_x#dropped normalization

            #yout_stack=np.broadcast_to(np.ma.expand_dims(yout,1),(self.nout,self.npr))
            yout_stack=np.expand_dims(yout,1)
            prob_x_stack_tup=prob_x.shape[:-2]+(self.nout,)+prob_x.shape[-2:] # -1 to -2 b/c batchcount
            prob_x_stack=np.broadcast_to(np.expand_dims(prob_x,yout_axis),prob_x_stack_tup)
            NWnorm=modeldict['NWnorm']

            residual_treatment=modeldict['residual_treatment']
            # print('before',NWnorm,'lossfn:',lssfn)
            if NWnorm=='across-except:batchnorm':
                if residual_treatment=='batchnorm_crossval':
                    NWnorm='none'
                else:
                    NWnorm='across'
            # print('after',NWnorm,'lossfn:',lssfn)


            if modeldict['regression_model']=='NW-rbf2':
                wt_stack=np.power(np.power(prob_yx,2)-np.power(prob_x_stack,2),0.5)
                if NWnorm=='across':
                    wt_stack=wt_stack/np.expand_dims(np.sum(wt_stack,axis=yout_axis),axis=yout_axis)
                yhat=np.sum(yout_stack*wt_stack,axis=yout_axis)#yout axis should be -2 # -3 b/c batchcount

            else:
                #self.logger.debug(f'prob_yx.shape:{prob_yx.shape}')
                #self.logger.debug(f'prob_yx:{prob_yx}')
                #self.logger.debug(f'prob_x_stack.shape:{prob_x_stack.shape}')
                #self.logger.debug(f'prob_x_stack:{prob_x_stack}')
                wt_stack=np.zeros(prob_yx.shape,dtype=np.float64)
                wt_stack[prob_x_stack>0]=prob_yx[prob_x_stack>0]/prob_x_stack[prob_x_stack>0]
                if NWnorm=='across':
                    wt_stack=wt_stack/np.expand_dims(np.sum(wt_stack,axis=yout_axis),axis=yout_axis)
                yhat=np.sum(yout_stack*wt_stack,axis=yout_axis)

            binary_threshold=modeldict['binary_y']   
            if not binary_threshold is None and not residual_treatment=='batchnorm_crossval':
                binary_yhat=np.zeros(yhat.shape)
                binary_yhat[yhat>binary_threshold]=1
                yhat=binary_yhat



            if not iscrossloss:
                cross_errors='no_cross_errors'


            if iscrossloss:
                if len(residual_treatment)>8:
                    cross_exp=float(residual_treatment[8:])
                    wt_stack=wt_stack**cross_exp

                cross_errors=(yhat[None,:]-yout[:,None])#this makes dim0=nout,dim1=nin
                crosswt_stack=wt_stack/np.expand_dims(np.sum(wt_stack,axis=1),axis=1)
                wt_cross_errors=np.sum(crosswt_stack*cross_errors,axis=1)#weights normalized to sum to 1, then errors summed to 1 per nin
                cross_errors=wt_cross_errors
            if modeldict['residual_treatment']=='batchnorm_crossval':
                return (yout,wt_stack,cross_errors)
            return (yhat,cross_errors)
        except FloatingPointError:
            self.nperror=1
            self.logger.exception('nperror set to 1 to trigger error and big loss')
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
        
            
    
    def predict_tool(self,xpr,fixed_or_free_paramdict,modeldict):
        """
        """
        try:
            xpr=(xpr-self.xmean)/self.xstd

            self.prediction=MY_KDEpredictloss(self, free_params, batchdata_dict, modeldict, fixed_or_free_paramdict,predict=None)
        
            return self.prediction.yhat
        except FloatingPointError:
            self.nperror=1
            self.logger.exception('nperror set to 1 to trigger error and big loss')
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
        

    def MY_KDEpredictloss(self, free_params, batchdata_dictlist, modeldict, fixed_or_free_paramdict,predict=None):
        
        #predict=1 or yes signals that the function is not being called for optimization, but for prediction.
        try:
            if self.forcefail:
                self.logger.DEBUG('forcefail activated')
            #p#rint(f'returning self.forcefail:{self.forcefail}')
                return self.forcefail
        except:
            pass
        if self.iter==1:
            try:
                self.success
                if not type(self.success) is np.float64:
                    self.forcefail=9.99999*10**295
                    #print(f'self.success:{self.success},type(self.success):{type(self.success)}')
                if self.success>self.loss_threshold:
                    #print(f'self.success:{self.success},self.loss_threshold:{self.loss_threshold}')
                    self.forcefail=9.99999*10**296
            except:
                self.forcefail=9.99999*10**297
        self.iter+=1
        if predict==None or predict=='no':
            predict=0
        if predict=='yes':
            predict=1
        if  type(fixed_or_free_paramdict['free_params']) is str and fixed_or_free_paramdict['free_params'] =='outside':  
            self.call_iter += 1  # then it must be a new call during optimization
        
        batchcount=self.batchcount

        fixed_or_free_paramdict=self.insert_detransformed_freeparams(fixed_or_free_paramdict,free_params)
        self.fixed_or_free_paramdict = fixed_or_free_paramdict
        try:
            residual_treatment=modeldict['residual_treatment']
        except KeyError:
            print(f'residual_treatment not found in modeldict')
            residual_treatment=None
        iscrossloss=residual_treatment[0:8]=='crossloss'
        
        if self.source=='monte': 
            yxtup_list=self.datagen_obj.yxtup_list
        #batchbatch_all_y_err=[]
        batchbatch_all_y=[]
        batchbatch_all_yhat=[]
        #print('self.batchbatchcount',self.batchbatchcount)
        maxbatchbatchcount=modeldict['maxbatchbatchcount']
        batchbatchcount=self.batchbatchcount
        if type(maxbatchbatchcount) is int:
            if maxbatchbatchcount<self.batchbatchcount:
                batchbatchcount=maxbatchbatchcount
        try:    
            for batchbatchidx in range(batchbatchcount):
                #print('batchbatchidx:',batchbatchidx)
                if self.source=='pisces':
                    yxtup_list=self.datagen_obj.yxtup_batchbatch[batchbatchidx]
                batchdata_dict_i=batchdata_dictlist[batchbatchidx]

                #y_err_tup = ()


                #self.MY_KDEpredict(yin, yout, xin, xpr, modeldict, fixed_or_free_paramdict)
                keylist=['yintup','youttup','xintup','xprtup']
                args=[]
                for key in keylist:
                    data_tup=batchdata_dict_i[key]
                    datalist=[np.expand_dims(data_array,axis=-1) for data_array in data_tup]
                    args.append(np.concatenate(datalist,axis=-1))
                args.extend([modeldict,fixed_or_free_paramdict]) 
                yhat_unstd_outtup=self.batchKDEpredict(*args)
                #self.logger.info(f'yhat_unstd_outtup_list: {yhat_unstd_outtup_list}')
                if modeldict['residual_treatment']=='batchnorm_crossval':
                    if self.validate:
                        all_y_list=batchdata_dict_i['ylist']
                    else:
                        all_y_list=[yxvartup[0] for yxvartup in yxtup_list]
                    all_y=np.concatenate(all_y_list,axis=0)
                    all_yhat,cross_errors=self.do_batchnorm_crossval(yhat_unstd_outtup, fixed_or_free_paramdict, modeldict)

                else:
                    self.logger.critical(f'residual_treatment is not batchnorm_crossval')
                    yhat_unstd,cross_errors=yhat_unstd_outtup

                if modeldict['residual_treatment']=='batch_crossval':
                    assert False, 'not developed'
                    ybatch=[]
                    for i in range(batchcount):
                        ycross_j=[]
                        for j,yxvartup in enumerate(yxtup_list):
                            if not j==i:
                                ycross_j.append(yxvartup[0])
                        ybatch.append(np.concatenate(ycross_j,axis=0))
                elif modeldict['residual_treatment']=='batchnorm_crossval':
                    # calculation of all_y moved up
                    #all_y_err=all_y-all_yhat    
                    if type(cross_errors[0]) is np.ndarray:
                        cross_errors=np.concatenate(cross_errors,axis=0)
                else:
                    ybatch=[tup[0] for tup in yxtup_list]#the original yx data is a list of tupples
                    
                
                if not modeldict['residual_treatment']=='batchnorm_crossval':
                    ylist=[];yhatlist=[]
                    for batch_i in range(batchcount):
                        y_batch_i=ybatch[batch_i]
                        ylist.append(y_batch_i)
                        yhat_batch_i=yhat_unstd[batch_i]
                        yhatlist.append(yhat_batch_i)
                        #y_err = y_batch_i - yhat_unstd[batch_i]
                        #y_err_tup = y_err_tup + (y_err,)
                    all_y=np.concatenate(ylist,axis=0)
                    all_yhat=np.concatenate(yhatlist,axis=0)
                    #all_y_err = np.concatenate(y_err_tup,axis=0)
                if iscrossloss:
                    #needs work. split into cross_all_y and cross_all_yhat?
                    all_y_err=np.concatenate([all_y_err,np.ravel(cross_errors)],axis=0)
                #batchbatch_all_y_err.append(all_y_err)
                batchbatch_all_y.append(all_y)
                batchbatch_all_yhat.append(all_yhat)
            batchbatch_all_y=np.concatenate(batchbatch_all_y,axis=0)
            batchbatch_all_yhat=np.concatenate(batchbatch_all_yhat,axis=0)
            self.logger.debug(f'batchbatch_all_y.shape:{batchbatch_all_y.shape}')
            self.logger.debug(f'batchbatch_all_y:{batchbatch_all_y}')
            self.logger.debug(f'batchbatch_all_yhat.shape:{batchbatch_all_yhat.shape}')
            self.logger.debug(f'batchbatch_all_yhat:{batchbatch_all_yhat}')
            
            binary_threshold=modeldict['binary_y']
            
            #batchbatch_all_y_err=np.concatenate([batchbatch_all_y_err],axis=0)
            #def doLoss(self,y,yhat,pthreshold=None,lssfn=None):f
            lossdict={'mse':None,'mae':None,'f1':None,'f2':None, 'splithinge':None, 'logloss':None, 'avg_prec_sc':None}
            for lf in lossdict:
                
                lossdict[lf]=self.doLoss(batchbatch_all_y,batchbatch_all_yhat,lssfn=lf)
            mse=lossdict['mse']
            ''' mse = self.doLoss(batchbatch_all_y,batchbatch_all_yhat,lssfn='mse')
            mae = self.doLoss(batchbatch_all_y,batchbatch_all_yhat,lssfn='mae')
            splithinge=self.doLoss(batchbatch_all_y,batchbatch_all_yhat,lssfn='splithinge')
            lossdict={'mse':mse,'mae':mae,'splithinge':splithinge}'''
            self.logger.info(f'lossdict:{lossdict}, self.sample_ymean:{self.sample_ymean},n:{batchbatch_all_yhat.shape}')

            if mse<0:
                loss=-mse*100000
                self.logger.critical(f'mse:{mse}')
            #assert maskcount==0,f'{maskcount} masked values found in all_y_err'

        except FloatingPointError:
            self.nperror=1
            self.logger.exception('nperror set to 1 to trigger error and big loss')
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
        if self.nperror==1:
            self.logger.info(f'resetting nperror to 0 and setting loss to:{0.999*10**275}')
            self.nperror=0
            lossdict={key:0.999*10**275 for key in ['mse','mae','splithinge','f1']}
        self.lossdict_and_paramdict_list.append((deepcopy(lossdict), deepcopy(fixed_or_free_paramdict)))
        self.doBinaryThreshold(batchbatch_all_y,batchbatch_all_yhat,threshold=binary_threshold)
        self.logger.debug(f'len(self.binary_y_loss_list_list): {len(self.binary_y_loss_list_list)},len(self.lossdict_and_paramdict_list):{len(self.lossdict_and_paramdict_list)}')
        if predict:
            self.logger.debug(f'')
            return 
        # self.return_param_name_and_value(fixed_or_free_paramdict,modeldict)

        
        bestlossdict,bestparams=self.sort_then_saveit(self.lossdict_and_paramdict_list, modeldict)
        bestloss=bestlossdict[self.loss_function]

        if self.loss_threshold and self.iter==3 and bestloss>self.loss_threshold:
            self.forcefail=bestloss
            print(f'forcefail(loss):{self.forcefail}')
        self.success=bestloss

        loss_function=modeldict['loss_function']
        thisloss=lossdict[loss_function]
        self.logger.debug(f'at end of iteration loss:{thisloss}, with loss_function:{loss_function}')
        return thisloss

                       
                       
    def prep_KDEreg(self,datagen_obj,modeldict,param_valdict,source='monte',predict=None,valdata=None):
        #free_params,args_tuple=self.prep_KDEreg(datagen_obj,modeldict,param_valdict)
        if 'max_bw_Ndiff' in modeldict:
            self.Ndiff=1
        else:
            self.Ndiff=0
        
        try:
            batchbatchcount1=datagen_obj.batchbatchcount
            batchbatchcount2=modeldict['maxbatchbatchcount']
            self.batchbatchcount=min([batchbatchcount1,batchbatchcount2])
        except: 
            self.logger.exception(f'error, setting self.batchbatchcount to 1')
            self.batchbatchcount=1 

        
        self.datagen_obj=datagen_obj
        
        model_param_formdict=modeldict['hyper_param_form_dict']
        xkerngrid=modeldict['xkern_grid']
        ykerngrid=modeldict['ykern_grid']
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        
        #build dictionary for parameters
        free_params,fixed_or_free_paramdict=self.setup_fixed_or_free(model_param_formdict,param_valdict)
        #self.fixed_or_free_paramdict=fixed_or_free_paramdict
        if predict:
            fixed_or_free_paramdict['free_params']='predict'#instead of 'outside'
                
        
        self.batchcount=datagen_obj.batchcount
        self.nin=datagen_obj.batch_n
        self.p=datagen_obj.param_count#p should work too
        #assert self.ydata.shape[0]==self.xdata.shape[0],'xdata.shape={} but ydata.shape={}'.format(xdata.shape,ydata.shape)


        if self.source=='monte':
            yxtup_list=datagen_obj.yxtup_list
        yxtup_listlist_std=[]
        ylist=[]
        for batchbatchidx in range(self.batchbatchcount):
            if self.source=='pisces':
                yxtup_list=datagen_obj.yxtup_batchbatch[batchbatchidx]
            [ylist.extend(yxtup[0]) for yxtup in yxtup_list] 
            yxtup_listlist_std.append(self.standardize_yxtup(yxtup_list,modeldict))
        if self.validate:
            yxtup_list_stdval=self.standardize_yxtup(valdata,modeldict)# just one 
            #     batchbatch worth of validation data at a time
            ylist,xpredict=zip(*yxtup_list_stdval)
            self.logger.debug(f'len(xpredict):{len(xpredict)}, shape xpredict[0]:{xpredict[0].shape}')
        else:xpredict=None    
        self.do_naiveloss(ylist)
        #p#rint('buildbatcdatadict')
        batchdata_dictlist=self.buildbatchdatadict(yxtup_listlist_std,xkerngrid,ykerngrid,modeldict,xpredict=xpredict)
        for batchdata_dict in batchdata_dictlist:
            batchdata_dict['ylist']=ylist
        #p#rint('for validation buildbatcdatadict')
        #val_batchdata_dict=self.buildbatchdatadict(val_yxtup_list_std,xkerngrid,ykerngrid,modeldict)
        self.npr=len(batchdata_dictlist[0]['xprtup'][0])
        
       
        args_tuple=(batchdata_dictlist, modeldict, fixed_or_free_paramdict)
        #val_args_tuple=(val_batchdata_dict, modeldict, fixed_or_free_paramdict)
        #self.logger.info(f'mykern modeldict:{modeldict}')
        
        return free_params,args_tuple#,val_args_tuple
    
    
    def buildbatchdatadict(self,yxtup_listlist,xkerngrid,ykerngrid,modeldict,xpredict=None):
        '''
        load up the data for each batch into a dictionary full of tuples
        batchnorm_crossval: each tuple item contains data for a batch with predictions for all but own batch
        validate: predictions are for out of sample data, repeated for each batch
        '''
        batchcount=self.batchcount
        batchdata_dictlist=[]
        for  yxtup_list in yxtup_listlist:
        
            #p#rint('from buildbatchdatadict: batchcount: ',batchcount)
            #p#rint('self.batchcount: ',self.batchcount)
            xintup = ()
            yintup = ()
            
            xprtup = ()
            youttup = ()
            if self.validate:
                xpredict_array=np.concatenate(xpredict,axis=0) #flatten across batches and nin so prediction data is npr,p,1 where npr is batch_n*batchcount
                
                xpri=[xpredict_array for _ in range(batchcount)]
            elif modeldict['residual_treatment']=='batch_crossval' or modeldict['residual_treatment']=='batchnorm_crossval':
                #the equivalent condition for the y values in the kernelloss function does not apply to batchnorm_crossval
                xpri=[]
                for i in range(batchcount):
                    xpricross_j=[]
                    for j,yxvartup in enumerate(yxtup_list):
                        if not j==i:
                            xpricross_j.append(yxvartup[1])
                    xpri_crossval_array=np.concatenate(xpricross_j,axis=0)
                        #p#rint('xpri_crossval_array.shape',xpri_crossval_array.shape)
                    xpri.append(xpri_crossval_array)
            
                
            else:
                xpri=[None]*batchcount #self.prep_out_grid will treat this as in-sample prediction
            for i in range(batchcount):
                xdata_std=yxtup_list[i][1]
                #p#rint('xdata_std.shape: ',xdata_std.shape)
                ydata_std=yxtup_list[i][0]
                #p#rint('xprii[i]',xpri[i])
                xpr_out_i,youti=self.prep_out_grid(xkerngrid,ykerngrid,xdata_std,ydata_std,modeldict,xpr=xpri[i])
                #p#rint('xpr_out_i.shape',xpr_out_i.shape)
                xintup=xintup+(xdata_std,)
                yintup=yintup+(ydata_std,)
                xprtup=xprtup+(xpr_out_i,)
                youttup=youttup+(youti,)
                #p#rint('youttup',youttup)
                #p#rint('xprtup[0].shape:',xprtup[0].shape)

            batchdata_dict={'xintup':xintup,'yintup':yintup,'xprtup':xprtup,'youttup':youttup}
            batchdata_dictlist.append(batchdata_dict)
        #p#rint([f'{key}:{type(val)},{type(val[0])}' for key,val in batchdata_dict.items()])
        return batchdata_dictlist
    
    
    '''#args_tuple_val_list=self.valprep_args_tuple(args_tuple,datagen_obj.yxtup_batchbatch_val)
    def valprep_args_tuple(self,args_tuple,yxtup_batchbatch_val):
        bbvcount=len(yxtup_batchbatch_val)
        args_tuple_val_list=[]
        batchdata_dictlist,modeldict,fof_paramdict=args_tuple
        for v in range(bbvcount):
            yxtup_list_val=yxtup_batchbatch_val[v]
            yxtup_list_val_std=self.standardize_yxtup(yxtup_list_val,modeldict)
            batchdata_dictlist_v=
            args_tuple_val_list.append((batchdata_dictlist_v,modeldict,fof_paramdict))'''
    
    
    def doLoss(self,y,yhat,pthreshold=None,lssfn=None):
        try:
            if not lssfn:lssfn=self.loss_function
            if pthreshold is None:
                    threshold=self.pthreshold    
            
            if lssfn=='mse':
                loss=metrics.mean_squared_error(y,yhat)
            if lssfn=='f1':
                yhat_01=np.zeros(y.shape,dtype=np.float64)
                yhat_01[yhat>threshold]=1
                loss=metrics.f1_score(y,yhat_01)
            elif lssfn=='mae':
                loss=metrics.mean_absolute_error(y,yhat)
            elif lssfn=='logloss':
                yhat_01=np.zeros(y.shape,dtype=np.float64)
                yhat_01[yhat>threshold]=1
                loss=metrics.log_loss(y,yhat_01)
            elif lssfn=='f2':
                yhat_01=np.zeros(y.shape,dtype=np.float64)
                yhat_01[yhat>threshold]=1
                loss=metrics.fbeta_score(y,yhat_01,beta=2)
            elif lssfn=='avg_prec_sc':
                loss=metrics.average_precision_score(y,yhat,average='micro')
            elif lssfn=='splithinge':
                yhat_01=np.zeros(y.shape,dtype=np.float64)
                yhat_01[yhat>threshold]=1
                loss=np.mean((threshold-yhat)*(y-yhat_01))      
            return loss
        except FloatingPointError:
            self.nperror=1
            self.logger.exception('nperror set to 1 to trigger error and big loss')
            return
        except:
            if not self.nperror:
                self.logger.exception(f'y:{y},yhat:{yhat} for self.datagen_dict:{self.datagen_dict}')
                assert False,'unexpected error'
            else:
                return

    def do_naiveloss(self,ylist):
        try:
            y=np.array(ylist)
            ymean=np.mean(y)
            ymeanvec=np.broadcast_to(ymean,y.shape)
            self.sample_ymean=ymean
            if ymean>0.5:yhat=np.ones(y.shape)
            else: yhat=np.zeros(y.shape)
            err=y-ymean
            err2=y-yhat
            
            self.naiveloss=self.doLoss(y,ymeanvec)
            self.naivemse=self.doLoss(y,ymeanvec,lssfn='mse')
            self.naivebinaryloss=self.doLoss(y,yhat,lssfn='mae')
        except:
            self.logger.exception('')
        

    
class optimize_free_params(kNdtool):
    """
    
    """

    def __init__(self,kcsavedir=None,myname=None):
        #np.seterr(over='warn',under='ignore', divide='raise', invalid='raise')
        #np.seterr(over='raise',under='raise', divide='raise', invalid='raise')
        self.datagen_dict=None
        self.opt_settings_dict=None
        self.savepath=None
        self.jobpath=None
        self.yhatmaskscount=None
        self.nperror=0
        self.binary_y_loss_list_list=None
        self.pthreshold=None
        self.nodesavepath=None
        self.naiveloss=None
        self.naivemse=None
        self.ymean=None #all sumstats for pipelines comes from biggest step
        self.ystd=None
        self.sample_ymean=None
        self.naivebinaryloss=None
        self.loss_function=None
        self.validate=None
        kNdtool.__init__(self,savedir=kcsavedir,myname=myname)
        self.pname=myname
        
    
    def run_opt(self,datagen_obj,optimizedict,savedir):
        Ndiff_depth_bwstartingvalue_variations=('hyper_param_dict:Ndiff_depth_bw',list(np.linspace(.2,1,2)))
        self.savedir=savedir
        self.savepath=optimizedict['savepath']
        self.jobpath=optimizedict['jobpath']
        
        try:
            pathpartslist=re.split(os.path.sep,self.savepath)
        except:
            pathpartslist=re.split('\\\\',self.savepath)
        self.nodesavepath=os.path.join('.','results','nodesave',*pathpartslist[2:])
        nodesavedir=os.path.split(self.nodesavepath)[0]
        if not os.path.exists(nodesavedir): os.makedirs(nodesavedir)
        
        
        #self.Ndiff_list_of_masks_x=xmask
        #self.Ndiff_list_of_masks_y=ymask
        
        
        self.call_iter=0#one will be added to this each time the outer loss function is called by scipy.minimize
        self.iter=0
        self.lossdict_and_paramdict_list=[]#will contain a tuple of  (lossdict, fixed_or_free_paramdict) at each call
        self.binary_y_loss_list_list=[]
        self.save_interval=1
        self.datagen_dict=optimizedict['datagen_dict']
        self.source=self.datagen_dict['source']
        
        try:
            self.logger=logging.getLogger(__name__)
            self.logger.info('started optimize_free_params object')
        except:
            logdir=os.path.join(os.getcwd(),'log')
            if not os.path.exists(logdir): os.mkdir(logdir)
            handlername=os.path.join(logdir,f'optimize_free_params-{self.pname}.log')
            logging.basicConfig(
                handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
                level=logging.DEBUG,
                format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%Y-%m-%dT%H:%M:%S')

            #handler=logging.RotatingFileHandler(os.path.join(logdir,handlername),maxBytes=8000, backupCount=5)
            self.logger = logging.getLogger(handlername)


        
        #self.logger.info(f'optimizedict for {self.pname}:{optimizedict}')

        opt_settings_dict=optimizedict['opt_settings_dict']
        method=opt_settings_dict['method']
        self.opt_settings_dict=opt_settings_dict
        opt_method_options=opt_settings_dict['options']
        self.loss_threshold=opt_settings_dict['loss_threshold']
        self.do_minimize=opt_settings_dict['do_minimize']
        
        
        #Extract from optimizedict
        modeldict=optimizedict['modeldict'] 
        
        self.loss_function=modeldict['loss_function']
        self.pthreshold=modeldict['pthreshold']
        if 'validate' in modeldict:
            self.validate=modeldict['validate']
        else:
            self.validate=0
        self.logger.debug(f'self.validate:{self.validate}')
        
        param_valdict=optimizedict['hyper_param_dict']
        self.forcefail=None
        self.success=None
        self.iter=0
        
        if self.validate:
            valdatalist=datagen_obj.yxtup_batchbatch_testf
            bbv=len(valdatalist)
            for v in range(bbv):
                printstring=f'validating {v+1}/{bbv}'
                self.logger.debug(printstring)
                print(printstring)
                valdata=valdatalist[v]
                transformed_free_params,args_tuple=self.prep_KDEreg(
                    datagen_obj,modeldict,param_valdict,self.source,valdata=valdata)
                self.MY_KDEpredictloss(transformed_free_params,*args_tuple,predict=1)
                self.sort_then_saveit(self.lossdict_and_paramdict_list,modeldict,getname=0)
                self.lossdict_and_paramdict_list=[]
                self.binary_y_loss_list_list=[]
            return
                
                    
                    
        transformed_free_params,args_tuple=self.prep_KDEreg(datagen_obj,modeldict,param_valdict,self.source)
        
        
        #startingloss=self.MY_KDEpredictloss(transformed_free_params,*args_tuple, predict=1)
        #self.logger.debug(f'new optimization. starting loss:{startingloss}')
        if type(self.loss_threshold) is str:
                if self.loss_threshold=='naiveloss':
                    self.loss_threshold=self.naiveloss
        
        #if 'species' in self.datagen_dict:
 
            #self.logger.warning(f'no species found in datagen_dict:{self.datagen_dict}', exc_info=True)
        if not self.do_minimize:
            try:
                self.MY_KDEpredictloss(transformed_free_params,*args_tuple, predict=1)
                self.sort_then_saveit(self.lossdict_and_paramdict_list,modeldict,getname=0)
            except:
                self.sort_then_saveit([[{self.loss_function:10.0**290},args_tuple[-1]]],modeldict,getname=0)
                self.logger.exception('problem with exception_model_save')

        else:
            try:
                if self.loss_threshold:
                    self.MY_KDEpredictloss(transformed_free_params,*args_tuple, predict=1)
                    startingloss=self.lossdict_and_paramdict_list[-1][0][self.loss_function]
                    if startingloss>self.loss_threshold:
                        do_opt=0
                        self.sort_then_saveit([[startingloss,args_tuple[-1]]],modeldict,getname=0)
                    else:
                        do_opt=1
                else: do_opt=1
                if do_opt:
                    if self.loss_threshold:
                        self.logger.info(f'-------------starting optimization with loss:{startingloss}-------------')
                    self.minimize_obj=minimize(self.MY_KDEpredictloss, transformed_free_params, args=args_tuple, method=method, options=opt_method_options)
                    
            except:
                self.sort_then_saveit([[10.0**289,args_tuple[-1]]],modeldict,getname=0)
                self.logger.exception('')
        

