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
from Ndiff import Ndiff
from mykernhelper import MyKernHelper

class kNdtool(Ndiff,MyKernHelper):
    """kNd refers to the fact that there will be kernels in kernels in these estimators

    """

    def __init__(self,savedir=None,myname=None):
        if savedir==None: savedir=os.getcwd()
        self.savedir=savedir
        self.name=myname
        self.cores=int(psutil.cpu_count(logical=False)-1)
        
        with open(os.path.join(os.getcwd(),'logconfig.yaml'),'rt') as f:
            configfile=yaml.safe_load(f.read())
        logging.config.dictConfig(configfile)
        self.logger = logging.getLogger('myKernLogger')
        
        Ndiff.__init__(self,)
        MyKernHelper.__init__(self,)
        


    
    ''' def return_param_name_and_value(self,fixed_or_free_paramdict,modeldict):
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
    '''
    """ 
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

    
    def makediffmat_itoj(self,xin,xpr,spatial=None):
        
        diffs= np.expand_dims(xin, axis=1) - np.expand_dims(xpr, axis=0)#should return ninXnoutXp if xin an xpr were ninXp and noutXp
        if spatial==1:
            #assuming the spatial variable is always the last one
            diffs[:,:,-1]=self.myspatialhucdiff(diffs[:,:,-1])
            
        #print('type(diffs)=',type(diffs))
        return diffs
    
    def myspatialhucdiff(self,nparray):#need to rewrite using np.nditer
        print('nparray.shape',nparray.shape)
        rowlist=[]
        for row in nparray:
            arraylist=[]
            for i in row:
                if i>0:
                    arraylist.append(int((np.log(i)+2)/2))
                    #if i is 3, huc10's match, log3<1, so (log(3)+2)/2 is a little over 1, so int returns 1.
                    #if i is 13, huc 10's match, log13>1 so (log(13)+2)/2 is a little over 1.5, so int returns 1.
                    # if i is 100, huc 10's do not match, but huc 8's do. (log(100)+2) is 4 and 4/2 is 2, so int returns 2
                    #if i is 999, huc8's match, log(999)<3so half that plus 2 floors to 2.
                    # if i is 10000, huc8's do not match, log 10000=4, so 4+2 is 6 and 6/2 is 3.
                else:
                    arraylist.append(0)
            rowlist.append(arraylist)
        return np.array(rowlist,dtype=float)

    def MY_KDE_gridprep_smalln(self,m,p):
        '''creates a grid with all possible combinations of m=n^p (kerngrid not nin or nout) evenly spaced values from -3 to 3.'''
        
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
        '''if modeldict['regression_model']=='logistic':
            if type(ykerngrid) is int:
                print(f'overriding modeldict:ykerngrid:{ykerngrid} to {"no"} b/c logisitic regression')
                ykerngrid='no' 
        '''
        #print('type(ykerngrid):',type(ykerngrid))
        ykerngrid_form=modeldict['ykerngrid_form']
        if xpr is None:
            xpr=xdata_std
            #print('1st xpr.shape',xpr.shape)
            
            self.predict_self_without_self='yes'
        elif not xpr.shape==xdata_std.shape: 
            self.predict_self_without_self='n/a'
        elif not np.allclose(xpr,xdata_std):
            self.predict_self_without_self='n/a'
        if type(ykerngrid) is int and xkerngrid=="no":
            yout=self.generate_grid(ykerngrid_form,ykerngrid)#will broadcast later
            #print('yout:',yout)
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
        #print('2nd xpr.shape',xpr.shape)
        #print('xdata_std.shape',xdata_std.shape)
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
        if form[0]=='binary':
            
            return np.linspace(0,1,count)
            
            
    
    def standardize_yx(self,xdata,ydata):
        self.xmean=np.mean(xdata,axis=0)
        self.ymean=np.mean(ydata,axis=0)
        self.xstd=np.std(xdata,axis=0)
        self.ystd=np.std(ydata,axis=0)
        standard_x=(xdata-self.xmean)/self.xstd
        standard_y=(ydata-self.ymean)/self.ystd
        return standard_x,standard_y

    def standardize_yxtup(self,yxtup_list,modeldict):
        #yxtup_list=deepcopy(yxtup_list_unstd)
        p=yxtup_list[0][1].shape[1]
        modelstd=modeldict['std_data']

        try: self.xmean,self.ymean,self.xstd,self.ystd
        except:
            self.xmean=self.datagen_obj.summary_stats_dict['xmean']
            self.ymean=self.datagen_obj.summary_stats_dict['ymean']
            self.xstd=self.datagen_obj.summary_stats_dict['xstd']
            self.ystd=self.datagen_obj.summary_stats_dict['ystd']

        
        if type(modelstd) is str: 
            if  modelstd=='all':
                x_stdlist=[i for i in range(p)]
            else:
                assert False, f'modeldict:std_data is {modelstd} but expected "all"'
        elif type(modelstd) is tuple:
            xmodelstd=modelstd[1]
            ymodelstd=modelstd[0]
            floatselecttup=self.datagen_obj.floatselecttup
            spatialselecttup=self.datagen_obj.spatialselecttup
            if xmodelstd=='float':
                x_stdlist=[i for i in range(len(floatselecttup))]
            if xmodelstd=='all':
                x_stdlist=[i for i in range(p)]
            if ymodelstd=='std':
                y_stdlist=[0]
            if ymodelstd==[]:
                y_stdlist=[]
            #xstdselect=modelstd[1]
            #x_stdlist[modelstd[1]]=1
        
        tupcount=len(yxtup_list)#should be same as batchcount
        
        for i in range(tupcount):
            
            xarray=yxtup_list[i][1]
            for j in x_stdlist:
                xarray[:,j]=(xarray[:,j]-self.xmean[j])/self.xstd[j]
            
            yarray=yxtup_list[i][0]
            if y_stdlist!=[]:
                yarray=(yarray-self.ymean)/self.xstd
            yxtup_list[i]=(yarray,xarray)
                
        return yxtup_list


        
    def ma_broadcast_to(self, maskedarray,tup):
            initial_mask=np.ma.getmask(maskedarray)
            broadcasted_mask=np.broadcast_to(initial_mask,tup)
            broadcasted_array=np.broadcast_to(maskedarray,tup)
            return np.ma.array(broadcasted_array, mask=broadcasted_mask)
            
    def sort_then_saveit(self,mse_param_list,modeldict,filename):
        
        fullpath_filename=os.path.join(self.savedir,filename)
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
    
    #def validate_KDEreg(self,yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict)):"""
        
    
    def BWmaker(self, fixed_or_free_paramdict, diffdict, modeldict,xory):
        if self.Ndiff:
            return self.NdiffBWmaker(modeldict['max_bw_Ndiff'], fixed_or_free_paramdict, diffdict, modeldict,xory)
    
    def MY_KDEpredict(self,yin,yout,xin,xpr,modeldict,fixed_or_free_paramdict):
        """moves free_params to first position of the obj function, preps data, and then runs MY_KDEreg to fit the model
            then returns MSE of the fit 
        Assumes last p elements of free_params are the scale parameters for 'el two' approach to
        columns of x.
        """

        """"""
            
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
        try :
            spatial=self.datagen_obj.spatial
        except:
            spatial=0
        try: spatialtransform=modeldict['spatialtransform']
        except: spatialtransform=None
        if modeldict['Ndiff_bw_kern']=='rbfkern':

            #xin_scaled=xin*x_bandscale_params
            #print('xin_scaled.shape',xin_scaled.shape)
            #xpr_scaled=xpr*x_bandscale_params
            #print('xpr_scaled.shape',xpr_scaled.shape)

            yin_scaled=yin*y_bandscale_params
            yout_scaled=yout*y_bandscale_params
            y_outdiffs=self.makediffmat_itoj(yin_scaled,yout_scaled)
            y_indiffs=self.makediffmat_itoj(yin_scaled,yin_scaled)
            
            outdiffs_scaled_l2norm=np.power(np.sum(np.power(
                self.makediffmat_itoj(xin,xpr,spatial=spatial,spatialtransform=spatialtransform)*x_bandscale_params,2)
                                                   ,axis=2),.5)
            indiffs_scaled_l2norm=np.power(np.sum(np.power(self.makediffmat_itoj(xin,xin,spatial=spatial,spatialtransform=spatialtransform)*x_bandscale_params,2),axis=2),.5)
            assert outdiffs_scaled_l2norm.shape==(xin.shape[0],xpr.shape[0]),f'outdiffs_scaled_l2norm has shape:{outdiffs_scaled_l2norm.shape} not shape:({self.nin},{self.npr})'

            diffdict={}
            diffdict['outdiffs']=outdiffs_scaled_l2norm#ninXnpr?
            diffdict['indiffs']=indiffs_scaled_l2norm#ninXnin?
            ydiffdict={}
            ydiffdict['outdiffs']=np.broadcast_to(y_outdiffs[:,:,None],y_outdiffs.shape+(self.npr,))#ninXnoutXnpr
            ydiffdict['indiffs']=np.broadcast_to(y_indiffs[:,:,None],y_indiffs.shape+(self.npr,))#ninXninXnpr
            diffdict['ydiffdict']=ydiffdict


        if modeldict['Ndiff_bw_kern']=='product':
            outdiffs=makediffmat_itoj(xin,xpr,spatial=spatial)#scale now? if so, move if...='rbfkern' down 
            #predict
            NWtup=self.MY_NW_KDEreg(yin_scaled,xin_scaled,xpr_scaled,yout_scaled,fixed_or_free_paramdict,diffdict,modeldict)[0]
            #not developed yet
        
        xbw = self.BWmaker(fixed_or_free_paramdict, diffdict, modeldict,'x')
        ybw = self.BWmaker(fixed_or_free_paramdict, diffdict['ydiffdict'],modeldict,'y')

        #p#rint('xbw',xbw)
        #p#rint('ybw',ybw)
        
        
        xbwmaskcount=np.ma.count_masked(xbw)
        
        if xbwmaskcount>self.nin:
            self.logger.info(f'xbwmaskcount: {xbwmaskcount}')
            self.logger.warning(f'np.ma.getmask(xbw): {np.ma.getmask(xbw)}')
        
        ybwmaskcount=np.ma.count_masked(ybw)
        if ybwmaskcount>0:
            self.logger.info(f'ybwmaskcount: {ybwmaskcount}, ybw.shape: {ybw.shape}')
            self.logger.info(f'np.ma.getmask(ybw): {np.ma.getmask(ybw)}')

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
            assert xoutdiffs.ndim==2, "xoutdiffs have ndim={} not 2".format(xoutdiffs.ndim)
            ykern_grid=modeldict['ykern_grid'];xkern_grid=modeldict['xkern_grid']
            if True:#type(ykern_grid) is int and xkern_grid=='no':
                xoutdifftup=xoutdiffs.shape[:-1]+(self.nout,)+(xoutdiffs.shape[-1],)
                #print('xoutdiffs.shape',xoutdiffs.shape,'xbw.shape',xbw.shape)
                xoutdiffs_stack=self.ma_broadcast_to(np.expand_dims(xoutdiffs,len(xoutdiffs.shape)-1),xoutdifftup)
                xbw_stack=np.broadcast_to(np.ma.expand_dims(xbw,axis=-2),xoutdifftup)
            newaxis=len(youtdiffs.shape)
            yx_outdiffs_endstack=np.ma.concatenate(
                (np.expand_dims(xoutdiffs_stack,newaxis),np.expand_dims(youtdiffs,newaxis)),axis=newaxis)
            yx_bw_endstack=np.ma.concatenate((np.ma.expand_dims(xbw_stack,newaxis),np.ma.expand_dims(ybw,newaxis)),axis=newaxis)
            #p#rint('type(xoutdiffs)',type(xoutdiffs),'type(xbw)',type(xbw),'type(modeldict)',type(modeldict))
            
            prob_x = self.do_KDEsmalln(xoutdiffs, xbw, modeldict)
            prob_yx = self.do_KDEsmalln(yx_outdiffs_endstack, yx_bw_endstack,modeldict)#
            
            KDEregtup = self.my_NW_KDEreg(prob_yx,prob_x,yout_scaled,modeldict)
            if modeldict['loss_function']=='batchnorm_crossval':
                return KDEregtup
            else:
                yhat_raw=KDEregtup[0]
                cross_errors=KDEregtup[1]
            
            yhat_std=yhat_raw*y_bandscale_params**-1#remove the effect of any parameters applied prior to using y.
        
        std_data=modeldict['std_data']
        
        if type(std_data) is str:
            if std_data=='all':
                yhat_un_std=yhat_std*self.ystd+self.ymean
        elif type(std_data) is tuple:
            if not std_data[0]==[]:
                yhat_un_std=yhat_std*self.ystd+self.ymean
            else: yhat_un_std=yhat_std
        
        #p#rint(f'yhat_un_std:{yhat_un_std}')

        
        if iscrossmse:
            cross_error=cross_errors*self.ystd
        return (yhat_un_std,cross_errors)
        
    def do_batchnorm_crossval(self, KDEregtup,fixed_or_free_paramdict):
        batchcount=self.batchcount
        if batchcount>1:
            yout,wt_stack,cross_errors=zip(*KDEregtup)
            
        else:
            #p#rint('len(KDEregtup)',len(KDEregtup))
            yout,wt_stack,cross_errors=KDEregtup
        nin=self.nin; ybatch=[];wtbatch=[];youtbatch=[]
        
        for i in range(batchcount):
            #i is indexing the batchcount chunks of npr that show up batchcount-1 times in crossvalidation
            #ybatchlist=[]
            wtbatchlist=[]
            youtbatchlist=[]
            
            
            '''crossyoutshape=yout.shape
            crossyoutshape[0]-=1
            crossyout=np.empty(crossyoutshape)
            for i in range(batchcount):
                for j in range(batchcount):
                    if not i==j:
                    crossyout[i,:,:]=yout[]
            '''
            np.repeat
            
            
            for j in range(batchcount):
                if j>i:
                    istart=(i)*nin
                    iend=istart+nin
                    #ybatchlist.append(yout_stack[j][:,istart:iend])
                    wt_i_from_batch_j=wt_stack[j][:,istart:iend]
                    yout_batchj=self.ma_broadcast_to(np.ma.expand_dims(yout[j],axis=-1),(self.nout,self.nin))
                    #p#rint(f'i:{i},j:{j},wt_i_from_batch_j.shape:{wt_i_from_batch_j.shape},istart:{istart},iend:{iend}')
                    wtbatchlist.append(wt_i_from_batch_j)
                    youtbatchlist.append(yout_batchj)
                elif j<i:
                    istart=(i-1)*nin
                    iend=istart+nin
                    #ybatchlist.append(yout_stack[j][:,istart:iend])
                    wt_i_from_batch_j=wt_stack[j][:,istart:iend]
                    yout_batchj=self.ma_broadcast_to(np.ma.expand_dims(yout[j],axis=-1),(self.nout,self.nin))
                    #p#rint(f'i:{i},j:{j},wt_i_from_batch_j.shape:{wt_i_from_batch_j.shape},istart:{istart},iend:{iend}')
                    wtbatchlist.append(wt_i_from_batch_j)
                    youtbatchlist.append(yout_batchj)
                else:
                    pass
            dimcount=np.ndim(wtbatchlist[0])
            #ybatchlist=[np.ma.expand_dims(yi,axis=dimcount) for yi in ybatchlist]
            wtbatchlist=[np.ma.expand_dims(wt,axis=0) for wt in wtbatchlist]
            youtbatchlist=[np.ma.expand_dims(youtj,axis=0) for youtj in wtbatchlist]
            #ybatchshape=[y.shape for y in ybatchlist]
            wtbatchshape=[wt.shape for wt in wtbatchlist]
            #p#rint('wtbatchshape',wtbatchshape)
            #ybatch.append(np.ma.concatenate(ybatchlist,axis=-2))#concatenating on the yout axis for each npr
            wtbatch.append(np.ma.concatenate(wtbatchlist,axis=0))
            youtbatch.append(np.ma.concatenate(youtbatchlist,axis=0))
                                                     
                                                     
        wtstack=np.ma.concatenate(wtbatch,axis=-1)#rhs axis is npr axis
        youtstack=np.ma.concatenate(youtbatch,axis=-1)
        wtstacksum=np.ma.sum(wtstack,axis=0)#summed over the new,batch axis
        wtstacksumsum=np.ma.sum(wtstacksum,axis=0)#summed over the yout axis
        wtstacknorm=wtstack/wtstacksumsum#broadcasting will be automatic since new dimensions are on lhs
        yhat_raw=np.ma.sum(wtstacknorm*youtstack,axis=0)#the npr axis is on rhs, so must be expanded manually. summation of yout axis, the lhs one at this point
        
                
        y_bandscale_params=self.pull_value_from_fixed_or_free('y_bandscale',fixed_or_free_paramdict)
        yhat_std=yhat_raw*y_bandscale_params**-1#remove the effect of any parameters applied prior to using y.
        yhat_un_std=yhat_std*self.ystd+self.ymean
        return yhat_un_std,cross_errors
    
    def do_KDEsmalln(self,diffs,bw,modeldict):
        if self.Ndiff:
            return self.Ndiffdo_KDEsmalln(diffs, bw, modeldict)
        
    
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
            return (yhat,'no_cross_errors')
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
        '''print('yout_axis(expected 0): ',yout_axis)
        print('prob_yx.shape',prob_yx.shape)
        print('prob_x.shape',prob_x.shape)'''
        #prob_yx_sum=np.broadcast_to(np.ma.expand_dims(np.ma.sum(prob_yx,axis=yout_axis),yout_axis),prob_yx.shape)
        #cdfnorm_prob_yx=prob_yx/prob_yx_sum
        #cdfnorm_prob_yx=prob_yx#dropped normalization
        #prob_x_sum=np.broadcast_to(np.ma.expand_dims(np.ma.sum(prob_x, axis=yout_axis),yout_axis),prob_x.shape)
        #cdfnorm_prob_x = prob_x / prob_x_sum
        #cdfnorm_prob_x = prob_x#dropped normalization
        
        #yout_stack=self.ma_broadcast_to(np.ma.expand_dims(yout,1),(self.nout,self.npr))
        yout_stack=np.ma.expand_dims(yout,1)
        prob_x_stack_tup=prob_x.shape[:-1]+(self.nout,)+(prob_x.shape[-1],)
        prob_x_stack=self.ma_broadcast_to(np.ma.expand_dims(prob_x,yout_axis),prob_x_stack_tup)
        NWnorm=modeldict['NWnorm']
                
        if modeldict['regression_model']=='NW-rbf2':
            wt_stack=np.ma.power(np.ma.power(prob_yx,2)-np.ma.power(prob_x_stack,2),0.5)
            if NWnorm=='across':
                wt_stack=wt_stack/np.ma.expand_dims(np.ma.sum(wt_stack,axis=yout_axis),axis=yout_axis)
            yhat=np.ma.sum(yout_stack*wt_stack,axis=yout_axis)#yout axis should be -2

        else:
            wt_stack=prob_yx/prob_x_stack
            if NWnorm=='across':
                wt_stack=wt_stack/np.ma.expand_dims(np.ma.sum(wt_stack,axis=yout_axis),axis=yout_axis)
            yhat=np.ma.sum(yout_stack*wt_stack,axis=yout_axis)#sum over axis=0 collapses across nin for each nout

        yhatmaskscount=np.ma.count_masked(yhat)
        if yhatmaskscount>0:
            self.logger.info(f'in my_NW_KDEreg, yhatmaskscount: {yhatmaskscount}')

        #p#rint(f'yhat:{yhat}')
        #p#rint("wt_stack.shape",wt_stack.shape)
        #self.logger.info(f'type(yhat):{type(yhat)}. yhat: {yhat}')
        
        if not iscrossmse:
            cross_errors='no_cross_errors'
            
            
        if iscrossmse:
            if len(lossfn)>8:
                cross_exp=float(lossfn[8:])
                wt_stack=wt_stack**cross_exp
            
            cross_errors=(yhat[None,:]-yout[:,None])#this makes dim0=nout,dim1=nin
            crosswt_stack=wt_stack/np.ma.expand_dims(np.ma.sum(wt_stack,axis=1),axis=1)
            wt_cross_errors=np.ma.sum(crosswt_stack*cross_errors,axis=1)#weights normalized to sum to 1, then errors summed to 1 per nin
            cross_errors=wt_cross_errors
        if modeldict['loss_function']=='batchnorm_crossval':
            return (yout,wt_stack,cross_errors)
        return (yhat,cross_errors)
            
    
    def predict_tool(self,xpr,fixed_or_free_paramdict,modeldict):
        """
        """
        xpr=(xpr-self.xmean)/self.xstd
        
        self.prediction=MY_KDEpredictMSE(self, free_params, batchdata_dict, modeldict, fixed_or_free_paramdict,predict=None)
        
        return self.prediction.yhat

    def MY_KDEpredictMSE(self, free_params, batchdata_dictlist, modeldict, fixed_or_free_paramdict,predict=None):
        
        #predict=1 or yes signals that the function is not being called for optimization, but for prediction.
        try:
            self.forcefail
            return self.forcefail
        except:
            pass
        if self.call_iter>10:
            try:
                self.success
            except:
                self.forcefail='fail'
        if predict==None or predict=='no':
            predict=0
        if predict=='yes':
            predict=1
        if  type(fixed_or_free_paramdict['free_params']) is str and fixed_or_free_paramdict['free_params'] =='outside':  
            self.call_iter += 1  # then it must be a new call during optimization
        
        batchcount=self.batchcount

        fixed_or_free_paramdict['free_params'] = free_params
        self.fixed_or_free_paramdict = fixed_or_free_paramdict
        try:
            lossfn=modeldict['loss_function']
        except KeyError:
            print(f'loss_function not found in modeldict')
            lossfn='mse'
        iscrossmse=lossfn[0:8]=='crossmse'

        
        if self.source=='monte': 
            yxtup_list=self.datagen_obj.yxtup_list
        batchbatch_all_y_err=[]
        print('self.batchbatchcount',self.batchbatchcount)
        maxbatchbatchcount=modeldict['maxbatchbatchcount']
        batchbatchcount=self.batchbatchcount
        if type(maxbatchbatchcount) is int:
            if maxbatchbatchcount<self.batchbatchcount:
                batchbatchcount=maxbatchbatchcount
            
        for batchbatchidx in range(batchbatchcount):
            print('batchbatchidx:',batchbatchidx)
            if self.source=='pisces':
                yxtup_list=self.datagen_obj.yxtup_batchbatch[batchbatchidx]
            batchdata_dict_i=batchdata_dictlist[batchbatchidx]
            
            y_err_tup = ()

            arglistlist=[] 
            for batch_i in range(batchcount):


                arglist=[]
                arglist.append(batchdata_dict_i['yintup'][batch_i])
                arglist.append(batchdata_dict_i['youttup'][batch_i])
                arglist.append(batchdata_dict_i['xintup'][batch_i])
                arglist.append(batchdata_dict_i['xprtup'][batch_i])

                arglist.append(modeldict)
                arglist.append(fixed_or_free_paramdict)
                arglistlist.append(arglist)

            self.process_count=2#self.cores
            if self.process_count>1 and batchcount>1:
                with multiprocessing.Pool(processes=self.process_count) as pool:
                    yhat_unstd_outtup_list=pool.map(self.MPwrapperKDEpredict,arglistlist)
                    sleep(2)
                    pool.close()
                    pool.join()
            else:
                yhat_unstd_outtup_list=[]
                for i in range(batchcount):
                    result_tup=self.MPwrapperKDEpredict(arglistlist[i])
                    #self.logger.info(f'result_tup: {result_tup}')
                    yhat_unstd_outtup_list.append(result_tup)
            #self.logger.info(f'yhat_unstd_outtup_list: {yhat_unstd_outtup_list}')
            if modeldict['loss_function']=='batchnorm_crossval':
                yhat_unstd,cross_errors=self.do_batchnorm_crossval(yhat_unstd_outtup_list,fixed_or_free_paramdict)

            else:
                if batchcount>1:
                    yhat_unstd,cross_errors=zip(*yhat_unstd_outtup_list)
                else:
                    yhat_unstd,cross_errors=yhat_unstd_outtup_list


            #print(f'after mp.pool,yhat_unstd has shape:{np.shape(yhat_unstd)}')



            if modeldict['loss_function']=='batch_crossval':
                ybatch=[]
                for i in range(batchcount):
                    ycross_j=[]
                    for j,yxvartup in enumerate(yxtup_list):
                        if not j==i:
                            ycross_j.append(yxvartup[0])
                    ybatch.append(np.concatenate(ycross_j,axis=0))
            elif modeldict['loss_function']=='batchnorm_crossval':
                all_y_list=[yxvartup[0] for yxvartup in yxtup_list]
                all_y=np.concatenate(all_y_list,axis=0)
                all_y_err=all_y-yhat_unstd    
                if type(cross_errors[0]) is np.ndarray:
                    cross_errors=np.concatenate(cross_errors,axis=0)

            else:
                ybatch=[tup[0] for tup in yxtup_list]#the original yx data is a list of tupples

            if not modeldict['loss_function']=='batchnorm_crossval':
                for batch_i in range(batchcount):
                    y_batch_i=ybatch[batch_i]
                    y_err = y_batch_i - yhat_unstd[batch_i]
                    y_err_tup = y_err_tup + (y_err,)


                all_y_err = np.ma.concatenate(y_err_tup,axis=0)
            if iscrossmse:
                all_y_err=np.ma.concatenate([all_y_err,np.ravel(cross_errors)],axis=0)
            batchbatch_all_y_err.append(all_y_err)
        batchbatch_all_y_err=np.ma.concatenate([batchbatch_all_y_err],axis=0)
        mse = np.ma.mean(np.ma.power(batchbatch_all_y_err, 2))
        maskcount=np.ma.count_masked(batchbatch_all_y_err)

        if maskcount>1:
            self.logger.warning(f'all_y_err maskcount:{maskcount}')
            self.logger.warning(f'all_y_err mask: {np.ma.getmask(all_y_err)}')
            mse=1000+mse*maskcount**3
        if mse<0:
            mse=-mse*100000
        #assert maskcount==0,f'{maskcount} masked values found in all_y_err'
        
        if predict==0:
            self.mse_param_list.append((mse, deepcopy(fixed_or_free_paramdict)))
            # self.return_param_name_and_value(fixed_or_free_paramdict,modeldict)
            
            t_format = "%Y%m%d-%H%M%S"
            self.iter_start_time_list.append(strftime(t_format))

            if self.call_iter == 3:
                tdiff = np.abs(
                    datetime.datetime.strptime(self.iter_start_time_list[-1], t_format) - datetime.datetime.strptime(
                        self.iter_start_time_list[-2], t_format))
                self.save_interval = int(max([15 - np.round(np.log(tdiff.total_seconds() + 1) ** 3, 0),
                                              1]))  # +1 to avoid negative and max to make sure save_interval doesn't go below 1
                self.logger.info(f'save_interval changed to {self.save_interval}')

            if self.call_iter % self.save_interval == 0:
                self.sort_then_saveit(self.mse_param_list[-self.save_interval * 2:], modeldict, 'model_save')
            if self.call_iter>10 and mse>self.mse_threshold:
                self.forcefail=mse
        self.success=1

        # assert np.ma.count_masked(yhat_un_std)==0,"{}are masked in yhat of yhatshape:{}".format(np.ma.count_masked(yhat_un_std),yhat_un_std.shape)

        return mse

    def MPwrapperKDEpredict(self,arglist):
        #p#rint(f'arglist inside wrapper is:::::::{arglist}')
        yin=arglist[0]
        yout=arglist[1]
        xin=arglist[2]
        xpr=arglist[3]
        modeldict=arglist[4]
        fixed_or_free_paramdict=arglist[5]
        KDEpredict_tup=self.MY_KDEpredict(yin, yout, xin, xpr, modeldict, fixed_or_free_paramdict)
        #p#rint('type(KDEpredict_tup)',type(KDEpredict_tup))
        #try:print(KDEpredict_tup[0].shape)
        #except:pass
        return KDEpredict_tup
    
        '''below functionality moved to datagen.py and accessed as datagen.summary_stats_dict['xmean'],etc
        def batchbatch_stats(self,yxtup_batchbatch):
        all_y=[ii for yxtup_list in yxtup_batchbatch for i in yxtup_list for ii in i[0]]
        all_x=[ii for yxtup_list in yxtup_batchbatch for i in yxtup_list for ii in i[1]]
        self.xmean=np.mean(all_x,axis=0)
        self.ymean=np.mean(all_y,axis=0)
        self.xstd=np.std(all_x,axis=0)
        self.ystd=np.std(all_y,axis=0)'''
                       
                       
    def prep_KDEreg(self,datagen_obj,modeldict,param_valdict,source='monte',predict=None):
        if predict==None:
            predict=0

        #free_params,args_tuple=self.prep_KDEreg(datagen_obj,modeldict,param_valdict)
        if 'max_bw_Ndiff' in modeldict:
            self.Ndiff=1
        else:
            self.Ndiff=0
        try: self.batchbatchcount=datagen_obj.batchbatchcount
            #self.batchbatch_stats(datagen_obj.yxtup_batchbatch) #moved to datagen.py
        except: self.batchbatchcount=1 
        
        self.datagen_obj=datagen_obj
        #self.spatialvar_loc=datagen_obj.spatial_loc
        
        model_param_formdict=modeldict['hyper_param_form_dict']
        xkerngrid=modeldict['xkern_grid']
        ykerngrid=modeldict['ykern_grid']
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        
        #build dictionary for parameters
        free_params,fixed_or_free_paramdict=self.setup_fixed_or_free(model_param_formdict,param_valdict)
        #self.fixed_or_free_paramdict=fixed_or_free_paramdict
        if predict==1:
            fixed_or_free_paramdict['free_params']='predict'#instead of 'outside'
                
        #save and transform the data
        #self.xdata=datagen_obj.x;self.ydata=datagen_obj.y#this is just the first of the batches, if batchcount>1
        
        self.batchcount=datagen_obj.batchcount
        self.nin=datagen_obj.batch_n
        self.p=datagen_obj.param_count#p should work too
        #assert self.ydata.shape[0]==self.xdata.shape[0],'xdata.shape={} but ydata.shape={}'.format(xdata.shape,ydata.shape)

        #standardize x and y and save their means and std to self

        if self.source=='monte':
            yxtup_list=datagen_obj.yxtup_list
        yxtup_listlist_std=[]
        for batchbatchidx in range(self.batchbatchcount):
            if self.source=='pisces':
                yxtup_list=datagen_obj.yxtup_batchbatch[batchbatchidx]
            yxtup_listlist_std.append(self.standardize_yxtup(yxtup_list,modeldict))
        
        #print('buildbatcdatadict')
        batchdata_dictlist=self.buildbatchdatadict(yxtup_listlist_std,xkerngrid,ykerngrid,modeldict)
        #print('for validation buildbatcdatadict')
        #val_batchdata_dict=self.buildbatchdatadict(val_yxtup_list_std,xkerngrid,ykerngrid,modeldict)
        self.npr=len(batchdata_dictlist[0]['xprtup'][0])
        print('self.npr',self.npr)
        #print('=======================')
        #print(f'batchdata_dict{batchdata_dict}')
        #print('=======================')

        #self.npr=xpr.shape[0]#probably redundant
        #self.yout=yout

        #pre-build list of masks
        if 'max_bw_Ndiff' in modeldict:
            #p#rint('---------------starting to make masks----------------')
            self.Ndiff_list_of_masks_y=self.max_bw_Ndiff_maskstacker_y(
                self.npr,self.nout,self.nin,self.p,max_bw_Ndiff,modeldict)
            self.Ndiff_list_of_masks_x=self.max_bw_Ndiff_maskstacker_x(
                self.npr,self.nout,self.nin,self.p,max_bw_Ndiff,modeldict)
            #p#rint('---------------completed making masks----------------')
        
        #setup and run scipy minimize
        args_tuple=(batchdata_dictlist, modeldict, fixed_or_free_paramdict)
        #val_args_tuple=(val_batchdata_dict, modeldict, fixed_or_free_paramdict)
        print(f'mykern modeldict:{modeldict}')
        
        return free_params,args_tuple#,val_args_tuple
    
    
    def buildbatchdatadict(self,yxtup_listlist,xkerngrid,ykerngrid,modeldict):
        #load up the data for each batch into a dictionary full of tuples
        # with each tuple item containing data for a batch from 0 to batchcount-1
        batchcount=self.batchcount
        batchdata_dictlist=[]
        for  yxtup_list in yxtup_listlist:
        
            #print('from buildbatchdatadict: batchcount: ',batchcount)
            #print('self.batchcount: ',self.batchcount)
            xintup = ()
            yintup = ()
            xprtup = ()
            youttup = ()
            if modeldict['loss_function']=='batch_crossval' or modeldict['loss_function']=='batchnorm_crossval':
                #the equivalent condition for the y values in the mse function does not apply to batchnorm_crossval
                xpri=[]
                for i in range(batchcount):
                    xpricross_j=[]
                    for j,yxvartup in enumerate(yxtup_list):
                        if not j==i:
                            xpricross_j.append(yxvartup[1])
                    xpri_crossval_array=np.concatenate(xpricross_j,axis=0)
                        #print('xpri_crossval_array.shape',xpri_crossval_array.shape)
                    xpri.append(xpri_crossval_array)


            else:
                xpri=[None]*batchcount #self.prep_out_grid will treat this as in-sample prediction
            for i in range(batchcount):
                xdata_std=yxtup_list[i][1]
                #print('xdata_std.shape: ',xdata_std.shape)
                ydata_std=yxtup_list[i][0]
                #print('xprii[i]',xpri[i])
                xpr_out_i,youti=self.prep_out_grid(xkerngrid,ykerngrid,xdata_std,ydata_std,modeldict,xpr=xpri[i])
                #print('xpr_out_i.shape',xpr_out_i.shape)
                xintup=xintup+(xdata_std,)
                yintup=yintup+(ydata_std,)
                xprtup=xprtup+(xpr_out_i,)
                youttup=youttup+(youti,)
                #print('youttup',youttup)
                #print('xprtup[0].shape:',xprtup[0].shape)

            batchdata_dict={'xintup':xintup,'yintup':yintup,'xprtup':xprtup,'youttup':youttup}
            batchdata_dictlist.append(batchdata_dict)
        #print([f'{key}:{type(val)},{type(val[0])}' for key,val in batchdata_dict.items()])
        return batchdata_dictlist


    def do_naivemse(self,datagen_obj):
        y=datagen_obj.ydataarray
        ymean=np.mean(y)
        err=y-ymean
        mse=np.mean(np.power(err,2))
        return mse


    
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
        kNdtool.__init__(self,savedir=savedir,myname=myname)
        self.call_iter=0#one will be added to this each time the outer MSE function is called by scipy.minimize
        self.mse_param_list=[]#will contain a tuple of  (mse, fixed_or_free_paramdict) at each call
        self.iter_start_time_list=[]
        self.save_interval=1
        self.datagen_dict=optimizedict['datagen_dict']
        self.source=self.datagen_dict['source']
        self.name=myname

        self.logger.info(f'optimizedict for {myname}:{optimizedict}')

        opt_settings_dict=optimizedict['opt_settings_dict']
        method=opt_settings_dict['method']
        opt_method_options=opt_settings_dict['options']
        self.mse_threshold=opt_settings_dict['mse_threshold']
        
        #Extract from optimizedict
        modeldict=optimizedict['modeldict'] 
        
        param_valdict=optimizedict['hyper_param_dict']

        
        if savedir==None:
            savedir=os.getcwd()
            
        
        self.naivemse=self.do_naivemse(datagen_obj)
            
            
        
        free_params,args_tuple=self.prep_KDEreg(datagen_obj,modeldict,param_valdict,self.source)
        self.minimize_obj=minimize(self.MY_KDEpredictMSE, free_params, args=args_tuple, method=method, options=opt_method_options)
        
        lastmse=self.mse_param_list[-1][0]
        lastparamdict=self.mse_param_list[-1][1]
        self.sort_then_saveit([[lastmse,lastparamdict]],modeldict,'model_save')
        #self.sort_then_saveit(self.mse_param_list[-self.save_interval*3:],modeldict,'final_model_save')
        self.sort_then_saveit(self.mse_param_list,modeldict,'final_model_save')
        self.logger.info(f'after final save, lastparamdict:{lastparamdict}')
        

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

    from random import shuffle,seed
    seed(1)

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
