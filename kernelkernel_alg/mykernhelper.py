import re
import numpy as np
import os
from time import strftime,sleep
import pickle
import datetime
import logging
from math import log

class MyKernHelper:
    def __init__(self,):
        try: self.logger
        except:
            self.logger=logging.getLogger(__name__)
        self.logger.critical('MyKernHelper is logging')
        
    def process_predictions(self,y,yhatdict):
        resultsdict={}
        for model_name,yhat in yhatdict.items():
            resultsdict[model_name]={}
            for lf in self.lossdict: #just the keys
                resultsdict[model_name][lf]=self.doLoss(y,yhat,lssfn=lf)
        return resultsdict
    
    
    def sort_then_saveit(self,lossdict_and_paramdict_list,modeldict,getname=0):

        try:
            
            species='species-'+self.datagen_dict['species']+'_'
        except AttributeError:
            self.logger.exception('self.datagen_dict not found in object')
            
        except KeyError:
            self.logger.Warning(f'no species found in self.datagen_dict', exc_info=True)
            species=''
        except:
            self.logger.exception('something happened when pulling species from self.datagen_dict')
        
        fullpath_filename=self.nodesavepath
        if getname:
            fullpath_filename=Helper().getname(fullpath_filename)
        lossfn=modeldict['loss_function']    
        
        losslist=[lossdict[lossfn] for lossdict,paramdict in lossdict_and_paramdict_list]
        minloss=min(losslist)
        thisloss=losslist[-1]
        bestiter_pos=losslist.index(minloss)
        bestlossdict,bestparams=lossdict_and_paramdict_list[bestiter_pos]
        thislossdict,thisparams=lossdict_and_paramdict_list[-1]
        savedict={}
        savedict['lossdict']=thislossdict
        savedict['loss']=thisloss
        savedict['naiveloss']=self.naiveloss
        #savedict['xdata']=self.xdata
        #savedict['ydata']=self.ydata
        savedict['params']=thisparams
        #if binary_y_loss_list is None:
            
        best_binary_y_loss_list=self.binary_y_loss_list_list[bestiter_pos]
        this_binary_y_loss_list=self.binary_y_loss_list_list[-1]
        #self.logger.debug(f'len(self.binary_y_loss_list_list):{len(self.binary_y_loss_list_list)}, self.binary_y_loss_list_list:{self.binary_y_loss_list_list}')
        try:
            if modeldict['binary_y'] is None:
                savedict['binary_y_result']=[]
            else:
                savedict['binary_y_result']=this_binary_y_loss_list.copy()
                savedict['binary_y_result'].append((f'sample_ymean:{self.sample_ymean}p=0.5:',self.naivebinaryloss))
        except:
            self.logger.exception('')
        savedict['modeldict']=modeldict
        now=strftime("%Y%m%d-%H%M%S")
        savedict['when_saved']=now
        savedict['datagen_dict']=self.datagen_dict
        savedict['savepath']=self.savepath
        savedict['jobpath']=self.jobpath
        savedict['opt_settings_dict']=self.opt_settings_dict
        if self.other_estimator_test_loss_dict:
            savedict['other_estimator_test_loss_dict']=self.other_estimator_test_loss_dict
        
        try:#this is only relevant after optimization completes
            savedict['minimize_obj']=self.minimize_obj
        except:
            pass
        savedict['do_minimize']=self.do_minimize
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
        
        if len(modellist)>0:
            lastsavetime=modellist[-1]['when_saved']
            runtime=datetime.datetime.strptime(now,"%Y%m%d-%H%M%S")-datetime.datetime.strptime(lastsavetime,"%Y%m%d-%H%M%S")
            print(f'time between saves for {self.pname} is {runtime}')
        modellist.append(savedict)
        
        for i in range(10):
            try:
                with open(fullpath_filename,'wb') as thefile:
                    pickle.dump(modellist,thefile)
                donestring=(f'saved to {fullpath_filename} at about {strftime("%Y%m%d-%H%M%S")} naiveloss,'
                    f'loss={(self.naiveloss,minloss)} and naivemse,mse,{(self.naivemse,bestlossdict["mse"])},'
                    f'and best_binary_y_loss_list:{best_binary_y_loss_list},this_binary_y_loss_list:{this_binary_y_loss_list}')
                print(donestring)
                print(f'bestparams:{bestparams}')
                print(f'thislossdict:{thislossdict}, thisparams:{thisparams}')
                self.logger.info(donestring)
                self.logger.debug(f'bestparams:{bestparams}')
                self.logger.debug(f'thislossdict:{thislossdict}, thisparams:{thisparams}')
                break
            except:
                if i==9:
                    self.logger.exception('')
                    print(f'mykern.py could not save to {fullpath_filename} after {i+1} tries')
        return (bestlossdict,bestparams)
    
    def doBinaryThreshold(self,y,yhat,threshold=None):
        try:
            #self.logger.debug(f'dobinaryThreshold started with threshold:{threshold}')
            if not threshold:
                threshold=self.pthreshold

            if type(threshold) is float:
                binary_yhat=np.zeros(yhat.shape)
                binary_yhat[yhat>threshold]=1
                binary_yhat[yhat>1]=yhat[yhat>1] # keep bad guesses bad so loss_threshold throws them out
                yhat=binary_yhat
            if type(threshold) is tuple:
                this_binary_y_loss_list=[]
                for threshold_i in threshold:
                    if type(threshold_i) is str:
                        #print(f'y.shape and yhat.shape:{y.shape} and {yhat.shape}')
                        if threshold_i=='avgavg':
                            avg_phat_0=np.mean(yhat[y==0])
                            avg_phat_1=np.mean(yhat[y==1])
                            threshold_i=(avg_phat_0+avg_phat_1)/2
                        elif threshold_i=='avgmedian':
                            median_phat_0=np.median(yhat[y==0])
                            median_phat_1=np.median(yhat[y==1])
                            threshold_i=(median_phat_0+median_phat_1)/2


                    binary_yhat=np.zeros(yhat.shape)
                    binary_yhat[yhat>threshold_i]=1
                    threshloss=self.doLoss(y,binary_yhat,lssfn='mae')#(np.mean(np.power(y-binary_yhat,2)))
                    this_binary_y_loss_list.append((threshold_i,threshloss))
                    #self.logger.debug(f'this_binary_y_loss_list:{this_binary_y_loss_list}')
                self.binary_y_loss_list_list.append(this_binary_y_loss_list)
        except: 
            self.logger.exception(f'unexpected error')
    
    
    def return_param_name_and_value(self,fixed_or_free_paramdict,modeldict):
        params={}
        paramlist=[key for key in modeldict['hyper_param_form_dict']]
        for param in paramlist:
            paramdict=fixed_or_free_paramdict[param]
            form=paramdict['fixed_or_free']
            const=paramdict['const']
            start,end=paramdict['location_idx']
            
            value=fixed_or_free_paramdict[f'{form}_params'][start:end]
            if const=='non-neg':
                const=f'{const}'+':'+f'{np.abs(value)} error not developed'
            params[param]={'value':value,'const':const}
        return params
    
    
    def insert_detransformed_freeparams(self, fixed_or_free_paramdict,free_params):
        new_params=np.empty(free_params.shape,dtype=np.float64)
        for param_name,param_feature_dict in fixed_or_free_paramdict.items():
            if not param_name in ['fixed_params','free_params']:
                const_val=param_feature_dict['const']
                if const_val=='non-neg':
                    for i in range(*param_feature_dict['location_idx']):
                        new_params[i]=np.exp(free_params[i])
                elif const_val[:4]=='ball': #https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19760004646.pdf
                    for i in range(*param_feature_dict['location_idx']):
                        new_params[i]=self.doball(free_params[i],const_val,transform=-1)
                
                    
        print(f'free_params:{free_params},new_params:{new_params}')    
        fixed_or_free_paramdict['free_params']=new_params
        return fixed_or_free_paramdict
    
    def pull_value_from_fixed_or_free(self,param_name,fixed_or_free_paramdict,transform=None):
        if transform is None:
            transform=0

        start,end=fixed_or_free_paramdict[param_name]['location_idx']
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='fixed':
            the_param_values=fixed_or_free_paramdict['fixed_params'][start:end]#end already includes +1 to make range inclusive of the end value
        if fixed_or_free_paramdict[param_name]['fixed_or_free']=='free':
            the_param_values=fixed_or_free_paramdict['free_params'][start:end]#end already includes +1 to make range inclusive of the end value
        if transform==1:
            assert False,'no longer relevant'
            if fixed_or_free_paramdict[param_name]['const']=='non-neg':#transform variable with e^(.) if there is a non-negative constraint
                the_param_values=np.abs(the_param_values)
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
        fixed_params=np.array([],dtype=np.float64);free_params=np.array([],dtype=np.float64);fixed_or_free_paramdict={}
        #build fixed and free vectors of hyper-parameters based on hyper_param_formdict
        print(f'param_valdict:{param_valdict}')
        for param_name,param_form in model_param_formdict.items():
            param_feature_dict={}
            param_val=param_valdict[param_name]
            #p#rint('param_val',param_val)
            #p#rint('param_form',param_form)
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
                if param_form == 'non-neg':
                    param_val=np.log(param_val)
                if param_form[:4]=='ball': #https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19760004646.pdf
                    param_val=self.doball(param_val,param_form,transform=1)
                    
                
                free_params=np.concatenate([free_params,param_val],axis=0)
                fixed_or_free_paramdict[param_name]=param_feature_dict
        fixed_or_free_paramdict['free_params']='outside'
        fixed_or_free_paramdict['fixed_params'] = fixed_params
        
        #self.logger.info(f'setup_fixed_or_free_paramdict:{fixed_or_free_paramdict}')
        return free_params,fixed_or_free_paramdict
    
    def doball(self,val,ballstring,transform=0):
        #self.logger.debug(f'val,ballstring,transform:{val,ballstring,transform}')
        endstring=ballstring[5:]#dropping ball_
        a,b=re.split('_',endstring)
        a=float(a)
        b=float(b)
        scale=(b-a)/2
        shift=(b+a)/2
        if transform==1:
            result=np.arcsin((val-shift)/scale)*2/np.pi
            if np.isnan(result).any() or np.isinf(result).any():
                self.logger.error(f'{result} from val:{val},a:{a},b:{b}!')
            return result
        elif transform==-1:
            result=scale*np.sin(np.pi/2*val)+shift
            if result.min()<a or result.max()>b:
                self.logger.error(f'outside range! {result} from val:{val},a:{a},b:{b}!')
            return result
        else: assert False, f"unexpected transform:{transform}"

    
    def makediffmat_itoj(self,x1,x2,spatial=None,spatialtransform=None):
        try:
            #if spatial:
            #    spatial_p=-1#x1.shape[-2]-1#this will be the last item along that axis
            #    self.logger.debug(f'spatial_p:{spatial_p}')
            #x1_ex=np.abs(np.expand_dims(x1, axis=1))
            #x2_ex=np.abs(np.expand_dims(x2, axis=0))#x1.size,x2.size,p,batch
            self.logger.info(f'spatial:{spatial},spatialtransform:{spatialtransform}')
            self.logger.info(f'x1.shape:{x1.shape},x1:{x1}')
            self.logger.info(f'x2.shape:{x2.shape},x2:{x2}')
            diffs= np.abs(np.expand_dims(x1, axis=1) - np.expand_dims(x2, axis=0))#should return ninXnoutXpxbatchcount if xin an xpr were ninXpXbatchcount and noutXpXbatchcount
        
            if spatial:
                #assuming the spatial variable is always the last one
                diffs[:,:,-1,:][diffs[:,:,-1,:]>0]=np.int_(np.log10(diffs[:,:,-1,:][diffs[:,:,-1,:]>0])/2)+1

                #diffs[:,:,-1,:]=self.myspatialhucdiff(diffs[:,:,-1,:]) # dims are (nin,nout,p,batch)
                if type(spatialtransform) is tuple:

                    if spatialtransform[0]=='divide':
                        diffs[:,:-1,:]=diffs[:,:-1,:]/spatialtransform[1]
                    if spatialtransform[0]=='ln1':
                        diffs[:,:-1,:]=np.log(diffs[:,:-1,:]+1)
                    if spatialtransform[0]=='norm1':
                        diffs[:,:-1,:]=diffs[:,:-1,:]/diffs[:,:-1,:].max()
            #print('type(diffs)=',type(diffs))
            return diffs
            
        except FloatingPointError:
            self.nperror=1
            self.logger.exception('nperror set to 1 to trigger error and big loss')
            assert False, 'makediffmat floatingpoint error!'
            return
        except:
            if not self.nperror:
                self.logger.exception('')
                assert False,'unexpected error'
            else:
                return
                     

    def myspatialhucdiff(self,nparray):#need to rewrite using np.nditer
        #print('nparray.shape',nparray.shape)
        rowlist=[]
        for row in nparray:
            arraylist=[]
            for i in row:
                if i>0:
                    arraylist.append(int((np.log10(i)+2)/2))
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
            

    


    
    
