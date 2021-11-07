import os
import re
from multiprocessing import Process,Queue
from mylogger import myLogger
from time import time,sleep
import numpy as np
import pandas as pd

'''

class StackY01Select(Process,myLogger):
    def __init__(self,q,a,y,key,chunks=10):
        self.q=q
        self.a=a
        self.y=y
        self.key=key
        self.chunks=chunks
        
        
    def run(self,):
        try:
            q=self.q
            a=self.a
            y=self.y
            key=self.key
            chunks=self.chunks

            keys=a.index.unique(level=key)
            if chunks<len(keys):
                chunks=len(keys)
            chunksize=-(-len(keys)//chunks)# ceil divide
            a_key_pos=a.index.names.index(key)
            y_key_pos=y.index.names.index(key)
            a_slice=[slice(None) for _ in range(len(a.index.levels))]
            y_slice=[slice(None) for _ in range(len(b.index.levels))]
            dflist=[]
            for ch in range(chunks):
                bite=slice(chunksize*ch,chunksize*(ch+1))
                a_slice[a_key_pos]=bite
                y_slice[y_key_pos]=bite
                a_ch=a.loc[a_slice]
                y_ch=y.loc[a_slice]
                a_,y_=a_ch.align(y_ch,axis=1)
                a__,y__=a__.align(y__,axis=0)
                a1_sel=a__[y__==1]
                a0_sel=a__[y__==0]
                
                
                a_sel=(a0_sel,a1_sel)
                dflist.append(a_sel)
            a0,a1=zip(*dflist)
            a0_df=pd.concat(a0,axis=0)
            a1_df=pd.concat(a1,axis=0)
            a_dfs=[a0_df,a1_df]
            self.q.put(a_dfs)
            #self.q.put(pd.concat(dflist,axis=0))
        except:
            self.logger.exception(f'stackmul outer catch')
            '''

class BatchOverlay(Process,myLogger):
    
    def __init__(self,q,i,gdf,mask,proc_count=8):
        super().__init__()
        self.q=q
        self.i=i
        self.gdf=gdf
        self.mask=mask
        
    def run(self):
        myLogger.__init__(self)
        import geopandas as gpd
        try:
            ch_list=[]
            n=self.gdf.shape[0]
            ch_size=n
            ch_count=-(-n//ch_size)
            for i in range(ch_count):
                start=ch_size*i
                stop=start+ch_size
                if stop>n:stop=n
                ch_list.append(gpd.overlay(self.gdf.iloc[start:stop],self.mask,how='intersection'))
            self.q.put((self.i,ch_list))
        except:
            self.logger.exception(f'error in BatchClipper i:{self.i}')


class MulXB(Process,myLogger):
    def __init__(self,q,i,x,b,wt):
        if not q is None:
            super().__init__()
        #spec_list=b.index.unique(level='species')
        self.i=i
        self.q=q
        self.x=x
        self.b=b
        self.wt=wt
        
    def run(self,):
        myLogger.__init__(self,name='MulXB.log')
        self.logger.info(f'running a MulXB proc')
        
        
        h12_list=self.x.index.unique(level='HUC12')
        h12_count=len(h12_list)
        h12_pos_x=self.x.index.names.index('HUC12')
        x_selector=[slice(None) for _ in range(len(self.x.index.names))]
        if (not self.wt is None) and 'HUC12' in self.wt.index.names:
            h12_pos_wt=self.wt.index.names.index('HUC12')
            wt_selector=[slice(None) for _ in range(len(self.wt.index.names))]
        chunk_n=100
        chunk_count=-(-h12_count//chunk_n)
        results=[]
        #results=None
        try:
            for ch in range(chunk_count):
                h12_ch=h12_list[ch*chunk_n:(ch+1)*chunk_n]
                x_selector[h12_pos_x]=h12_ch
                x_ch=self.x.loc[tuple(x_selector)]
                x_ch.index=x_ch.index.remove_unused_levels()
                x,b=x_ch.align(self.b,axis=0) #add comids,huc12's from bigX
                x,b=x.align(b,axis=1,level='var')#add reps,splits from coefs
                self.x_,self.b_=x,b
                xb=x.mul(b,axis=0)
                if not self.wt is None:
                    if 'HUC12' in self.wt.index.names:
                        wt_selector[h12_pos_wt]=h12_ch
                        wt_ch=self.wt.loc[tuple(wt_selector)]
                    else:
                        self.logger.warning(f'HUC12 not found in wt.index.names:{self.wt.index.names}')
                        wt_ch=self.wt
                    xb,w=xb.align(wt_ch,axis=0)
                    #xb,w=xb.align(w,axis=1)
                    w.columns=w.columns.droplevel('var')
                    #self.xb=xb;self.w=w
                    self.logger.info(f'xb:{xb}')
                    self.logger.info(f'w:{w}')
                    xbw=xb.mul(w,axis=0)
                    xbw_=xbw.sum(axis=1,level='var') 


                    result=xbw_            
                else:
                    result=xb.mean(axis=1,level='var')
                self.logger.info(f'MUlXb appending results {ch+1} of {chunk_count}')
                """if results is None:
                    results=result
                else:
                    results=pd.concat([results,result],axis=0)
                """
                results.append(result)
            #self.logger.info(f'MulXB result:{result}')\
            
            results=pd.concat(results,axis=0)
            if not self.q is None:
                self.q.put((self.i,results))
            else:
                self.result=results
        except:
            self.logger.exception(f'error with MulXB')
            assert False,'halt'
    
class MatchCollapseHuc12(Process,myLogger):
    def __init__(self,q,i,huc12list,coef_df,fitscor,adiff_scor_chunk,y_chunk,spec_wt=None,scale_by_X=False,return_weights=False,presence_filter=False,wt_type='fitscor_diffscor',cv_collapse=False):
        #myLogger.__init__(self,name='MatchCollapseHuc12.log')
        #self.logger.info(f'starting up a MatchCollapseHuc12 proc')
        super().__init__()
        self.i=i
        self.q=q
        self.huc12list=huc12list
        self.coef_df=coef_df
        self.fitscor=fitscor
        self.adiff_scor_chunk=adiff_scor_chunk
        self.y_chunk=y_chunk # a series
        self.spec_wt=spec_wt
        self.scale_by_X=scale_by_X
        self.return_weights=return_weights
        self.presence_filter=presence_filter
        self.wt_type=wt_type
        self.cv_collapse=cv_collapse
        
        self.dflist=[]
        
    
    def run(self,):
        myLogger.__init__(self,name='MatchCollapseHuc12.log')
        self.logger.info(f'running a MatchCollapseHuc12 proc')
        pid=os.getpid()
        self.logger.info(f'mch is running at pid:{pid}')
        try:
            hcount=len(self.huc12list)
            blocksize=10
            blockcount=-(-hcount//blocksize) # ceiling divide
            block_selector=[(blocksize*i,blocksize*(i+1))for i in range(blockcount)]
            for b,block_idx in enumerate(block_selector):
                """if b==2:
                    self.logger.error(f'breaking the block loop for debugging!!!!')
                    break
                """    
                self.logger.info(f'pid:{pid} on block_idx:{block_idx}, {b}/{blockcount}')
                huc12block=self.huc12list[block_idx[0]:block_idx[1]]
                huc12_adiff_scor=self.adiff_scor_chunk.loc[
                    (slice(None),huc12block,slice(None),slice(None)),:]
                huc12_adiff_scor.index=huc12_adiff_scor.index.remove_unused_levels()
                #self.huc12_adiff_scor=huc12_adiff_scor
                #this should add comids and huc12s to fitscor
                wt=self.make_wt(self.fitscor,huc12_adiff_scor,wt_type=self.wt_type)
                if self.return_weights:
                    coef_df=None

                    wt_cvnorm=self.fitscor_diffscor_CV_wtd_mean_coef(wt,coef_df,return_weights=True,cv_collapse=self.cv_collapse)
                    self.dflist.append(wt_cvnorm)
                    continue
                elif not self.scale_by_X:
                    coef_df=self.coef_df
                    varnorm_coef,wt=self.fitscor_diffscor_CV_wtd_mean_coef(
                        wt,coef_df,return_weights=False,cv_collapse=self.cv_collapse)
                    self.varnorm_coef=varnorm_coef

                else: #only use part of coef_df at a time, accomplished by right join above. 
                    h12_pos=self.coef_df.index.names.index('HUC12')
                    selector=[slice(None) for _ in range(len(self.coef_df.index.names))]
                    selector[h12_pos]=huc12block
                    selector=tuple(selector)
                    varnorm_coef=self.coef_df.loc[selector] 
                    varnorm_coef.index=varnorm_coef.index.remove_unused_levels()
                    # coef_df already has been aligned with huc12's,
                    ##so can use .loc on it to make next aligns easy. Also wts already applied 
                    ##and summed over CV when scaled-by-X'''

                if self.presence_filter:
                    y_chunk=self.y_chunk
                    h12_pos=y_chunk.index.names.index('HUC12')
                    selector=[slice(None) for _ in range(len(y_chunk))]
                    selector[h12_pos]=huc12block
                    y_ch_ch=y_chunk.loc[tuple(selector)]
                    y_ch_ch.index=y_ch_ch.index.remove_unused_levels()
                    
                    
                    varnorm_coefs=self.presence_filter_coefs(varnorm_coef,y_ch_ch)
                    hucnorm_varnorm_coefs=[]
                    for varnorm_coef in varnorm_coefs:
                        hucnorm_varnorm_coefs.append(self.huc12_wtd_coef(wt,varnorm_coef))
                    self.dflist.append(hucnorm_varnorm_coefs)
                else:
                #self.logger.info(f'wt:{wt}')
                    hucnorm_varnorm_coef=self.huc12_wtd_coef(wt,varnorm_coef)
                    self.dflist.append(hucnorm_varnorm_coef)
                ######## end for loop
            
            if not self.q is None:
                self.logger.info(f'mch w/pid:{pid} adding to queue')
                self.q.put((self.i,self.dflist))
            else:
                self.result=self.dflist
        except:
            self.logger.exception('')
            
    
    def presence_filter_coefs(self,varnorm_coef,y_chunk):
        '''if type(y_chunk) is pd.DataFrame:
            self.logger.info(f'converting y_chunk to pd.Series')
            y_chunk=y_chunk.loc[:,'y']'''
        #self.logger.info(f'before axis 0 align, varnorm_coef:{varnorm_coef}')
        #self.logger.info(f'and y_chunk:{y_chunk}')
        coef_,y_=varnorm_coef.align(y_chunk,axis=0,join='left') # drop extra hucs and add estimator
        #y_ser=y_.loc[:,'y']
        #coef__,y__=coef_.align(y_ser,axis=1,level='var') # broadcast y across vars,reps,splits
        coef0=coef_.loc[y_.values==0].copy()
        coef1=coef_.loc[y_.values==1].copy()
        self.logger.info(f'presence_filter coef0.shape:{coef0.shape}')
        self.logger.info(f'presence_filter coef1.shape:{coef1.shape}')
        return([coef0,coef1])
    
    def huc12_wtd_coef(self,wt,varnorm_coef):
        try:
            if type(wt) is int:
                return varnorm_coef
            wtmean=wt.mean(axis=1,level='var') #for combining across ests/specs, 
            #self.wtmean=wtmean
            ##get average weight for each coef
            if self.spec_wt=='equal':
                level=['HUC12','species']
            else:
                level='HUC12'

            denom2=wtmean.sum(axis=0,level=level)
            denom2_a,_=denom2.align(wtmean,axis=0)

            #denom2_a.columns=denom2_a.columns.droplevel('var')
            #self.denom2=denom2;self.denom2_a=denom2_a
            normwt_huc=wtmean.divide(denom2_a,axis=0)

            #self.normwt_huc=normwt_huc
            varnorm_coef,normwt_huc=varnorm_coef.align(normwt_huc,axis=0,join='inner')
            #normwt_huc.columns=normwt_huc.columns.droplevel('var')
            wt2_coef=varnorm_coef.mul(normwt_huc.values,axis=0)
            #self.wt2_coef=wt2_coef
            if self.spec_wt=='equal':
                spec_varnorm_coef=wt2_coef.sum(axis=0,level=level)
                hucnorm_varnorm_coef=spec_varnorm_coef.mean(axis=0,level='HUC12')
            else:
                hucnorm_varnorm_coef=wt2_coef.sum(axis=0,level='HUC12')
            #self.hucnorm_varnorm_coef=hucnorm_varnorm_coef
            return hucnorm_varnorm_coef
        except: 
            self.logger.exception(f'outer catch')
    
    def cvnorm_wts(self,wt):
        if not type(wt) is pd.DataFrame:
            return wt
        denom=wt.sum(axis=1,level='var') # sum across rep_idx, cv_idx  #.sum(axis=0,level='HUC12')
        denom_a,_=denom.align(wt,axis=1,level='var')
        wt_cvnorm=wt.divide(denom_a.values)
        return wt_cvnorm
    
    
    def make_wt(self,fitscor,huc12_adiff_scor,wt_type='fitscor_diffscor'):
        _,fitscor_a=huc12_adiff_scor.align(fitscor,axis=0,join='left')
        
        
        if re.search('fit',wt_type):
            fit=True
        else:
            fit=False
        if re.search('diff',wt_type):
            diff=True
        else:
            diff=False
            
        if diff and fit:
            fitscor_a.columns=fitscor_a.columns.droplevel('var')
            wt=huc12_adiff_scor.multiply(fitscor_a,axis=0)     
        elif fit:
            wt=fitscor_a
        elif diff:
            wt=huc12_adiff_scor
        else:
            self.logger.warning(f'wt_type:{wt_type}, fit:{fit}, diff:{diff} so setting wt=1')
            wt=1
        return wt
    
    def fitscor_diffscor_CV_wtd_mean_coef(self,wt,coef_df,return_weights=False,cv_collapse=False):
        
        
        if not cv_collapse==True:
            if cv_collapse=='split':
                wt=wt.mean(axis=1,level=['rep_idx','var'])
                if not return_weights:
                    coef_df=coef_df.mean(axis=1,level=['rep_idx','var'])
            wt_cvnorm=self.cvnorm_wts(wt)
            #self.wt=wt

            #self.denom=denom;self.denfom_a=denom_a;#self.denom_aa=denom_aa
            #self.wt_cvnorm=wt_cvnorm
            #self.huc12_adiff_scor=huc12_adiff_scor
            if return_weights:
                return wt_cvnorm #move to next block
            ###coef_df enters
            if type(wt_cvnorm) in [int,str]:
                return coef_df,wt

            wt_cvnorm.columns=wt_cvnorm.columns.droplevel('var')
            coef_df,wt_cvnorm=coef_df.align(wt_cvnorm,axis=0,join='right')
            coef_df,wt_cvnorm=coef_df.align(wt_cvnorm,axis=1)
            wtd_coef=coef_df.mul(wt_cvnorm,axis=0)
            varnorm_coef=wtd_coef.sum(axis=1,level='var')# 
            return varnorm_coef,wt
        else:
            if type(wt) is int:
                pass
            else:
                wt_clps=wt.mean(axis=1,level='var')
            if return_weights:
                return wt_clps
            coef_df_clps=coef_df.mean(axis=1,level='var')
            return coef_df_clps,wt
        
                    
            
            
class MpHelper(myLogger):   
    def __init__(self):
        myLogger.__init__(self,name='mphelper.log')
        
            
    def runAsMultiProc(self,the_proc,args_list,kwargs={},no_mp=False,concat=None):
        try:
            starttime=time()
            if no_mp:q=None
            else:q=Queue()
            I=len(args_list)
            q_args_list=[[q,i,*args_list[i]] for i in range(I)]
            proc_count=I
            procs=[the_proc(*q_args_list[i],**kwargs) for i in range(proc_count)]
            if no_mp:
                self.procs=procs
                if type(no_mp) is int:procs=procs[-no_mp:]
                self.procs=procs
                for proc in procs:
                    proc.run()
                self.logger.info(f'procs have run with no_mp:{no_mp}')
                results=[proc.result for proc in procs]
                return results
            else:
                [proc.start() for proc in procs]
            if concat is None:
                outlist=[None for _ in range(I)]
            else:
                outdf=pd.DataFrame()
            countdown=proc_count
        except:
            self.logger.exception('error in runasmultiproc')
            assert False,'unexpected error'
        if no_mp:
            self.logger.warning(f'something went wrong with no_mp')
            return
        while countdown and not no_mp:
            try:
                self.logger.info(f'multiproc checking q. countdown:{countdown}')
                i,result=q.get(True,20)
                self.logger.info(f'multiproc has something from the q!')
                if concat is None:
                    outlist[i]=result
                else:
                    outdf=pd.concat([outdf,result],axis=concat)
                countdown-=1
                self.logger.info(f'proc completed. countdown:{countdown}')
            except:
                #self.logger.exception('error')
                if not q.empty(): self.logger.exception(f'error while checking q, but not empty')
                else: sleep(5)
        [proc.join() for proc in procs]
        q.close()
        self.logger.info(f'all procs joined sucessfully')
        endtime=time()
        self.logger.info(f'pool complete at {endtime}, time elapsed: {(endtime-starttime)/60} minutes')
        if concat is None:
            return outlist
        else:
            return outdf
    