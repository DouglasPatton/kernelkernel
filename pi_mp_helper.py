import os
from multiprocessing import Process,Queue
from mylogger import myLogger
from time import time,sleep
import numpy as np

class MulXB(Process,myLogger):
    def __init__(self,q,x,b,wt):
        if not q is None:
            super().__init__()
        #spec_list=b.index.unique(level='species')
        self.q=q
        self.x=x
        self.b=b
        self.wt=wt
        
    def run(self,):
        myLogger.__init__(self,name='MulXB.log')
        self.logger.info(f'running a MulXB proc')
        try:
            x,b=self.x.align(self.b,axis=0) #add comids,huc12's from bigX
            x,b=x.align(b,axis=1,level='var')#add reps,splits from coefs
            self.x_,self.b_=x,b
            xb=x.mul(b,axis=0)
            if not self.wt is None:
                xb,w=xb.align(self.wt,axis=0)
                #xb,w=xb.align(w,axis=1)
                w.columns=w.columns.droplevel('var')
                self.xb=xb;self.w=w
                self.logger.info(f'xb:{xb}')
                self.logger.info(f'w:{w}')
                xbw=xb.mul(w,axis=0)
                xbw_=xbw.sum(axis=1,level='var') 
                result=xbw_            
            else:
                result=xb
            self.logger.info(f'MulXB result:{result}')
            if not self.q is None:
                self.q.put(result)
            else:
                self.result=result
        except:
            self.logger.exception(f'error with MulXB')
            assert False,'halt'
    
class MatchCollapseHuc12(Process,myLogger):
    def __init__(self,q,huc12list,coef_df,fitscor,adiff_scor_chunk,spec_wt=None):
        #myLogger.__init__(self,name='MatchCollapseHuc12.log')
        #self.logger.info(f'starting up a MatchCollapseHuc12 proc')
        super().__init__()
        self.q=q
        self.huc12list=huc12list
        self.coef_df=coef_df
        self.fitscor=fitscor
        self.adiff_scor_chunk=adiff_scor_chunk
        self.spec_wt=spec_wt
    
    def run(self,):
        myLogger.__init__(self,name='MatchCollapseHuc12.log')
        self.logger.info(f'running a MatchCollapseHuc12 proc')
        pid=os.getpid()
        self.logger.info(f'mch is running at pid:{pid}')
        try:
            dflist=[]
            hcount=len(self.huc12list)
            blocksize=100
            blockcount=-(-hcount//blocksize) # ceiling divide
            block_selector=[(blocksize*i,blocksize*(i+1))for i in range(blockcount)]
            for b,block_idx in enumerate(block_selector):
                self.logger.info(f'pid:{pid} on block_idx:{block_idx}, {b}/{blockcount}')
                huc12block=self.huc12list[block_idx[0]:block_idx[1]]
                huc12_adiff_scor=self.adiff_scor_chunk.loc[(slice(None),huc12block,slice(None),slice(None)),:]
                huc12_adiff_scor.index=huc12_adiff_scor.index.remove_unused_levels()
                self.huc12_adiff_scor=huc12_adiff_scor
                #this should add comids and huc12s to fitscor
                _,fitscor=huc12_adiff_scor.align(self.fitscor,axis=0,join='left')
                wt=huc12_adiff_scor.multiply(fitscor,axis=0)
                self.wt=wt
                if self.coef_df is None:
                    dflist.append(wt)
                else: 
                    denom=wt.sum(axis=1,level='var') # sum across rep_idx, cv_idx  #.sum(axis=0,level='HUC12')
                    denom_a,_=denom.align(wt,axis=1,level='var')
                    normwt_var=wt.divide(denom_a.values)
                    self.denom=denom;self.denom_a=denom_a;#self.denom_aa=denom_aa
                    self.normwt_var=normwt_var
                    self.huc12_adiff_scor=huc12_adiff_scor
                   
                    ###coef_df enters
                    coef_df=self.coef_df
                    normwt_var.columns=normwt_var.columns.droplevel('var')
                    coef_df,normwt_var=coef_df.align(normwt_var,axis=0,join='right')
                    coef_df,normwt_var=coef_df.align(normwt_var,axis=1)
                    wt_coef=coef_df.mul(normwt_var,axis=0)
                    varwt_coef=wt_coef.sum(axis=1,level='var')#.sum(axis=0,level='HUC12')
                    self.varwt_coef=varwt_coef
                    #self.logger.info(f'wt:{wt}')
                    wtmean=wt.mean(axis=1,level='var') #for combining across ests/specs, 
                    self.wtmean=wtmean
                    ##get average weight for each coef
                    if self.spec_wt=='equal':
                        level=['HUC12','species']
                    else:
                        level='HUC12'



                    denom2=wtmean.sum(axis=0,level=level)
                    denom2_a,_=denom2.align(wtmean,axis=0)

                    #denom2_a.columns=denom2_a.columns.droplevel('var')
                    self.denom2=denom2;self.denom2_a=denom2_a
                    normwt_huc=wtmean.divide(denom2_a,axis=0)

                    self.normwt_huc=normwt_huc
                    varwt_coef,normwt_huc=varwt_coef.align(normwt_huc,axis=0,join='inner')
                    #normwt_huc.columns=normwt_huc.columns.droplevel('var')
                    wt2_coef=varwt_coef.mul(normwt_huc.values,axis=0)
                    self.wt2_coef=wt2_coef
                    if self.spec_wt=='equal':
                        spec_varwt_coef=wt2_coef.sum(axis=0,level=level)
                        hucwt_varwt_coef=spec_varwt_coef.mean(axis=0,level='HUC12')
                    else:
                        hucwt_varwt_coef=wt2_coef.sum(axis=0,level='HUC12')
                    self.hucwt_varwt_coef=hucwt_varwt_coef
                    dflist.append(hucwt_varwt_coef)
            ######## end former for loop
            self.logger.info(f'mch w/pid:{pid} adding to queue')
            if not q is None:
                self.q.put(dflist)
            else:
                self.result=dflist
            """t=0
            while t<5
                try:
                    self.q.put(hucwt_varwt_coef,10) #since no dflist w/o loop
                    return
                except:
                    self.logger.exception(f'error adding to q')
                    t+=1
            self.logger.error(f'failed to add job to q')    
            self.logger.error(f'hucwt_varwt_coef:{hucwt_varwt_coef}')
                
            #return chunk_df"""
        except:
            self.logger.exception('')
            
class MpHelper(myLogger):   
    def __init__(self):
        myLogger.__init__(self,name='mphelper.log')
        
            
    def runAsMultiProc(self,the_proc,args_list,kwargs={},no_mp=False):
        try:
            starttime=time()
            if no_mp:q=None
            else:q=Queue()
            q_args_list=[[q,*args] for args in args_list]
            proc_count=len(args_list)
            procs=[the_proc(*q_args_list[i],**kwargs) for i in range(proc_count)]
            if no_mp:
                self.procs=procs
                for proc in procs:
                    proc.run()
                self.logger.info(f'procs have run with no_mp:{no_mp}')
                results=[proc.result for proc in procs]
                return results
            else:
                [proc.start() for proc in procs]
            outlist=[]
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
                result=q.get(True,20)
                self.logger.info(f'multiproc has something from the q!')
                outlist.append(result)
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
        return outlist
    