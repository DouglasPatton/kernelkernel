from multiprocessing import Process,Queue
from time import time,sleep
import re
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from pi_results import PiResults
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helpers import Helper
from mylogger import myLogger



class MatchCollapseHuc12(Process,myLogger):
    def __init__(self,q,huc12list,coef_df,fitscor,adiff_scor_chunk):
        #myLogger.__init__(self,name='MatchCollapseHuc12.log')
        #self.logger.info(f'starting up a MatchCollapseHuc12 proc')
        super().__init__()
        self.q=q
        self.huc12list=huc12list
        self.coef_df=coef_df
        self.fitscor=fitscor
        self.adiff_scor_chunk=adiff_scor_chunk
    
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
            for block_idx in block_selector:
                self.logger.info(f'pid:{pid} on block_idx:{block_idx}')
                huc12block=self.huc12list[block_idx[0]:block_idx[1]]
            #######
                huc12_adiff_scor=self.adiff_scor_chunk.loc[(slice(None),huc12block,slice(None),slice(None)),:]
                huc12_adiff_scor.index=huc12_adiff_scor.index.remove_unused_levels()
                self.huc12_adiff_scor=huc12_adiff_scor
                #this should add comids and huc12s to fitscor
                _,fitscor=huc12_adiff_scor.align(self.fitscor,axis=0,join='left')
                wt=huc12_adiff_scor.multiply(fitscor,axis=0)
                self.wt=wt
                denom=wt.sum(axis=1,level='var')#.sum(axis=0,level='HUC12')
                denom_a,_=denom.align(wt,axis=1,level='var')
                normwt_var=wt.divide(denom_a.values)
                self.denom=denom;self.denom_a=denom_a;#self.denom_aa=denom_aa
                self.normwt_var=normwt_var
                self.huc12_adiff_scor=huc12_adiff_scor
                coef_df=self.coef_df
                normwt_var.columns=normwt_var.columns.droplevel('var')
                _,normwt_var=coef_df.align(normwt_var,axis=0)
                _,normwt_var=coef_df.align(normwt_var,axis=1)
                wt_coef=coef_df.mul(normwt_var,axis=0)
                varwt_coef=wt_coef.sum(axis=1,level='var')#.sum(axis=0,level='HUC12')
                self.varwt_coef=varwt_coef

                wtmean=wt.mean(axis=1,level='var') #for combining across ests/specs, 
                self.wtmean=wtmean
                ##get average weight for each coef
                denom2=wtmean.sum(axis=0,level='HUC12')
                denom2_a,_=denom2.align(wtmean,axis=0)
                #denom2_a.columns=denom2_a.columns.droplevel('var')
                self.denom2=denom2;self.denom2_a=denom2_a
                normwt_huc=wtmean.divide(denom2_a,axis=0)

                self.normwt_huc=normwt_huc
                _,normwt_huc=varwt_coef.align(normwt_huc,axis=0)
                #normwt_huc.columns=normwt_huc.columns.droplevel('var')
                wt2_coef=varwt_coef.mul(normwt_huc,axis=0)
                hucwt_varwt_coef=varwt_coef.sum(axis=0,level='HUC12')
                dflist.append(hucwt_varwt_coef)
            ######## end former for loop
            self.q.put(dflist)
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
        
                

class Mapper(myLogger):
    def __init__(self):
        myLogger.__init__(self,name='Mapper.log')
        cwd=os.getcwd()
        self.geo_data_dir=os.path.join(cwd,'geo_data')
        self.print_dir=os.path.join(cwd,'print')
        self.boundary_data_path=self.boundaryDataCheck()
        self.boundary_dict={}
        self.pr=PiResults()
    
    def runAsMultiProc(self,the_proc,args_list):
        try:
            starttime=time()
            q=Queue()
            q_args_list=[[q,*args] for args in args_list]
            proc_count=len(args_list)
            procs=[the_proc(*q_args_list[i]) for i in range(proc_count)]
            [proc.start() for proc in procs]
            outlist=[]
            countdown=proc_count
        except:
            self.logger.exception('')
            assert False,'unexpected error'
        while countdown:
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
    
    def build_wt_comid_feature_importance(self,rebuild=0,fit_scorer='f1_micro',zzzno_fish=False):#,wt_kwargs={'norm_index':'COMID'}):
        #get data
        name='wt_comid_feature_importance'
        if zzzno_fish:
            name+='zzzno fish'
            
        if not rebuild: #just turning off rebuild here
            try:
                saved_data=self.pr.getsave_postfit_db_dict(name)
                return saved_data['data']#sqlitedict needs a key to pickle and save an object in sqlite
            except:
                self.logger.info(f'rebuilding {name} but rebuild:{rebuild}')
        datadict=self.pr.stack_predictions(rebuild=rebuild)
        
        y=datadict['y']#.astype('Int8')
        yhat=datadict['yhat']#.astype('Int8')
        coef_scor_df=datadict['coef_scor_df']#.astype('float32')
        
        #split coefs and scors and drop rows missing 
        ##coefs from both (e.g., boosting, rbfsvm)
        scor_indices=[]
        coef_indices=[]
        coef_scor_cols=coef_scor_df.columns
        for i,tup in enumerate(coef_scor_cols):
            if tup[0][:7]=='scorer:':
                scor_indices.append(i)
            else:
                coef_indices.append(i)
        scor_df=coef_scor_df.iloc[:,scor_indices]  
        coef_df=coef_scor_df.iloc[:,coef_indices]  
        coef_df=coef_df.dropna(axis=0,how='all')
        self.logger.info(f'BEFORE drop coef_df.shape:{coef_df.shape},scor_df.shape:{scor_df.shape}')
        coef_df,scor_df=coef_df.align(scor_df,axis=0,join='inner')#dropping scors w/o coefs
        self.logger.info(f'AFTER drop coef_df.shape:{coef_df.shape},scor_df.shape:{scor_df.shape}')
        
        #get the scorer
        scor_select=scor_df.loc[:,('scorer:'+fit_scorer,slice(None),slice(None))]
        
        #drop non-linear ests
        for est in ['gradient-boosting-classifier',
                    'hist-gradient-boosting-classifier','rbf-svc']:
            try:yhat.drop(est,level='estimator',inplace=True)
            except:self.logger.exception(f'error dropping est:{est}, moving on')
        
        #make diffs
        self.y=y;self.yhat=yhat
        yhat_a,y_a=yhat.align(y,axis=0)
        adiff_scor=np.exp(yhat_a.subtract(y_a.values,axis=0).abs().mul(-1)) # a for abs
        self.adiff_scor=adiff_scor
        #wt ceofs w/ score
        self.scor_select=scor_select
        self.coef_df=coef_df

        scor_select.columns=scor_select.columns.droplevel('var')
        #coef_a,scor_a=coef_df.align(scor_select,axis=1)
        """need to drop zzzno_fish before summing for weight normalization"""
        if zzzno_fish:
            assert False,'not developed'
        else:
            scor_select.drop('zzzno fish',level='species',inplace=True)
            coef_df.drop('zzzno fish',level='species',inplace=True)
            adiff_scor.drop('zzzno fish',level='species',inplace=True)
        #break into chunks and line up coefs w/ huc12's in groups b/c 
        proc_count=10
        huc12_list=y.index.levels[1].to_list()
        chunk_size=-(-len(huc12_list)//proc_count) # ceiling divide
        huc12chunk=[huc12_list[chunk_size*i:chunk_size*(i+1)] for i in range(proc_count)]
        adiff_scor_chunk=[adiff_scor.loc[(slice(None),huc12chunk[i],slice(None),slice(None)),:] for i in range(proc_count)]
        args_list=[[huc12chunk[i],coef_df,scor_select,adiff_scor_chunk[i]] for i in range(proc_count)]
        q=Queue()
        """self.mch_list=[]
        self.results=[]
        for args in args_list[:1]:
            args=[q,*args]
            mch=MatchCollapseHuc12(*args)
            self.mch_list.append(mch)
            self.results.append(self.mch_list[-1].run())
        
        wtd_coef_df=pd.concat(self.results,axis=0)"""
        print(f'starting {proc_count} procs')
        dflistlist=self.runAsMultiProc(MatchCollapseHuc12,args_list)
        print('multiprocessing complete')
        dflist=[]
        for dfl in dflistlist:
            dflist.extend(dfl)
        self.dflist=dflist
        print('concatenating dflist')
        wtd_coef_df=pd.concat(dflist,axis=0)  
        
            
        #wtd_coef_df=scorwtd_coef_df.multiply(adiff,axis=0)
        self.wtd_coef_df=wtd_coef_df
        
        
        save_data={'data':wtd_coef_df} 
        self.pr.getsave_postfit_db_dict(name,save_data)
        return wtd_coef_df
    
    
    
    
    def boundaryDataCheck(self):
        #https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/WBD%20v2.3%20Model%20Poster%2006012020.pdf
        datalink="https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip"
        boundary_data_path=os.path.join(self.geo_data_dir,'WBD_National_GDB.gdb')
        if not os.path.exists(boundary_data_path):
            assert False, f"cannot locate boundary data. download from {datalink}"
        return boundary_data_path
    
    def getHucBoundary(self,huc_level):
        level_digits=huc_level[-2:]
        if level_digits[0]=='0':
            level_digits=level_digits[1]
        self.boundary_dict[huc_level]=gpd.read_file(self.boundary_data_path,layer=f'WBDHU{level_digits}')
        
    def hucBoundaryMerge(self,data_df,right_on='HUC12'):
        #if right_on.lower()=='huc12':
        huc_level=right_on.lower()
        merge_kwargs={'left_on':huc_level,'right_on':right_on}
        boundary=self.getHucBoundary(huc_level)
        try: self.boundary_dict[huc_level]
        except: self.getHucBoundary(huc_level)
        return self.boundary_dict[huc_level].merge(data_df,left_on=huc_level,right_on=right_on)
  
    def plot_top_features(self,split=None,top_n=10,rebuild=0):
        wtd_coef_df=self.build_wt_comid_feature_importance(rebuild=rebuild)
        if split is None:
            cols=wtd_coef_df.columns
            drop_vars=['tmean','tmax','msst','mwst','precip','slope','wa','elev','mast','tmin']
            for col in cols:
                for varstr in drop_vars:
                    if re.search(varstr,col.lower()):
                        wtd_coef_df.drop(col,axis=1,inplace=True)
                        self.logger.info(f'dropped col: {col}')
                        break #stop searching
            big_mean=wtd_coef_df.mean(axis=0)
            big_cols_sorted=list(big_mean.sort_values().index)[::-1]#descending
            big_top_2n=(big_cols_sorted[:top_n],
                        big_cols_sorted[-top_n:])
            top_pos_coef_df=wtd_coef_df.loc[:,big_top_2n[0]]
            top_neg_coef_df=wtd_coef_df.loc[:,big_top_2n[1]]
            #self.big_top_wtd_coef_df=big_top_wtd_coef_df
            pos_sort_idx=np.argsort(top_pos_coef_df,axis=1).iloc[:,::-1]
            neg_sort_idx=np.argsort(top_neg_coef_df,axis=1)#.iloc[:,::-1]
            #select the best and worst columns
            big_top_cols_pos=pos_sort_idx.apply(
                lambda x: big_top_2n[0][x[0]],axis=1)
            big_top_cols_neg=neg_sort_idx.apply(
                lambda x: big_top_2n[1][x[-1]],axis=1)
            colname='top_predictive_variable'
            big_top_cols_pos=big_top_cols_pos.rename(colname)
            big_top_cols_neg=big_top_cols_neg.rename(colname)
            self.big_top_cols_pos=big_top_cols_pos
            self.big_top_cols_neg=big_top_cols_neg
            
            geo_pos_cols=self.hucBoundaryMerge(big_top_cols_pos)
            geo_neg_cols=self.hucBoundaryMerge(big_top_cols_neg)
               
            fig=plt.figure(dpi=300,figsize=[10,14])
            ax=fig.add_subplot(2,1,1)
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.1)
            geo_pos_cols.plot(column=colname,ax=ax,cmap='tab20c',legend=True)#,legend_kwds={'orientation':'vertical'})
            self.add_huc2_conus(ax)
            
            ax=fig.add_subplot(2,1,2)
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.1)
            geo_neg_cols.plot(column=colname,ax=ax,cmap='tab20c',legend=True)#,legend_kwds={'orientation':'vertical'})
            self.add_huc2_conus(ax)
    
            
            fig.savefig(Helper().getname(os.path.join(self.print_dir,f'huc12_top{top_n}_features.png')))

            
            
            return
        elif split[:3]=='huc': # add a new index level on lhs for huc2,etc
            split=int(split[3:])
            idx1=wtd_coef_df.index
            idx_add=[i[-2][:split] for i in idx1] # idx1 is a list of tuples 
            ##'species','estimator','HUC12','COMID', so [-2]  is 'HUC12'
            idx2=[(idx_add,*idx1[i]) for i in range(len(idx1))] # prepend idx_add
        expanded_midx=pd.MultiIndex.from_tuples(idx2,names=(split,*idx1.names))
        wtd_coef_df2=wtd_coef_df.reindex(index=expanded_midx)
        split_coef_mean=wtd_coef_df2.mean(axis=0,level=split) # mean across huc
        split_coef_rank=np.argsort(split_coef_mean,axis=1)
        
    

    
      
        
    def add_huc2_conus(self,ax):
        try: huc2=self.boundary_dict['huc02']
        except: 
            self.getHucBoundary('huc02')
            huc2=self.boundary_dict['huc02']
        huc2_conus=huc2.loc[huc2.loc[:,'huc2'].astype('int')<19,'geometry']
        huc2_conus.boundary.plot(linewidth=1,color=None,edgecolor='k',ax=ax)
    
    def plot_features_huc12(self,rebuild=0,norm_index='COMID'):
        
        wtd_coef_df=self.build_wt_comid_feature_importance(rebuild=rebuild,norm_index=norm_index)
        if norm_index=='COMID':
            wtd_coef_df.sum(axis=0,level=norm_index)
            huc12_coef_df=wtd_coef_df.mean(axis=0,level='HUC12') # sum b/c normalized 
        elif norm_index.lower()=='huc12':
            huc12_coef_df=wtd_coef_df.mean(axis=0,level='HUC12')
        else: assert False,f'unexpected norm_index:{norm_index}'
        self.huc12_coef_df=huc12_coef_df
        
        columns=list(huc12_coef_df.columns)
        colsort=np.argsort(huc12_coef_df.mean(axis=0).abs())[::-1]
        huc12_coef_gdf=self.hucBoundaryMerge(huc12_coef_df,right_on='HUC12')
        self.huc12_coef_gdf=huc12_coef_gdf
            
        fig=plt.figure(dpi=300,figsize=[10,8])
        ax=fig.add_subplot(1,1,1)
        

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
       
        huc12_coef_gdf.plot(column=columns[colsort[0]],ax=ax,cax=cax ,legend=True)#,legend_kwds={'orientation':'vertical'})
        self.add_huc2_conus(ax)
        fig.savefig(Helper().getname(os.path.join(self.print_dir,'huc12_features.png')))

    
    '''def build_prediction_and_error_dfs(self,rebuild=0):
        agg_prediction_spec_df=self.pr.build_aggregate_predictions_by_species(rebuild=rebuild)
        yhat=agg_prediction_spec_df.drop('y_train',level='estimator').loc[:,agg_prediction_spec_df.columns!='y']
        y=agg_prediction_spec_df.xs('y_train',level='estimator').loc[:,['y']]
        y_a,yhat_a= y.align(yhat,axis=0)
        diff=-1*yhat_a.sub(y_a['y'],axis=0)
        diff.columns=[f'err_{i}' for i in range(diff.shape[1])]
        self.y_a=y_a
        self.yhat_a=yhat_a
        self.diff=diff
        return y_a,yhat_a,diff'''
    
    
    def draw_huc12_truefalse(self,rebuild=0):
        try: self.boundary_dict['huc12']
        except: self.getHucBoundary('huc12')
        y,yhat,diff=self.build_prediction_and_error_dfs(rebuild=rebuild)
        nofish_diff=diff.xs('zzzno fish',level='species').copy(deep=True)
        diff=diff.drop('zzzno fish',level='species')
        
        abs_diff=diff.abs()
        nofish_abs_diff=nofish_diff.abs()
        
        err_geo_df=self.mean_flatten_and_align_huc12(diff)
        nofish_err_geo_df=self.mean_flatten_and_align_huc12(nofish_diff)
        abs_diff_geo_df=self.mean_flatten_and_align_huc12(abs_diff)
        nofish_abs_diff_geo_df=self.mean_flatten_and_align_huc12(nofish_abs_diff)
        
        
        abs_diff_geo_df.loc[:,'err']=-1*abs_diff_geo_df.loc[:,'err'].sub(1)
        rvrs_abs_diff_geo_df=abs_diff_geo_df
        nofish_abs_diff_geo_df.loc[:,'err']=-1*nofish_abs_diff_geo_df.loc[:,'err'].sub(1)
        rvrs_nofish_abs_diff_geo_df=nofish_abs_diff_geo_df
        fig=plt.figure(dpi=300,figsize=[10,8])
        ax=fig.add_subplot(2,1,1)


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        err_geo_df.plot(column='err',ax=ax,cax=cax,legend=True,cmap='brg',)#,legend_kwds={'orientation':'vertical'})
        ax2=fig.add_subplot(2,1,2)
        nofish_err_geo_df.plot(column='err',ax=ax2,cax=cax,legend=True,cmap='brg',)
        
        
        fig.savefig(Helper().getname(os.path.join(self.print_dir,'error_map_with_zzzno-fish.png')))

            
    
    def mean_flatten_and_align_huc12(self,err_df):
        mean_err=err_df.mean(axis=1).mean(level='HUC12')
        err_df=mean_err.rename('err').reset_index()#.rename(columns={'HUC12':'huc12'})# reset_index moves index to columns for merge, rename b/c no name after the mean.
        geo_err_df=self.hucBoundaryMerge(err_df,right_on='HUC12')
        return geo_err_df
