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
from pi_mp_helper import MatchCollapseHuc12,MpHelper

class Mapper(myLogger):
    def __init__(self):
        myLogger.__init__(self,name='Mapper.log')
        cwd=os.getcwd()
        self.geo_data_dir=os.path.join(cwd,'geo_data')
        self.print_dir=os.path.join(cwd,'print')
        self.boundary_data_path=self.boundaryDataCheck()
        self.boundary_dict={}
        self.pr=PiResults()
        self.fit_scorer=self.pr.fit_scorer
    
      
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
  
    def plot_top_features(self,split=None,top_n=10,rebuild=0,zzzno_fish=False,filter_vars=False,spec_wt=None,fit_scorer=None):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        
        wtd_coef_df=self.pr.build_wt_comid_feature_importance(rebuild=rebuild,zzzno_fish=zzzno_fish,spec_wt=spec_wt,fit_scorer=fit_scorer)
        if split is None:
            cols=wtd_coef_df.columns
            if filter_vars:
                drop_vars=['tmean','tmax','msst','mwst','precip',
                           'slope','wa','elev','mast','tmin','Al2O3']
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
            pos_sort_idx=np.argsort(top_pos_coef_df,axis=1).iloc[:,::-1] #descending
            neg_sort_idx=np.argsort(top_neg_coef_df,axis=1)#.iloc[:,::-1] # ascending
            #select the best and worst columns
            big_top_cols_pos=pos_sort_idx.apply(
                lambda x: big_top_2n[0][x[0]],axis=1)
            big_top_cols_neg=neg_sort_idx.apply(
                lambda x: big_top_2n[1][x[0]],axis=1)
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
    
            
            fig.savefig(Helper().getname(os.path.join(
                self.print_dir,f'huc12_top{top_n}_features.png')))

            
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
    
    def plot_confusion_01predict(self,rebuild=0,fit_scorer=None):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        datadict=self.pr.stack_predictions(rebuild=rebuild)
        wt_df=self.pr.build_wt_comid_feature_importance(rebuild=rebuild,return_weights=True)
        datadict=self.pr.drop_zzz(datadict)
        y=datadict['y']#.drop('zzzno fish',level='species')
        yhat=datadict['yhat']#.drop('zzzno fish',level='species')
        coef_scor_df=datadict['coef_scor_df']#.drop('zzzno fish',level='species')
        csd_vars=coef_scor_df.columns.levels[0].to_list()
        scor_vars=[var for var in csd_vars if var[:7]=='scorer:']
        
        
        
        
        
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
