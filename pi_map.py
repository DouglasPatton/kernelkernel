from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import shapely

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
        if level_digits[0]=='0' or not level_digits[0].isdigit():
            level_digits=level_digits[1] #e.g. HUC02 --> 2
        return gpd.read_file(self.boundary_data_path,layer=f'WBDHU{level_digits}')
        
    def hucBoundaryMerge(self,data_df,right_on='HUC12'):
        # hucboundary files have index levels like 'huc2' not 'HUC02'
        #if right_on.lower()=='huc12':
        if type(huc_level) is str:
            huc_level=right_on.lower()
        level_digits=huc_level[-2:]
        if huc_level[-2]=='0':
            huc_level=huc_level[:-2]+huc_level[-1]
        h12_list=data_df.index.unique(level=right_on)
        boundary=self.getHucBoundary(huc_level)
        #print('huc_level',huc_level)
        #print('boundary.index.names',boundary.index.names.index)
        #boundary_huc_pos=boundary.columns.names.index(huc_level)
        #selector=[slice(None) for _ in range(len())]
        #selector[boundary_huc_pos]=h12_list
        boundary_clip=boundary.loc[boundary[huc_level].isin(h12_list)]
        #boundary_clip=boundary.loc[selector]
        return boundary_clip.merge(data_df,left_on=huc_level,right_on=right_on)
    
    def plot_y01(self,zzzno_fish=False,rebuild=0,huc_level=None):
        coef_df,scor_df,y,yhat=self.pr.get_coef_stack(
            rebuild=rebuild,drop_zzz=not zzzno_fish,return_y_yhat=True,
            drop_nocoef_scors=False)
        
        h12_y1sum=y.sum(axis=0,level=['HUC12','COMID']).mean(axis=0,level='HUC12')
        h12_y1sum.columns=['species found']
        h12_y0sum=(1-y).sum(axis=0,level=['HUC12','COMID']).mean(axis=0,level='HUC12')
        h12_y0sum.columns=['species not found']
        h12_y1mean=y.mean(axis=0,level=['HUC12','COMID']).mean(axis=0,level='HUC12')
        h12_y1mean.columns=['share of species found']
        h12_ycount=h12_y1sum+h12_y0sum.values
        h12_ycount.columns=['potential species found']
        dfs=pd.concat([h12_y1sum,h12_y0sum,h12_y1mean,h12_ycount])
        cols=list(dfs.columns)
        c=2 # c* of olumns
        r=-(-len(cols)//c)
        tuplist=[(r,c,i+1) for i in range(len(cols))]
        #tuplist=[(len(cols),x+1,c) for x in range(len(cols)//c) for y in range(r)]
        
        df=pd.concat([h12_y1sum,h12_y0sum,h12_y1mean,h12_ycount])
        if huc_level is None:
            geo_df=self.hucBoundaryMerge(df)
        else:
            df_h=self.hucAggregate(df,huc_level)
            geo_df,huc_level=self.hucBoundaryMerge(df_h,right_on=huc_level)
        self.geo_h12_df=geo_h12_df
        
        fig=plt.figure(dpi=600,figsize=[12,8])
        fig.suptitle('dependent variable')
        for i in range(len(cols)):
            col=cols[i]
            self.map_plot(geo_h12_df,col,subplot_tup=tuplist[i],fig=fig)
        fig.savefig(Helper().getname(os.path.join(self.print_dir,'y01'+'.png')))
        fig.show()
        
    def hucAggregate(self,df,huc_level):
        if type(huc_level) is int:
            if huc_level<10:
                huc_name='HUC0'+str(huc_level)
            else:
                huc_name='HUC'+str(huc_level)
            huc_name='HUC'+str(int)
            huc_digs=str(huc_level)
        elif huc_level[:3].lower()=='huc':
            huc_digs=huc_level[3:]
            huc_name=huc_level
        else:
            huc_digs=huc_level
            huc_name='HUC'+str(huc_level)
        if huc_digs[0]==0:
            huc_digs=huc_digs[1:]
        huc_dig_int=int(huc_digs)
        idx1=df.index
        if not type(idx1) is pd.MultiIndex:
            idx1=pd.MultiIndex.from_tuples([(idx,) for idx in idx1],names=['HUC12']) # make into a tuple, so iterates like a multiindex
            huc_pos=0
        else:
            huc_pos=None
            for pos,name in enumerate(idx1.index.names):
                if re.search('huc',name.lower()):
                    huc_pos=pos
                    break
            assert not huc_pos is None, f'huc_pos is None!, idx1.index.names:{idx1.index.names}'
        
        idx2=[(idx1[i][huc_pos][:huc_dig_int],*idx1[i]) for i in range(len(idx1))] # idx1 is a list of tuples 
        ##'species','estimator','HUC12','COMID', so [-2]  is 'HUC12'
        #idx2=[(idx_add,*idx1[i]) for i in range(len(idx1))] # prepend idx_add
        
        
        expanded_midx=pd.MultiIndex.from_tuples(idx2,names=(huc_name,*idx1.names))
        
        df.index=expanded_midx
        df_mean=df.mean(axis=0,level=huc_name) # mean across huc
        return df_mean,huc_name
    
    def map_plot(self,gdf,col,add_huc_geo=False,fig=None,ax=None,subplot_tup=(1,1,1),title=None):
        if ax is None:
            if fig is None:
                savefig=1
                fig=plt.figure(dpi=300,figsize=[10,8])
            else:
                savefig=0
             
            ax=fig.add_subplot(*subplot_tup)
        if title is None:
            title=col
        ax.set_title(title)
        self.add_huc2_conus(self,ax,huc2_select=None)
        if add_huc_geo:
            gdf=self.hucBoundaryMerge(gdf)
        self.gdf=gdf
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        gdf.plot(column=col,ax=ax,cax=cax,legend=True,)#,legend_kwds={'orientation':'vertical'})
        #collection=self.plot_collection(ax,gdf.geometry,values=gdf[col],colormap='cool')
        
        if savefig:fig.savefig(Helper().getname(os.path.join(self.print_dir,title+'.png')))

    def plot_collection(self,ax, geoms, values=None, colormap='Set1',  facecolor=None, edgecolor=None,
                            alpha=.9, linewidth=1.0, **kwargs):
        patches = []

        for multipoly in geoms:
            for poly in multipoly:

                a = np.asarray(poly.exterior)
                if poly.has_z:
                    poly = shapely.geometry.Polygon(zip(*poly.exterior.xy))

                patches.append(Polygon(a))

        patches = PatchCollection(patches, facecolor=facecolor, linewidth=linewidth, edgecolor=edgecolor, alpha=alpha, **kwargs)

        if values is not None:
            patches.set_array(values)
            patches.set_cmap(colormap)

        ax.add_collection(patches, autolim=True)
        ax.autoscale_view()
        return patches
    
         
    
            
    def plot_top_features(
        self,split=None,top_n=10,rebuild=0,zzzno_fish=False,):
        coef_df,scor_df,y,yhat=self.get_coef_stack(
            rebuild=rebuild,drop_zzz=not zzzno_fish,return_y_yhat=True,
            drop_nocoef_scors=True)
        
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        if presence_filter:
            title=f'top {top_n} presence-hi/absence-lo features'
        else:
            title=f'top {top_n} hi/lo features'
        if scale_by_X:
            title+='_Xscaled'
        if zzzno_fish:
            title+='_zzzno fish'
        if filter_vars:
            title+='_var-filter'
        if spec_wt=='even':
            title+='_even-wt-spec'
        title+='_'+fit_scorer
        
        wtd_coef_dfs=self.pr.build_wt_comid_feature_importance(
            presence_filter=presence_filter,rebuild=rebuild,zzzno_fish=zzzno_fish,
            spec_wt=spec_wt,fit_scorer=fit_scorer,scale_by_X=scale_by_X)
        self.wtd_coef_dfs=wtd_coef_dfs
        if type(wtd_coef_dfs) is pd.DataFrame:
            wtd_coef_dfs=[wtd_coef_dfs]
        
        for i,wtd_coef_df_ in enumerate(wtd_coef_dfs):
            self.logger.info(f'pi_map dropping cols from wtd_coef_df i:{i}')
            #wtd_coef_df_.astype(np.float32,copy=False)
            cols=wtd_coef_df_.columns
            if filter_vars:
                drop_vars=[
                    'tmean','tmax','msst','mwst','precip',
                    'slope','wa','elev','mast','tmin','Al2O3','slp']
                for col in cols:
                    for varstr in drop_vars:
                        if re.search(varstr,col.lower()):
                            try:
                                wtd_coef_df_.drop(col,axis=1,inplace=True)
                                self.logger.info(f'dropped col: {col}')
                            except:
                                self.logger.info(f'failed trying to drop col:{col} matching varstr:{varstr} and current cols:{list(wtd_coef_df_.columns)}')
                            
                            break #stop searching
        if split is None:
            
            self.plot_hilo_coefs(wtd_coef_dfs,top_n,title)
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
        
    def plot_hilo_coefs(self,wtd_coef_dfs,top_n,title):
        cols_sorted_list=[]
        for wtd_coef_df in wtd_coef_dfs:

            big_mean=wtd_coef_df.mean(axis=0)
            cols_sorted_list.append(list(big_mean.sort_values().index))#ascending

        big_top_2n=(cols_sorted_list[0][:top_n],
                    cols_sorted_list[-1][-top_n:]) #-1 could be last item or 1st if len=1
        top_neg_coef_df=wtd_coef_dfs[0].loc[:,big_top_2n[0]]
        top_pos_coef_df=wtd_coef_dfs[-1].loc[:,big_top_2n[1]]
        #self.big_top_wtd_coef_df=big_top_wtd_coef_df
        pos_sort_idx=np.argsort(top_pos_coef_df,axis=1) #ascending
        neg_sort_idx=np.argsort(top_neg_coef_df,axis=1)#.iloc[:,::-1] # ascending
        #select the best and worst columns
        big_top_cols_pos=pos_sort_idx.apply(
            lambda x: big_top_2n[1][x[-1]],axis=1)
        big_top_cols_neg=neg_sort_idx.apply(
            lambda x: big_top_2n[0][x[0]],axis=1)
        colname='top_predictive_variable'
        big_top_cols_pos=big_top_cols_pos.rename(colname)
        big_top_cols_neg=big_top_cols_neg.rename(colname)


        self.big_top_cols_pos=big_top_cols_pos
        self.big_top_cols_neg=big_top_cols_neg

        self.logger.info('starting boundary merge pos')
        geo_pos_cols=self.hucBoundaryMerge(big_top_cols_pos)


        fig=plt.figure(dpi=300,figsize=[10,14])
        fig.suptitle(title)
        ax=fig.add_subplot(2,1,1)
        ax.set_title('top positive features')
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)
        geo_pos_cols.plot(column=colname,ax=ax,cmap='tab20c',legend=True)#,legend_kwds={'orientation':'vertical'})
        self.add_huc2_conus(ax)

        self.logger.info('starting boundary merge neg')
        geo_neg_cols=self.hucBoundaryMerge(big_top_cols_neg)

        ax=fig.add_subplot(2,1,2)
        ax.set_title('top negative features')
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)
        geo_neg_cols.plot(column=colname,ax=ax,cmap='tab20c',legend=True)#,legend_kwds={'orientation':'vertical'})
        self.add_huc2_conus(ax)


        fig.savefig(Helper().getname(os.path.join(
            self.print_dir,f'huc12_top{top_n}_features.png')))
    
        
    def add_huc2_conus(self,ax,huc2_select=None):
        try: huc2=self.boundary_dict['huc02']
        except: 
            self.getHucBoundary('huc02')
            huc2=self.boundary_dict['huc02']
        if huc2_select is None:
            huc2_conus=huc2.loc[huc2.loc[:,'huc2'].astype('int')<19,'geometry']
        else:
            huc2_conus=huc2.loc[huc2.loc[huc2_select,'huc2'].astype('int')<19,'geometry']
        huc2_conus.boundary.plot(linewidth=1,color=None,edgecolor='k',ax=ax)
    
    def plot_confusion_01predict(self,rebuild=0,fit_scorer=None,drop_zzz=True):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        wt_df=self.pr.build_wt_comid_feature_importance(rebuild=rebuild,return_weights=True)
        coef_df,scor_df,y,yhat=self.pr.get_coef_stack(rebuild=rebuild,drop_zzz=drop_zzz,return_y_yhat=True)
        
        
        
        
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
