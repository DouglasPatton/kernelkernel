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
        plt.rc_context({
            
            'figure.autolayout': True,
            'axes.edgecolor':'k', 
            'xtick.color':'k', 'ytick.color':'k', 
            'figure.facecolor':'grey'})
      
    def boundaryDataCheck(self):
        #https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/WBD%20v2.3%20Model%20Poster%2006012020.pdf
        datalink="https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip"
        boundary_data_path=os.path.join(self.geo_data_dir,'WBD_National_GDB.gdb')
        if not os.path.exists(boundary_data_path):
            assert False, f"cannot locate boundary data. download from {datalink}"
        return boundary_data_path
    
    def getHucBoundary(self,huc_level):
        print(huc_level)
        level_digits=huc_level[-2:]
        if level_digits[0]=='0' or not level_digits[0].isdigit():
            level_digits=level_digits[1] #e.g. HUC02 --> 2
        return gpd.read_file(self.boundary_data_path,layer=f'WBDHU{level_digits}')
        
    def hucBoundaryMerge(self,data_df,right_on='HUC12'):
        # hucboundary files have index levels like 'huc2' not 'HUC02'
        #if right_on.lower()=='huc12':
        self.logger.info(f'starting boundary merge')
        self.data_df=data_df
        if type(right_on) is str:
            huc_level=right_on.lower()
        elif type(right_on) is int:
            huc_level=f'huc{right_on}' #'huc' not necessry
        else:
            assert False, f'unexepected right_on:{right_on} not str or int'
        level_digits=huc_level[-2:]
        if huc_level[-2]=='0':
            huc_level=huc_level[:-2]+huc_level[-1]
        if type(data_df.index) is pd.MultiIndex:
            h_list=data_df.index.unique(level=right_on)
        else:
            h_list=list(data_df.index)
        boundary=self.getHucBoundary(huc_level)
        #print('huc_level',huc_level)
        #print('boundary.index.names',boundary.index.names.index)
        #boundary_huc_pos=boundary.columns.names.index(huc_level)
        #selector=[slice(None) for _ in range(len())]
        #selector[boundary_huc_pos]=h12_list
        boundary_clip=boundary.loc[boundary[huc_level].isin(h_list)]
        #boundary_clip=boundary.loc[selector]
        merged=boundary_clip.merge(data_df,left_on=huc_level,right_on=right_on)
        self.logger.info(f'boundary merge completed')
        return merged
    
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
            df_h,huc_level=self.hucAggregate(df,huc_level,collapse='mean')
            geo_df=self.hucBoundaryMerge(df_h,right_on=huc_level)
        self.geo_df=geo_df
        
        fig=plt.figure(dpi=600,figsize=[12,8])
        fig.suptitle('dependent variable')
        for i in range(len(cols)):
            col=cols[i]
            self.map_plot(geo_df,col,subplot_tup=tuplist[i],fig=fig)
        if huc_level:
            name=f'y01_{huc_level}.png'
        else:
            name='y01'+'.png'
        
        fig.savefig(Helper().getname(os.path.join(self.print_dir,name)))
        fig.show() 
        
    def plot_confusion_01predict(self,rebuild=0,fit_scorer=None,drop_zzz=True,wt=None,huc_level=None,normal_color=True):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        coef_df,scor_df,y,yhat=self.pr.get_coef_stack(rebuild=rebuild,drop_zzz=drop_zzz,return_y_yhat=True)
        if not wt is None:
            wt_df=self.pr.build_wt_comid_feature_importance(rebuild=rebuild,return_weights=True)
            assert False,'not developed'

        yhat_a,y_a=yhat.align(y,axis=0)
        y_vals=y_a.values
        diff=yhat_a.subtract(y_vals,axis=0).astype(np.float16)*-1 # y-yhat


        pos=diff[y_vals==1]
        neg=diff[y_vals==0].abs()
        
        tp=(1-pos).mean(axis=1)
        tn=(1-neg).mean(axis=1)
        fp=neg.mean(axis=1) #dif is neg for an fp, so reverse it to signal fp
        fn=pos.mean(axis=1)
        
        
        
        confu_list=[tp,fp,fn,tn]
        confu_varnames=['true positive','false positive','false negative','true negative']
        for i in range(len(confu_list)):confu_list[i].rename(confu_varnames[i],inplace=True)
        
        self.confu_list=confu_list
        #confu_list=[self.swap_index_by_level(confu_list[i],'var',confu_varnames[i],axis=1) for i in range(4)]                 
        
        if not wt is None:
            assert False,'not developed'
        if huc_level is None:
            huc_level='HUC12' 
            confu_list=[ser.mean(axis=0,level=huc_level) for ser in confu_list] #ser for series
        else:
            #confu_df,huc_level=self.hucAggregate(confu_df,huc_level,collapse='mean')    
            confu_list,huc_levels=zip(*[self.hucAggregate(df,huc_level,collapse='mean') for df in confu_list])  
            huc_level=huc_levels[0]
        """   
        self.logger.info(f'starting confu_list join')
        confu_df=confu_list[0].join(confu_list[1:])
        self.logger.info(f'completed join')
        """
        
        geo_dfs=[self.hucBoundaryMerge(df,right_on=huc_level) for df in confu_list]
        self.geo_dfs=geo_dfs
        df_col_vars=confu_varnames #
        c=2 #  of columns
        r=2
        tuplist=[(r,c,i+1) for i in range(len(df_col_vars))]
        
        fig=plt.figure(dpi=600,figsize=[12,8])
        fig.suptitle(f'all species normalized confusion matrix by {huc_level}')
        for i in range(len(df_col_vars)):
            col=df_col_vars[i]
            geo_df=geo_dfs[i]
            self.logger.info(f'adding plot {i+1} of {len(df_col_vars)}')
            if normal_color:
                plotkwargs={'vmin':0,'vmax':1}
            else:
                plotkwargs={}
            self.map_plot(geo_df,col,subplot_tup=tuplist[i],fig=fig,plotkwargs=plotkwargs)

        name=f'confusion_matrix_map_{huc_level}.png'
        
        fig.savefig(Helper().getname(os.path.join(self.print_dir,name)))
        fig.show() 
                     
        
                     
    def swap_index_by_level(self,df,level,new,axis=0):
        if axis==0:
            idx=df.index
        elif axis==1:
            idx=df.columns
        else:
            assert False,'no other option'
        level_names=idx.names
        level_pos=level_names.index(level)
        assert type(idx) is pd.MultiIndex, f'expecting df to have multiindex index. but type(idx):{type(idx)}'
        new_idx=[]
        for tup in idx:
            not_tup=list(tup)
            not_tup[level_pos]=new
            new_idx.append(tuple(not_tup))
        new_midx=pd.MultiIndex.from_tuples(new_idx,names=level_names)
        if axis==0:
            df.index=new_midx
        elif axis==1:
            df.columns=new_midx
        else: 
            assert False,'no other option'
        return df    
        
        
    
        
        
        
    def hucAggregate(self,df,huc_level,collapse='mean'):
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
            for pos,name in enumerate(idx1.names):
                if re.search('huc',name.lower()):
                    huc_pos=pos
                    break
            assert not huc_pos is None, f'huc_pos is None!, idx1.index.names:{idx1.index.names}'
        
        idx2=[(idx1[i][huc_pos][:huc_dig_int],*idx1[i]) for i in range(len(idx1))] # idx1 is a list of tuples 
        ##'species','estimator','HUC12','COMID', so [-2]  is 'HUC12'
        #idx2=[(idx_add,*idx1[i]) for i in range(len(idx1))] # prepend idx_add
        
        
        expanded_midx=pd.MultiIndex.from_tuples(idx2,names=(huc_name,*idx1.names))
        
        df.index=expanded_midx
        if not collapse is None:
            if collapse=='mean':
                return df.mean(axis=0,level=huc_name),huc_name # mean across huc
            else: assert False, 'not developed'
        return df,huc_name
    
    def map_plot(self,gdf,col,add_huc_geo=None ,fig=None,ax=None,subplot_tup=(1,1,1),title=None,plotkwargs={}):
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
        self.add_huc2_conus(ax,huc2_select=None)
        if not add_huc_geo is None:
            assert type(add_huc_geo) is str,f'expecting string like HUC8, but got: {add_huc_geo}'
            gdf=self.hucBoundaryMerge(gdf,right_on=add_huc_geo)
        self.gdf=gdf
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        gdf.plot(column=col,ax=ax,cax=cax,legend=True,**plotkwargs)#,legend_kwds={'orientation':'vertical'})
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
    
         
    
    def plot_top_features(self,huc_level=None,split=None,top_n=10,rebuild=0,zzzno_fish=False,
        filter_vars=False,spec_wt=None,fit_scorer=None,scale_by_X=False,presence_filter=False):
        
        """
        coef_df,scor_df,y,yhat=self.pr.get_coef_stack(
            rebuild=rebuild,drop_zzz=not zzzno_fish,return_y_yhat=True,
            drop_nocoef_scors=True)
        """
        
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
        if huc_level:
            title+=f'_{huc_level}'
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
            
            self.plot_hilo_coefs(wtd_coef_dfs,top_n,title,huc_level=huc_level)
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
        
    def plot_hilo_coefs(self,wtd_coef_dfs,top_n,title,huc_level=None):
        cols_sorted_list=[]
        selector=[0,-1]
        select_cols_list=[]
        lo_hi_names=['lowest_coef','highest_coef']
        for df_idx in selector:
            wtd_coef_df=wtd_coef_dfs[df_idx]
            if not huc_level is None:
                wtd_coef_df,_=self.hucAggregate(wtd_coef_df,huc_level)
        
            
            #the variable names    
            cols=wtd_coef_df.columns
            #sort acros cols, i.e., each row
            coef_sort_idx=np.argsort(wtd_coef_df,axis=1) 
            
            select_cols=coef_sort_idx.apply(
                lambda x:cols[x.iloc[df_idx]],axis=1,) #df_idx will choose the first or last from each sorted list
            
            n_select_cols=select_cols.value_counts(ascending=False).index.tolist()[:top_n]
            select_cols_list.append(n_select_cols)
            
            
            '''below code won't work for separate wtd_dfs
            # for each row, take best and worst in dataframe with col for lo, col for hi.
            lo_hi_cols=coef_sort_idx.apply(
                lambda x:pd.Series([cols[x.iloc[0]],cols[x.iloc[-1]]],index=lo_hi_names,dtype='category'),axis=1,)
            #count frequency of each list of worst then best. creating a series for worst and best since ordering is different
            #lo_val_counts=[lo_hi_cols.loc[:,n].value_counts(ascending=False) for n in lo_hi_names]
            #lo_hi_val_counts=lo_hi_cols.apply(lambda col:col.value_counts(ascending=True),axis=0)
            
            #get the most frequent big and small            
            big_vote_lo_hi=[lo_val_counts[i].iloc[slice(None,top_n).index] for i in range(2)]            
            self.big_vote_lo_hi=big_vote_lo_hi'''
    
        """ 
            ###################
            assert False, 'developing new ranking approach!'
            ####################
            big_mean=wtd_coef_df.mean(axis=0)#mean aross all hucs
            #sort_ord=np.argsort(big_mean)
            #cols_sorted_list.append(wtd_coef_df.columns.to_list()[sort_ord])
            cols_sorted_list.append(list(big_mean.sort_values().index))#ascending

        big_top_2n=(cols_sorted_list[0][:top_n],
                    cols_sorted_list[-1][-top_n:]) #-1 could be last item or 1st if len=1
        """
        big_top_2n=tuple(select_cols_list)
        top_neg_coef_df=wtd_coef_dfs[0].loc[:,big_top_2n[0]]
        top_pos_coef_df=wtd_coef_dfs[-1].loc[:,big_top_2n[1]]
        #self.big_top_wtd_coef_df=big_top_wtd_coef_df
        pos_sort_idx=np.argsort(top_pos_coef_df,axis=1) #ascending
        neg_sort_idx=np.argsort(top_neg_coef_df,axis=1)#.iloc[:,::-1] # ascending
        #select the best and worst columns
        big_top_cols_pos=pos_sort_idx.apply(
            lambda x: big_top_2n[1][x.iloc[-1]],axis=1)
        big_top_cols_neg=neg_sort_idx.apply(
            lambda x: big_top_2n[0][x.iloc[0]],axis=1)
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
            huc2=self.getHucBoundary('huc02')
        if huc2_select is None:
            huc2_conus=huc2.loc[huc2.loc[:,'huc2'].astype('int')<19,'geometry']
        else:
            huc2_conus=huc2.loc[huc2.loc[huc2_select,'huc2'].astype('int')<19,'geometry']
        huc2_conus.boundary.plot(linewidth=1,color=None,edgecolor='k',ax=ax)
   
        
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
