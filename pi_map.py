import os
import geopandas as gpd
from pi_results import PiResults
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helpers import Helper

class Mapper:
    def __init__(self):
        cwd=os.getcwd()
        self.geo_data_dir=os.path.join(cwd,'geo_data')
        self.print_dir=os.path.join(cwd,'print')
        self.boundary_data_path=self.boundaryDataCheck()
        self.pr=PiResults()
        
    def boundaryDataCheck(self):
        #https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/WBD%20v2.3%20Model%20Poster%2006012020.pdf
        datalink="https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip"
        boundary_data_path=os.path.join(self.geo_data_dir,'WBD_National_GDB.gdb')
        if not os.path.exists(boundary_data_path):
            assert False, f"cannot locate boundary data. download from {datalink}"
        return boundary_data_path
    
    def getHuc12Boundary(self):
        self.h12_boundary=gpd.read_file(self.boundary_data_path,layer='WBDHU12')
    
    def draw_huc12_truefalse(self,rebuild=0):
        try: self.h12_boundary
        except: self.getHuc12Boundary()
        agg_prediction_spec_df=self.pr.build_aggregate_predictions_by_species(rebuild=rebuild)
        yhat=agg_prediction_spec_df.drop('y_train',level='estimator').loc[:,agg_prediction_spec_df.columns!='y']
        y=agg_prediction_spec_df.xs('y_train',level='estimator').loc[:,['y']]
        y_a,yhat_a= y.align(yhat,axis=0)
        
        
        diff=-1*yhat_a.sub(y_a['y'],axis=0)
        nofish_diff=diff.xs('zzzno fish',level='species')
        diff.drop('zzzno fish',level='species')
        
        abs_diff=diff.abs()
        nofish_abs_diff=nofish_diff.abs()
        
        err_geo_df=self.mean_flatten_and_align_huc12(diff)
        nofish_err_geo_df=self.mean_flatten_and_align_huc12(nofish_diff)
        abs_diff_geo_df=self.mean_flatten_and_align_huc12(abs_diff)
        nofish_abs_diff_geo_df=self.mean_flatten_and_align_huc12(nofish_abs_diff)
        
        """mean_prediction_err=diff.mean(axis=1).mean(level='HUC12')
        nofish_mpe=nofish_diff.mean(axis=1).mean(level='HUC12')
        mean_prediction_abs_err=diff.abs().mean(axis=1).mean(level='HUC12')
        nofish_mpae=nofish_diff.abs().mean(axis=1).mean(level='HUC12')
        err_df=mean_self.mean_flatten_and_align_huc12(prediction_err.rename('err').reset_index().rename(columns={'HUC12':'huc12'})# reset_index moves index to columns for merge, rename b/c no name after the mean.
        nofish_err_df=nofish_mpe.rename('err').reset_index().rename(columns={'HUC12':'huc12'})
        err_geo_df=self.h12_boundary.merge(err_df,on='huc12')
        nofish_err_geo_df=self.h12_boundary.merge(nofish_err_df,on='huc12')
        
        mean_predicted_correct=1-mean_prediction_abs_err
        nofish_mean_predicted_correct=mean_predicted_correct.xs('zzzno fish',level='species')
        mean_predicted_correct.drop('zzzno fish',level='species')"""
        
        rvrs_abs_diff_geo_df=1-abs_diff_geo_df
        rvrs_nofish_abs_diff_geo_df=1-nofish_abs_diff_geo_df
        fig=plt.figure(dpi=300,figsize=[10,8])
        ax=fig.add_subplot(1,1,2)


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        err_geo_df.plot(column='err',ax=ax,cax=cax,legend=True,cmap='brg',alpha=rvrs_abs_diff_geo_df)#,legend_kwds={'orientation':'vertical'})
        ax2=fig.add_subplot(2,1,2)
        nofish_err_geo_df.plot(column='err',ax=ax2,cax=cax,legend=True,cmap='brg',alpha=rvrs_nofish_abs_diff_geo_df)
        
        
        fig.savefig(Helper().getname(os.path.join(self.print_dir,'error_map.png')))

            
    
    def mean_flatten_and_align_huc12(self,err_df):
        mean_err=err_df.mean(axis=1).mean(level='HUC12')
        err_df=mean_err.rename('err').reset_index().rename(columns={'HUC12':'huc12'})# reset_index moves index to columns for merge, rename b/c no name after the mean.
        geo_err_df=self.h12_boundary.merge(err_df,on='huc12')
        return geo_err_df
