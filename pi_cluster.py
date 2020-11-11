from sklearn.cluster import AgglomerativeClustering
from pi_results import PiResults
from mylogger import myLogger

class SkCluster(myLogger)

    def clusterSpecsByCoefs(self,clusterer='kmeans',k=None,zzzno_fish=False,cv_collapse=False,scor_wt=True,est=None,fit_scorer=None):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        
        coef_df,scor_df=self.get_coef_stack(rebuild=rebuild,drop_zzz=not zzzno_fish,drop_nocoef_scors=True)
        if not est is None:
            if type(est) is str:
                ests=[est]
            else:
                ests=est
            coef_df,scor_df=self.select_by_index_level_vals([coef_df,scor_df],ests,level_name='estimator')
        scor_select=scor_df.loc[:,('scorer:'+fit_scorer,slice(None),slice(None))]
        
        if cv_collapse:
            coef_df=coef_df.mean(axis=1,level='var')
            scor_select=scor_select.mean(axis=1,level='var')
            if scor_wt:
                denom=scor_select.sum(axis=0,level='species')
                _,denom=scor_select.align(denom,axis=0,join='left')
                scor_select_normed=scor_select.divide(denom,axis=0)
                wtd_coef_df=(coef_df*scor_select_normed).sum(axis=0,level='species')
            else:
                wtd_coef_df=coef_df.mean(axis=0,level='species')
        else:        
            if scor_wt:
                scor_select.columns=scor_select.columns.droplevel('var') #
                denom=scor_select.sum(axis=1).sum(axis=0,level='species')
                _,denom=scor_select.align(denom,axis=0,join='left')
                scor_select_normed=scor_select.divide(denom,axis=0) #broacast over axis1
                wtd_coef_df=(scor_select_normed*coef_df).sum(axis=1,level='var').sum(axis=0,level='species')
            else: assert False,'no cv_collapse requires scor_wt'
            

