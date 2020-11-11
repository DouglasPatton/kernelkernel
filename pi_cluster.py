from sklearn.cluster import AgglomerativeClustering
from pi_results import PiResults
from mylogger import myLogger

class SkCluster(myLogger)
    def __init__(self,):
        
    def clusterSpecsByCoefs(
        self,coef_df,scor_df,clusterer='AgglomerativeClustering',n_clusters=2,zzzno_fish=False,
        cv_collapse='split',scor_wt=True,est=None,fit_scorer=None
        ):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        
        
        if not est is None:
            if type(est) is str:
                ests=[est]
            else:
                ests=est
        scor_select=scor_df.loc[:,('scorer:'+fit_scorer,slice(None),slice(None))]
        
        if cv_collapse is True:
            coef_df=coef_df.mean(axis=1,level='var')
            scor_select=scor_select.mean(axis=1,level='var')
            if scor_wt:
                denom=scor_select.sum(axis=0,level='species')
                _,denom=scor_select.align(denom,axis=0,join='left')
                scor_select_normed=scor_select.divide(denom,axis=0)
                wtd_coef_df=(coef_df*scor_select_normed).sum(axis=0,level='species')
            else:
                wtd_coef_df=coef_df.mean(axis=0,level='species')
        
        else: #i.e. if cv_collapse is Split or False   
            if cv_collapse=='split':
                coef_df=coef_df.mean(axis=1,level=['var','rep_idx'])
                scor_select=scor_select.mean(axis=1,level=['var','rep_idx'])
            if scor_wt:
                scor_select.columns=scor_select.columns.droplevel('var') #
                denom=scor_select.sum(axis=1).sum(axis=0,level='species')
                _,denom=scor_select.align(denom,axis=0,join='left')
                scor_select_normed=scor_select.divide(denom,axis=0) #broacast over axis1
                wtd_coef_df=(scor_select_normed*coef_df).sum(axis=1,level='var').sum(axis=0,level='species')
            else: assert False,'no cv_collapse requires scor_wt'
        if clusterer='AgglomerativeClustering'
            cluster_model=AgglomerativeClustering(n_clusters=n_clusters)
        specs=wtd_coef_df.index
        cluster_model.fit(wtd_coef_df)
        labels=cluster_model.labels_
        spec_clust_dict={l:[] for l in list(set(labels))}
        for i in range(labels):
            label=labels[i]
            spec=specs[i]
            spec_clust_dict[label].append(spec)
        return spec_clust_dict
            
            
            

