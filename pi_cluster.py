from sklearn.cluster import AgglomerativeClustering,KMeans,DBSCAN
from sklearn.preprocessing import StandardScaler
from pi_results import PiResults
from mylogger import myLogger

class SkCluster(myLogger):
    def __init__(self,fit_scorer='f1_micro'):
        func_name=f'SkCluster'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        self.fit_scorer=fit_scorer
        
    def select_by_index_level_vals(self,df_list,level_vals,level_name='species'):
        new_list=[]
        if not type(df_list) is list:
            returndf=True
            df=[df]
        if not type(level_vals) is list:
            level_vals=[level_vals]
        else:
            returndf=False
        for df in df_list:
            pos=df.index.names.index(level_name)
            selector=[slice(None) for _ in range(len(df.index.names))]
            selector[pos]=level_vals
            selector=tuple(selector)
            new_list.append(df.loc[selector,:])
        if returndf:
            return new_list[0]
        else:
            return new_list 
        
    def clusterSpecsByCoefs(
        self,coef_df,scor_df,clusterer='AgglomerativeClustering',n_clusters=2,zzzno_fish=False,
        cv_collapse='split',scor_wt=True,est=None,fit_scorer=None,row_norm=True
        ):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        
        
        if not est is None:
            if type(est) is str:
                ests=[est]
            else:
                ests=est
            coef_df,scor_df=self.select_by_index_level_vals([coef_df,scor_df],ests,level_name='estimator')
            
        scor_select=scor_df.loc[:,('scorer:'+fit_scorer,slice(None),slice(None))]
        self.coef_df=coef_df
        if row_norm:
            sum_by_levs=[name for name in coef_df.columns.names if not name=='var']
            denom=coef_df.sum(axis=1,level=sum_by_levs)
            _,denom=coef_df.align(denom,axis=1,join='left')
            coef_df=coef_df.divide(denom,axis=0)
            #print('coef_df',coef_df)
        
        if cv_collapse is True:
            coef_df=coef_df.mean(axis=1,level='var')
            
            if scor_wt:
                scor_select=scor_select.mean(axis=1,level='var')
                denom=scor_select.sum(axis=0,level='species')
                _,denom=scor_select.align(denom,axis=0,join='left')
                scor_select_normed=scor_select.divide(denom,axis=0)
                """self.logger.info(f'before align: scor_select_normed.columns:{scor_select_normed.columns}')
                _,scor_select_normed=coef_df.align(scor_select_normed,axis=1,join='left')
                self.logger.info(f'AFTER align: scor_select_normed.columns:{scor_select_normed.columns}')"""
                wtd_coef_df=coef_df.mul(scor_select_normed.values,axis=0).sum(axis=0,level='species')
                #print('wtd_coef_df',wtd_coef_df)
            else:
                wtd_coef_df=coef_df.mean(axis=0,level='species')
        
        else: #i.e. if cv_collapse is Split or False   
            if cv_collapse=='split':
                coef_df=coef_df.mean(axis=1,level=['var','rep_idx'])
                scor_select=scor_select.mean(axis=1,level=['var','rep_idx'])
            if scor_wt:
                scor_select.columns=scor_select.columns.droplevel('var') #
                denom=scor_select.sum(axis=1).sum(axis=0,level='species')
                self.scor_select=scor_select
                self.denom=denom
                #assert False,'debug'
                #print('denom1',denom)
                denom,_=denom.align(scor_select,axis=0,)
                #print('denom2',denom)
                scor_select_normed=scor_select.divide(denom,axis=0) 
                #print('scor_select_normed1',scor_select_normed)
                _,scor_select_normed=coef_df.align(scor_select_normed,axis=1,join='left')
                #print('scor_select_normed2',scor_select_normed)
                product=coef_df.mul(scor_select_normed,axis=0)
                wtd_coef_df=(product).sum(axis=1,level='var').mean(axis=0,level='species')#.sum(axis=0..?
            else: assert False,'no cv_collapse requires scor_wt'
                
        
        if clusterer.lower()=='AgglomerativeClustering'.lower():
            cluster_model=AgglomerativeClustering(n_clusters=n_clusters)
        elif clusterer.lower()=='KMeans'.lower():
            cluster_model=KMeans(n_clusters=n_clusters,random_state=0)
        elif clusterer.lower()=='DBSCAN'.lower():
            cluster_model=DBSCAN(eps=10,metric = 'minkowski', p = 1,min_samples=5)
        self.wtd_coef_df=wtd_coef_df    
        wtd_coef_df.iloc[:]=StandardScaler().fit_transform(wtd_coef_df)
        
        specs=wtd_coef_df.index
        cluster_model.fit(wtd_coef_df)
        labels=cluster_model.labels_
        spec_clust_dict={l:[] for l in list(set(labels))}
        for i in range(len(labels)):
            label=labels[i]
            spec=specs[i]
            spec_clust_dict[label].append(spec)
        return spec_clust_dict
            
            
            

