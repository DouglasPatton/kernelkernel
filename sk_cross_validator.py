from mylogger import myLogger
class regressor_q_stratified_cv:
    def __init__(self,n_splits=10,n_repeats=2,group_count=10,random_state=0):
        self.group_count=group_count
        cvkwargs=dict(n_splits=n_splits,n_repeats=n_repeats,random_state=random_state)
        self.cv=RepeatedStratifiedKFold(**cvkwargs)
            
    def split(self,X,y,groups=None):
        split1=np.array_split(np.ones(y.shape),self.group_count)
        groupsplit=[i*split1[i] for i in range(self.group_count)]
        y_srt_order=np.argsort(y)
        qgroups=np.empty_like(y)
        qgroups[y_srt_order]=np.concatenate(groupsplit,axis=0)
        return self.cv.split(X,qgroups,groups)
    
    def get_n_splits(self,X,y,groups=None):
        return self.cv.get_n_splits(X,y,groups)
                
        
if __name__=="__main__":
    n_splits=5
    n_repeats=5
    group_count=5
    cv=regressor_q_stratified_cv(n_splits=n_splits,n_repeats=n_repeats,group_count=group_count,random_state=0)
    import numpy as np
    n=1000000
    y=np.linspace(-n//2,n//2,n+1)
    n=y.size
    np.random.shuffle(y)
    X=y.copy()[:,None] # make 2d
    
    i=0;j=0;splist=[];test_idx_list=[]
    for train,test in cv.split(X,y):
        if i==0:print(f'cv results for *test* set {j} ')
        #print(train,test)
        range_i=np.ptp(y[train])
        splist.append(range_i)
        test_idx_list.append(train)
        print(f'range for rep:{j}, fold:{i}, {range_i}')
        i+=1
        if i==n_splits:
            test_unique_count=np.size(np.unique(np.concatenate(test_idx_list)))
            print(f'range of ranges, {np.ptp(np.array(splist))}')
            print(f'unique elements:{test_unique_count} for n:{n}','\n')
            splist=[];test_idx_list=[]
            i=0;j+=1
                    
    """
        
    def old(self,):
        # test data already removed
        n,k=X_train.shape
        onecount=int(ydataarray.sum())
        sum_count_arr=np.ones([n,])
        if cat_reg=='cat':
            cats=np.unique(y_train)
            cat_count_dict={}
            for cat in cats:
                cat_count_dict[cat]=np.sum(sum_count_arr[y_train==cat])
        elif cat_reg=='reg':
            
            cat_edges=np.quantile(y_train,np.linspace(0,1,6))
            q=np.quantile(y_train)    
            #bins,_=np.histogram(y_train,bins=q)
        zerocount=n-onecount
        countlist=[zerocount,onecount]
        if onecount<zerocount:
            smaller=1
        else:
            smaller=0
        
        if not min_y is None:
            if min_y<1:
                min_y=int(batch_n*min_y)
            batch01_n=[None,None]
            batch01_n[smaller]=min_y # this makes it max_y too....
            batch01_n[1-smaller]=batch_n-min_y
            max_batchcount=countlist[smaller]//min_y
            if max_batchcount<batchcount:
                #self.logger.info(f'for {self.species} batchcount changed from {batchcount} to {max_batchcount}')
                batchcount=max_batchcount
            subsample_n=batchcount*batch_n
            oneidx=np.arange(n)[ydataarray==1]
            zeroidx=np.arange(n)[ydataarray==0]
            bb_select_list=[]
            for bb_idx in range(cv_reps):
                ones=np.random.choice(oneidx,size=batch01_n[1]*batchcount,replace=False)
                zeros=np.random.choice(zeroidx,size=batch01_n[0]*batchcount,replace=False)
                bb_select_list.append(np.concatenate([ones,zeros],axis=0))
        else:
            max_batchcount=n//batch_n
            if max_batchcount<batchcount:
                #self.logger.info(f'for {self.species} batchcount changed from {batchcount} to {max_batchcount}')
                batchcount=max_batchcount
            subsample_n=batchcount*batch_n
            for bb_idx in range(cv_reps):
                bb_select_list.append(np.random.choice(np.arange(n),size=subsample_n),replace=False)
        batchbatchlist=[[None for __ in range(batchcount)] for _ in range(cv_reps)]
        SKF=StratifiedKFold(n_splits=batchcount, shuffle=False)
        for bb_idx in range(cv_reps):
            bb_x_subsample=xdataarray[bb_select_list[bb_idx],:]
            bb_y_subsample=ydataarray[bb_select_list[bb_idx]]
            for j,(train_index,test_index) in enumerate(SKF.split(bb_x_subsample,bb_y_subsample)):
                batchbatchlist[bb_idx][j]=(bb_y_subsample[test_index],bb_x_subsample[test_index,:])
        batchsize=batch_n*batchcount
        
        
        self.batchcount=batchcount
        self.expand_datagen_dict('batchcount',self.batchcount)
        fullbatchbatch_n=cv_reps*batchsize
        self.fullbatchbatch_n=fullbatchbatch_n
        self.expand_datagen_dict('fullbatchbatch_n',self.fullbatchbatch_n)
        self.logger.info(f'yxtup shapes:{[(yxtup[0].shape,yxtup[1].shape) for yxtuplist in batchbatchlist for yxtup in yxtuplist]}')
        self.yxtup_batchbatch=batchbatchlist
        
        """