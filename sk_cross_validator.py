class CV_cat_reg_supreme
    def (self,X_train,y_train,folds=10,reps=2,catreg='cat',strategy='balanced'):
        #needs updating!
        
        # test data already removed
        batchbatchcount=self.batchbatchcount
        n=ydataarray.shape[0]; p=xdataarray.shape[1]
        onecount=int(ydataarray.sum())
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
            for bb_idx in range(batchbatchcount):
                ones=np.random.choice(oneidx,size=batch01_n[1]*batchcount,replace=False)
                zeros=np.random.choice(zeroidx,size=batch01_n[0]*batchcount,replace=False)
                bb_select_list.append(np.concatenate([ones,zeros],axis=0))
        else:
            max_batchcount=n//batch_n
            if max_batchcount<batchcount:
                #self.logger.info(f'for {self.species} batchcount changed from {batchcount} to {max_batchcount}')
                batchcount=max_batchcount
            subsample_n=batchcount*batch_n
            for bb_idx in range(batchbatchcount):
                bb_select_list.append(np.random.choice(np.arange(n),size=subsample_n),replace=False)
        batchbatchlist=[[None for __ in range(batchcount)] for _ in range(batchbatchcount)]
        SKF=StratifiedKFold(n_splits=batchcount, shuffle=False)
        for bb_idx in range(batchbatchcount):
            bb_x_subsample=xdataarray[bb_select_list[bb_idx],:]
            bb_y_subsample=ydataarray[bb_select_list[bb_idx]]
            for j,(train_index,test_index) in enumerate(SKF.split(bb_x_subsample,bb_y_subsample)):
                batchbatchlist[bb_idx][j]=(bb_y_subsample[test_index],bb_x_subsample[test_index,:])
        batchsize=batch_n*batchcount
        
        
        self.batchcount=batchcount
        self.expand_datagen_dict('batchcount',self.batchcount)
        fullbatchbatch_n=batchbatchcount*batchsize
        self.fullbatchbatch_n=fullbatchbatch_n
        self.expand_datagen_dict('fullbatchbatch_n',self.fullbatchbatch_n)
        self.logger.info(f'yxtup shapes:{[(yxtup[0].shape,yxtup[1].shape) for yxtuplist in batchbatchlist for yxtup in yxtuplist]}')
        self.yxtup_batchbatch=batchbatchlist
        
        