import mycluster
if __name__=="__main__":

    '''

    
    Ndiff_type_variations=('modeldict:Ndiff_type',['recursive','product'])
    max_bw_Ndiff_variations=('modeldict:max_bw_Ndiff',[2])
    Ndiff_start_variations=('modeldict:Ndiff_start',[1,2])
    ykern_grid_variations=('modeldict:ykern_grid',[33])
    product_kern_norm_variations=('modeldict:product_kern_norm',['self','own_n'])
    normalize_Ndiffwtsum_variations=('modeldict:normalize_Ndiffwtsum',['own_n','across'])
    optdict_variation_list=[Ndiff_type_variations,max_bw_Ndiff_variations,Ndiff_start_variations,ykern_grid_variations,product_kern_norm_variations,normalize_Ndiffwtsum_variations]
    
    
    batch_n_variations=('batch_n',[32])
    batchcount_variations=('batchcount',[40])
    ftype_variations=('ftype',['linear','quadratic'])
    param_count_variations=('param_count',[1,2])
    datagen_variation_list=[batch_n_variations,batchcount_variations,ftype_variations,param_count_variations]
    
    mycluster.run_cluster(myname='master',optdict_variation_list=optdict_variation_list,datagen_variation_list=datagen_variation_list)

    '''
    
    mycluster.run_cluster(myname='master',optdict_variation_list=None,datagen_variation_list=None,local_test='no')
    
    

