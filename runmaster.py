if __name__=="__main__":
    import mycluster
    '''
    Ndiff_type_variations=('modeldict:Ndiff_type',['recursive','product'])
    max_bw_Ndiff_variations=('modeldict:max_bw_Ndiff',[2,3])
    Ndiff_start_variations=('modeldict:Ndiff_start',[1,2])
    product_kern_norm_variations=('modeldict:product_kern_norm',['self','own_n'])
    normalize_Ndiffwtsum_variations=('modeldict:normalize_Ndiffwtsum',['own_n','across'])
    optdict_variation_list=[Ndiff_type_variations,max_bw_Ndiff_variations,Ndiff_start_variations,product_kern_norm_variations,normalize_Ndiffwtsum_variations]
    
    train_n_variations=('train_n',[30,45,60])
    ykern_grid_variations=('ykern_grid',[31,46,61])
    ftype_variations=('ftype',['linear','quadratic'])
    param_count_variations=('param_count',[1,2])
    datagen_variation_list=[train_n_variations,ftype_variations,param_count_variations]
    '''
    
    
    Ndiff_type_variations=('modeldict:Ndiff_type',['recursive','product'])
    optdict_variation_list=[Ndiff_type_variations]
    
    
    train_n_variations=('train_n',[7,12])
    datagen_variation_list=[train_n_variations]
    #'''
    
    mycluster.run_cluster(mytype='master',optdict_variation_list=optdict_variation_list,datagen_variation_list=datagen_variation_list,local_test='no')
    
    

