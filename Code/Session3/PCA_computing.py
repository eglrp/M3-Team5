from sklearn.decomposition import PCA

def PCA_to_data(D, Train_descriptors, num_components):
    
    pca = PCA(n_components = num_components)
    pca.components_
    pca.fit(D)
    
    new_D = pca.transform(D)
    
    for idx,TrainDes in enumerate(Train_descriptors):        
            train_descriptor = pca.transform(TrainDes)
            Train_descriptors[idx]=train_descriptor

    return new_D, Train_descriptors, pca