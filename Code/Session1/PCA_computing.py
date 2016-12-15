from sklearn.decomposition import PCA

def PCA_to_data(D, num_components):
    
    pca = PCA(n_components = num_components)
    pca.fit(D)
    new_D = pca.transform(D)
    
    return new_D, pca