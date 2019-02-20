import numpy as np

class PCA:
    
    def __init__(self):
        pass
        
    def perform_PCA(self, inputs, kernel, dim_num):
        #compute centered graham matrix
        graham_matrix = kernel.matrix_from_data(inputs)
        n = graham_matrix.shape[0]
        I_U = np.identity(n) - np.ones((n,n))/n
        centered_graham_matrix = (I_U)@graham_matrix@(I_U)
        
        del graham_matrix
        
        #compute the dim_num first eigen vectors of centered_graham _matrix
        eigen_values, eigen_vectors = np.linalg.eigh(centered_graham_matrix)

        idx = np.argsort(eigen_values)[::-1][:dim_num]
        s_eigen_values = eigen_values[idx]
        s_eigen_vectors = eigen_vectors[idx]

        projections = centered_graham_matrix@(np.divide(s_eigen_vectors,np.sqrt(s_eigen_values[:,None]))).T
            
        del centered_graham_matrix
        
        
        return projections
        
       
        
        
        