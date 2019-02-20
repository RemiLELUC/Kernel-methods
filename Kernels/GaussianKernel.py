from . import Kernel
import numpy as np

class GaussianKernel(Kernel.Kernel):
    
    def __init__(self, bandwidth):
        Kernel.Kernel.__init__(self,lambda x,y : np.exp(-(np.linalg.norm(x-y)/bandwidth)**2))
        self.bandwidth = bandwidth
        
    def compute_similarity_matrix(self,x,y):
        n = x.shape[0]
        p = y.shape[0]
        
        X_dup = np.einsum('pnm -> npm', np.tile(x[np.newaxis,:,:], reps=(p,1,1)))
        Y_dup = np.tile(y[np.newaxis,:,:], reps=(n,1,1))
        
        res = np.exp(-(np.linalg.norm(X_dup-Y_dup, axis=2)/self.bandwidth)**2)
        
        del X_dup
        del Y_dup
        
        return res
        