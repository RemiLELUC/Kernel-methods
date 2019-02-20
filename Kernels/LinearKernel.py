from . import Kernel
import numpy as np

class LinearKernel(Kernel.Kernel):
    
    def __init__(self):
        Kernel.Kernel.__init__(self, lambda x,y : (x.T)@y)
        
    def compute_similarity_matrix(self, x, y):
        return x@(y.T)