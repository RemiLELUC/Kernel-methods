import itertools
import re
import numpy as np
from . import Kernel
import copy

class KSpectrumKernel(Kernel.Kernel):
    
    def __init__(self,k,alphabet):
        Kernel.Kernel.__init__(self, None)
        self.k = k
        self.alphabet_length = len(alphabet)
        self.k_words = map(lambda x : "".join(x), itertools.product(alphabet,repeat = k))
        
    def compute_occurences(self,x):
        k_words = copy.deepcopy(self.k_words)
        res = np.fromiter(map(lambda word : len(re.findall('(?='+word+')',x)), k_words), dtype = int)
        del k_words
        return res
    
    def compute_similarity_matrix(self, x, y):
        vfun = np.vectorize(self.compute_occurences, signature='()->(n)')
        X = np.apply_along_axis(func1d=vfun, arr=x, axis = 0)
        Y = np.apply_along_axis(func1d=vfun, arr=y, axis = 0)
            
        return X@(Y.T)
        
          
        