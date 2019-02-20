from . import Kernel
import numpy as np
import re
from tqdm import tqdm

def psi_letter(letter, word, lam):
    iter_for_letters = np.where(np.array(list(word)) == letter)[0]
    if iter_for_letters.shape[0] == 0:
        return 0
    else:
        return np.sum(np.power(lam, len(word) - iter_for_letters + 1))
        
                           
class LambdaSubstringKernel(Kernel.Kernel):
    
    def __init__(self, k, lam, alphabet):
        Kernel.Kernel.__init__(self,None)
        self.k = k
        self.lam = lam
        self.alphabet = alphabet
        self.dico_K = [dict() for i in range(k+1)]
        self.dico_B = [dict() for i in range(k+1)]
        
    def K_n(self, n, x, y):
        if (x,y) in self.dico_K[n]:
            return self.dico_K[n][x,y]
        else:
            res = 0
            if (np.minimum(len(x), len(y)) < n) or n == 0:
                pass
            else:
                u,a = x[:-1], x[-1]
                a_occs = np.where(np.array(list(y)) == a)[0]
                a_occs = a_occs[a_occs > 0]
                sum_B_n_1 = 0
                if a_occs.shape[0] != 0: 
                    sum_B_n_1 = np.sum([self.B_n(n-1,x , y[0:a_occ - 1]) for a_occ in a_occs ])
                res = self.lam*self.K_n(n,u,y) + (self.lam**2)*sum_B_n_1
                self.dico_K[n][x,y] = res
            return res
            
    def B_n(self, n, x, y):
        if (x,y) in self.dico_B[n]:
            return self.dico_B[n][x,y]
        else:
            res = 0
    
            if (np.minimum(len(x), len(y)) < n) or n == 0:
                pass
            
            elif n == 1:

                psi_x = np.array([psi_letter(letter, x, self.lam) for letter in self.alphabet])
                psi_y = np.array([psi_letter(letter, y, self.lam) for letter in self.alphabet])
                
                res = psi_y@psi_x
                  
            else:
                u,a = x[:-1],x[-1]
                v,b = y[:-1],y[-1]
                res = self.lam*(self.B_n(n,u,y) + self.B_n(n,x,v)) - (self.lam**2)*self.B_n(n,u,v)
                if a == b:
                    res+=(self.lam**2)*self.B_n(n-1, u, v)
                    
            self.dico_B[n][x,y] = res
            return res
        
        
    def compute_similarity_matrix(self,x,y):
        K = np.zeros((x.shape[0], y.shape[0]))
        for i in tqdm(range(x.shape[0]), position = 1):
            for j in tqdm(range(i,y.shape[0]), position = 0):
                     K[i,j] = self.K_n(self.k, x[i], y[j])
                     
        return K

        
      



            
            
        
        
        
        
        
        