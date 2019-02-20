#!/usr/bin/env python
# coding: utf-8

# # KERNEL DATA CHALLENGE
# ## AUTHORS : Thibault Desfontaines, Rémi Leluc, Gauthier Tallec
import Classifiers.SVMClassifier as svm

import Preprocessers.PCA as pca

import ModelTesters.ModelOptimizer as mo

import Kernels.LinearKernel as lker
import Kernels.GaussianKernel as gker
import Kernels.KSpectrumKernel as kspker


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




def bound_normalisation(x):
    if x == 0:
        return -1
    else:
        return x


# ## DATA IMPORTS

filepath = 'challenge-dataset/'
##Training sets
###Inputs
Xtr0_mat100 = pd.read_csv(filepath + 'Xtr0_mat100.csv', delimiter=' ', header = None).values
Xtr1_mat100 = pd.read_csv(filepath + 'Xtr1_mat100.csv', delimiter=' ', header = None).values
Xtr2_mat100 = pd.read_csv(filepath + 'Xtr2_mat100.csv', delimiter=' ', header = None).values

Xtr0 = pd.read_csv(filepath + 'Xtr0.csv', delimiter=',', header = 0)['seq'].values
Xtr1 = pd.read_csv(filepath + 'Xtr1.csv', delimiter=',', header = 0)['seq'].values
Xtr2 = pd.read_csv(filepath + 'Xtr2.csv', delimiter=',', header = 0)['seq'].values

###Labels
Ytr0 = (pd.read_csv(filepath + 'Ytr0.csv', delimiter=',', sep='\n', header = 0)['Bound']
          .map(bound_normalisation)
          .values
       )

Ytr1 = (pd.read_csv(filepath + 'Ytr1.csv', delimiter=',', sep='\n', header = 0)['Bound']
          .map(bound_normalisation)
          .values
       )
Ytr2 = (pd.read_csv(filepath + 'Ytr2.csv', delimiter=',', sep='\n', header = 0)['Bound']
          .map(bound_normalisation)
          .values
       )

##Testing Sets
Xte0_mat100 = pd.read_csv(filepath + 'Xte0_mat100.csv', delimiter=' ', header = None).values
Xte1_mat100 = pd.read_csv(filepath + 'Xte1_mat100.csv', delimiter=' ', header = None).values
Xte2_mat100 = pd.read_csv(filepath + 'Xte0_mat100.csv', delimiter=' ', header = None).values

Xte0 = pd.read_csv(filepath + 'Xte0.csv', delimiter=',', header = 0)['seq'].values
Xte1 = pd.read_csv(filepath + 'Xte1.csv', delimiter=',', header = 0)['seq'].values
Xte2 = pd.read_csv(filepath + 'Xte2.csv', delimiter=',', header = 0)['seq'].values


# ## MODEL TESTS
# 
# ### SVM CLASSIFIER WITH K-SPECTRUM KERNEL

#KERNEL PART
##Kernel parameters
alphabet = ['A','T','G','C']
k_spec = 4

##Kernel Instanciation
k_spectral_kernel = kspker.KSpectrumKernel(k = k_spec,
                                           alphabet = alphabet)

#SVM PART
##Regularization Grid Search parameter
hyper_parameters_list = np.exp(np.arange(-5,3)*np.log(10)).reshape(-1,1)

##SVM Instanciation
svm_classifier= svm.SVMClassifier(kernel = k_spectral_kernel)


#MODEL OPTIMIZATION PART
##Cross validation parameters
k_fold = 10

##Model Instanciation
model_optimizer = mo.ModelOptimizer(svm_classifier)
optimal_parameters = model_optimizer.find_optimal_parameters(k_fold, Xtr0, Ytr0, hyper_parameters_list)

print(optimal_parameters)


"""
#Perform PCA with linear Kernel
pca_kernel = lker.LinearKernel()
pca_agent = pca.PCA()
pca_inputs_al = pca_agent.perform_PCA(inputs = inputs_al,
                                      kernel = pca_kernel,
                                      dim_num =3)

pca_inputs_tr = pca_inputs_al[:1900]
pca_inputs_te = pca_inputs_al[1900:]
"""


# In[ ]:


"""
#Computing bandwidth for gaussian kernel as mean norms of all distances between vectors
n = pca_inputs_al.shape[0]
Inputs_dup = np.tile(pca_inputs_al[np.newaxis,:,:], reps = (n,1,1))
mean_distance = np.mean(np.linalg.norm(Inputs_dup - np.einsum('pnm-> npm', Inputs_dup), axis = 2))
del Inputs_dup
"""


# In[ ]:


"""
lam=10e-4
gamma = 1
bandwidth = gamma*mean_distance
kernel_svm = gker.GaussianKernel(bandwidth)
"""


# In[ ]:


"""
svmClassifier = svm.SVMClassifier(lam = lam, kernel = kernel_svm)
svmClassifier.fit(pca_inputs_tr, labels_tr.astype(np.double))
"""


# In[ ]:


"""
accuracy = svmClassifier.compute_test_accuracy(pca_inputs_te, labels_te)
"""


# In[ ]:


"""
print(accuracy)
"""

