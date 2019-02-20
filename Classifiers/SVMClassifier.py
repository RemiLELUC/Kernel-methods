#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Fri Feb  8 21:52:27 2019

@author: gtallec
"""
from . import Classifier               
from qpsolvers import solve_qp  
from cvxopt import solvers, matrix

import numpy as np        
import matplotlib.pyplot as plt



class SVMClassifier(Classifier.Classifier):
    
    def __init__(self):
        Classifier.Classifier.__init__(self)
        self.lam = None
        self.predictor=None
        
    def setParams(self):
        if len(self.hyperParameters) != 1:
            print('WRONG PARAMETRIZATION OF SVM MODEL')
        self.lam = self.hyperParameters[0]
            
        
    def fit(self, embedding, labels):
        kernel_mat = embedding
        n = labels.shape[0]
        P = matrix(kernel_mat, tc = 'd')
        q = matrix(-labels, tc = 'd')
        G = matrix(np.vstack([np.diag(labels), np.diag(-labels)]), tc = 'd')
        h = matrix(np.hstack([(1/(2*self.lam*n))*np.ones((n,)), np.zeros((n,))]), tc = 'd')
        self.predictor = np.array(solvers.qp(P=P,
                                  q=q,
                                  G=G,
                                  h=h)['x'])
        
    def predict(self,test_embedding):
        return np.sign((self.predictor.reshape(-1,1)).T@test_embedding)[0]

""" 
    def plot_boundaries(self):
        inputs = self.inputs
        labels = self.labels
        n = inputs.shape[0]
        
        inputl_1 = inputs[labels==1]
        inputl_2 = inputs[labels==-1]
        
        coefs = (self.predictor.reshape(n,1)).T@inputs
        coef_dir = coefs[0][0]/coefs[0][1]
        
        X = np.linspace(-5,5,10)
        Y_1 = 1 - coef_dir*X
        Y_2 = -1 - coef_dir*X
        plt.plot(X,Y_1)
        plt.plot(X,Y_2)
        plt.scatter(inputl_1[:,0], inputl_1[:,1],color='blue')
        plt.scatter(inputl_2[:,0], inputl_2[:,1],color='red')
        plt.grid(True)
        plt.axis()
        plt.show()
"""     

        
        
                
                
            
        
        
        
    
    