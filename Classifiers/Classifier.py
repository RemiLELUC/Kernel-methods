#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 21:44:41 2019

@author: gtallec
"""
import numpy as np
class Classifier:
    
    def __init__(self):
        self.predictions = None
        self.hyperParameters = []
        
    def setHyperParameters(self,hyperParameters):
        self.hyperParameters = hyperParameters
        self.setParams()
        
    def setParams(self):
        pass
    
    def fit(self, train_embedding, labels):
        pass
    

        

    
        

        