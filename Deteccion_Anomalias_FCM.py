# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:47:10 2018

@author: jamontol
"""

import skfuzzy as fuzz
from itertools import islice
import numpy as np


class Deteccion_Anomalias_FCM:
    
    def __init__(self, data, modo, q, r):
         
        self.t = data[0]
        self.y = data[1]
        
        self.r = r
        self.q = q
        self.n = (int) (len(self.y)-q)/r+1
        
        self.m = 2 #coeficiente de fuzzificacion
        
        self.secuencias = []
        
        self.modo = modo
        
        for w in self.window(self.y):
      
            self.secuencias.append(w)
            
        pass
            
#        for w in self.window(self.y,self.n):
#        
#            self.secuencias = w

    def window(self,seq):
    
        "Returns a sliding window (of width q and step r) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = list(islice(it, self.q))
        if len(result) == self.q:
            yield result
        step_size = max(1, int(round(self.r)))
        
        while True:
            new_elements = list(islice(it, step_size))
            
            if len(new_elements) < step_size:
                break
            result = result[step_size:] + new_elements
            yield result
        
#        for elem in it:
#            result = result[1:] + (elem,)
#            yield result
    
    def A_priori(self):
        
        pass
    
    
    def Fuzzy_Clustering_Method(self, matriz_data):
        
        fpcs = []
            
        for ncenters in range(2,10):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(matriz_data, ncenters, self.m, error=0.005, maxiter=1000, init=None)
        
            # Store fpc values for later
            fpcs.append(fpc)
            # Plot assigned clusters, for each data point in training set
            #cluster_membership = np.argmax(u, axis=0)
        
        n_centros_opt = np.argmax(fpcs)+2 #entre 2 y n clusters
        
        cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(matriz_data, 2, self.m, error=0.005, maxiter=1000)
        
        secuencias_prima = []
        
        for k, x in enumerate(self.secuencias):
            
#            if (k==478): 
#                print('Stop')
            secuencias_prima.append(np.matmul(np.power(u_orig[:,k],self.m),cntr)/(np.sum(np.power(u_orig[:,k],self.m)))) 
        
        matriz_data_prima = np.stack(secuencias_prima,1)
        
        Error=[]
        
        for k, x in enumerate(self.secuencias):
            Error.append(np.asscalar(np.linalg.norm(matriz_data[:,k] - matriz_data_prima[:,k], 2, None, 1)))
        
        return Error
        
    def Aurocorrelation(self, X):
        
        
#        
#        suma = np.dot(X_m[1:],X_m[:-1])
#        cociente = sum(k*k  for k in X_m)
#        Xdm_auto1 = suma/cociente
#        
        nx = len(X)  
        
        X = X.transpose()

        X_list = X.tolist()
        X_auto_list = []
        
        for k,valor in enumerate(X):
        
            # Remove sample mean.
            
            Xdm = X_list[k]- np.mean(X_list[k]) # Si es un array de 0's la autocorrelacion será una array de NaN's
            
            Xdm_auto = np.correlate(Xdm, Xdm, mode='full')
            
            Xdm_auto /= Xdm_auto[nx-1]
            
            X_auto_list.append(Xdm_auto[nx:])
            
        X_auto = np.stack(X_auto_list,1)
        
        return X_auto
        
    def explain_anomalies(self):
        
        X_data = np.stack(self.secuencias,1)
        
        if (self.modo == 'amplitude'):
           
            anomalia_grado = self.Fuzzy_Clustering_Method(X_data)
            
        elif (self.modo == 'shape'):
            
            Y_data = self.Aurocorrelation(X_data)
            
            anomalia_grado = self.Fuzzy_Clustering_Method(Y_data)
        
        return anomalia_grado
            
        
        
        