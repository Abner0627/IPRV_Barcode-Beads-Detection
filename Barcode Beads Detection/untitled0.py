# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 21:35:24 2021

@author: Lab
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

#%%
def _pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def _pad(X, k):
    XX_shape = tuple(np.subtract(X.shape, k.shape) + 1)
    if XX_shape!=X.shape:
        P = np.subtract(X.shape, XX_shape) // 2
        if len(np.unique(P))!=1:
            print('kernel is not square matrix')
        else:
            X_ = np.pad(X, P[0], _pad_with)
    else:
        X_ = np.copy(X)
    return X_

#%%
A = np.array([[0,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,0],
             [0,0,0,1,1,1,1,0],
             [0,1,1,1,1,1,1,0],
             [0,1,1,1,1,1,1,0],
             [0,1,1,1,1,0,0,0],
             [0,1,1,1,1,0,0,0],
             [0,0,0,0,0,0,0,0]])

B = np.ones((3, 3))
sub_shape = (3,3)

def _kick(res, sub_shape):
    L = []
    for i in res:
        if i.shape==sub_shape:
            L.append(i)
    return L

H, W = A.shape
Hk, Wk = B.shape

res = []
for h in range(H):
    for w in range(W):
       res.append(A[h:h+Hk, w:w+Wk]) 
L = _kick(res, sub_shape)

'''
#%%
# =============================================================================
# A_ = np.array([[ 0,  1,  2,  3,  4],
#               [ 5,  6,  7,  8,  9],
#               [10, 11, 12, 13, 14],
#               [15, 16, 17, 18, 19],
#               [20, 21, 22, 23, 24]])
# =============================================================================

# =============================================================================
# A = np.array([[ 0,  1,  2,  3,  4],
#               [ 5,  6,  7,  8,  9],
#               [10, 11, 12, 13, 14],
#               [15, 16, 17, 18, 19]])
# =============================================================================

A_ = np.ones((5,5))

B = np.ones((3,3))

A = _pad(A_, B)

#%%
sub_shape = (3,3)
view_shape = tuple(np.subtract(A.shape, sub_shape) + 1) + sub_shape     
# 計算視野域(H', W', Hk, Wk) 
# H' = H - (K - 1)

strides = A.strides + A.strides

sub_matrices = as_strided(A,view_shape,strides)

#%%
def _kick(res, sub_shape):
    L = []
    for i in res:
        if i.shape==sub_shape:
            L.append(i)
    return L

H, W = A.shape
Hk, Wk = B.shape

res = []
for h in range(H):
    for w in range(W):
       res.append(A[h:h+Hk, w:w+Wk]) 
L = _kick(res, sub_shape)
    
res_ = np.array(L).reshape(view_shape)

print(np.min(sub_matrices==res_))

AA = np.einsum('klij,ij->kl', sub_matrices, B)
BB = AA / 9
'''
