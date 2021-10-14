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

def _pad(X, k, padder=0):
    XX_shape = tuple(np.subtract(X.shape, k.shape) + 1)
    if XX_shape!=X.shape:
        P = np.subtract(X.shape, XX_shape) // 2
        if len(np.unique(P))!=1:
            print('kernel is not square matrix')
        else:
            X_ = np.pad(X, P[0], _pad_with, padder=padder)
    else:
        X_ = np.copy(X)
    return X_

def _dilation(X, k):
    X_pad = _pad(X, k)
    H, W = X_pad.shape
    Hk, Wk = k.shape
    X_ = np.copy(X_pad)
    for h in range(H):
        for w in range(W):
            if w+Wk<=W and h+Hk<=H:
                window = X_pad[h:h+Hk, w:w+Wk]
                window_f = window.flatten()
                if window_f[len(window_f) // 2] == 1:
                    X_[h:h+Hk, w:w+Wk] = k         
    return X_

def _erosion(X, k):
    X_pad = _pad(X, k)
    H, W = X_pad.shape
    Hk, Wk = k.shape
    X_ = np.zeros_like(X_pad)
    for h in range(H):
        for w in range(W):
            if w+Wk<=W and h+Hk<=H:
                window = X_pad[h:h+Hk, w:w+Wk]
                if np.min(window==k):
                    X_[h + (Hk - 1) // 2, w + (Wk - 1) // 2] = 1
    # pad 取消
    return X_ 

#=============================================
def _dilation2(X, k):
    X = ~(X.astype(bool)) * 1
    X_pad = _pad(X, k, padder=1)
    view_shape = tuple(np.subtract(X_pad.shape, k.shape) + 1) + k.shape
    strides = X_pad.strides + X_pad.strides
    sub_matrices = as_strided(X_pad, view_shape, strides) 
    cv = np.einsum('klij,ij->kl', sub_matrices, k)
    cv_ = cv // (k.shape[0] * k.shape[1]) 
    
    return ~(cv_.astype(bool)) * 1
    
def _erosion2(X, k):
    X_pad = _pad(X, k)
    view_shape = tuple(np.subtract(X_pad.shape, k.shape) + 1) + k.shape
    strides = X_pad.strides + X_pad.strides
    sub_matrices = as_strided(X_pad, view_shape, strides) 
    cv = np.einsum('klij,ij->kl', sub_matrices, k)
    
    return cv // (k.shape[0] * k.shape[1])

def _conv2d(X, k):
    X_pad = _pad(X, k)
    view_shape = tuple(np.subtract(X_pad.shape, k.shape) + 1) + k.shape
    strides = X_pad.strides + X_pad.strides
    sub_matrices = as_strided(X_pad, view_shape, strides) 
    cv = np.einsum('klij,ij->kl', sub_matrices, k)
    return cv    

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g  

def _adpt_thold(image, kernel_size=11, c=2):
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    k = gaussian_kernel(kernel_size, sigma=sigma)
    img_pad = _pad(image, k)
    view_shape = tuple(np.subtract(img_pad.shape, k.shape) + 1) + k.shape
    strides = img_pad.strides + img_pad.strides
    sub_matrices = as_strided(img_pad, view_shape, strides) 
    thold = np.einsum('klij,ij->kl', sub_matrices, k) / np.sum(k) - c
    return thold

#=============================================

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
view_shape = tuple(np.subtract(A.shape, sub_shape) + 1) + sub_shape
strides = A.strides + A.strides
sub_matrices = as_strided(A,view_shape,strides) 

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
