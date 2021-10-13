# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 21:35:24 2021

@author: Lab
"""

import numpy as np

A = np.array([[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24]])

B = np.random.rand(3,3)

#%%
H, W = A.shape
Hk, Wk = B.shape
