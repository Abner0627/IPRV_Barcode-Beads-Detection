import numpy as np

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g  

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

def _view_matrix(X, k):
    H, W = X.shape
    Hk, Wk = k.shape
    view_shape = tuple(np.subtract(X.shape, k.shape) + 1) + k.shape 
    res = []
    for h in range(H):
        for w in range(W):
            if w+Wk<=W and h+Hk<=H:
                res.append(X[h:h+Hk, w:w+Wk]) 
    res_ = np.array(res).reshape(view_shape) 
    return res_  


def _adpt_thold(image, kernel_size=11):
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    k = gaussian_kernel(kernel_size, sigma=sigma)
    img_pad = _pad(image, k)
    view = _view_matrix(img_pad, k)
    thold = np.einsum('klij,ij->kl', view, k) / np.sum(k) - 2 
    return thold
