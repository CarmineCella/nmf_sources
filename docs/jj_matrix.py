# jj_matrix.py
#     created by J.J. Burred on 29-7-2019

import numpy as np

EPS = np.finfo(np.float32).eps  # as used in sklearn (nmf.py)

# mean squared error
def jj_mse(a,b):
    return ((np.abs(a)-np.abs(b))**2).mean()

def jj_nmf(X, k=10, it=100, cost='eucl'):
    # VERIFIED with sklearn.decomposition.nmf (29-7-2019)

    nr = X.shape[0]
    nc = X.shape[1]

    # init matrices (sklearn method):
    avg = np.sqrt(X.mean() / k)
    rng = np.random.RandomState(0)  # fixed random seed
    H = avg * rng.randn(k, nc)
    W = avg * rng.randn(nr, k)
    np.abs(W, W)
    np.abs(H, H)

    if cost=='eucl':
        for i in range(it):
            # update W
            N = np.dot(X,H.T)
            P = np.dot(np.dot(W,H),H.T)
            P[P == 0] = EPS
            W *= np.divide(N,P)
            # update H
            N = np.dot(W.T,X)
            P = np.dot(np.dot(W.T,W),H)
            P[P == 0] = EPS
            H *= np.divide(N,P)

    elif cost=='kl':
        I = np.ones([nr,nc])
        for i in range(it):
            # update W
            WH = np.dot(W,H)
            # WH += EPS
            WH[WH == 0] = EPS
            N = np.dot(np.divide(X,WH),H.T)
            P = np.dot(I,H.T)
            # P += EPS
            P[P == 0] = EPS
            W *= np.divide(N,P)
            #update H
            WH = np.dot(W,H)
            # WH += EPS
            WH[WH == 0] = EPS
            N = np.dot(W.T,np.divide(X,WH))
            P = np.dot(W.T,I)
            # P += EPS
            P[P == 0] = EPS
            H *= np.divide(N,P)

    elif cost=='is':
        I = np.ones([nr,nc])
        for i in range(it):
            # update W
            WH = np.dot(W,H)
            WH[WH == 0] = EPS
            N = np.dot(np.divide(X,WH**2),H.T)
            P = np.dot(np.divide(I,WH),H.T)
            P[P == 0] = EPS
            W *= np.divide(N,P)**0.5              # the 0.5 exponent is recommended by Fevotte 2011 (MM algo), and implemented in sklearn
            W[W < np.finfo(np.float64).eps] = 0.  # done in sklearn for stability (nmf.py line 796)
            # update H
            WH = np.dot(W,H)
            WH[WH == 0] = EPS
            N = np.dot(W.T,np.divide(X,WH**2))
            P = np.dot(W.T,np.divide(I,WH))
            P[P == 0] = EPS
            H *= np.divide(N,P)**0.5
            H[H < np.finfo(np.float64).eps] = 0.

    return W,H
