from scipy.optimize import nnls
import numpy as np
from scipy import linalg as la
import pickle
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--partial', type=float, default=0.99, help="partial observation")
parser.add_argument('--jacobi', type=int, default=0, help="Jacobi (1) or not (0)")
parser.add_argument('--multi', type=int, default=0, help="multiresolution (1) or not (0)")
args = parser.parse_args()

def jacobi(u, A, b):

    D = np.diag(A) + 1e-8
    R_ = A - np.diag(D)
    Z = np.diag(D**(-1)) @ b
    B = - np.diag(D**(-1)) @ R_

    w = 2e-2
    for i in range(5):
        u = (1-w) * u + w * (Z + B @ u)
    return u
    
def optimize(u, A, B, iteration):

    if (iteration < 5) and (args.jacobi == 1):
        u = jacobi(u, A, B)
    else:
        L = la.cholesky(A + np.eye(A.shape[1]) * 1e-8)
        y = la.solve_triangular(L.T, B, lower=True)
        u = la.solve_triangular(L, y, lower=False)
        # u = la.solve(A + np.eye(A.shape[0]) * 1e-8, B)
    return u


def myalgo(mask, T, C1, C2, R, iteration):
        
    # set up
    rec = 0
    d1, d2, d3 = T.shape
    new_d1, _, _ = C1.shape
    _, new_d2, _ = C2.shape
    
    def interpolation(A, A_):
        d, _ = A.shape
        tmp = np.zeros(A.shape)
        if (d % 2 == 0):
            tmp[::2, :] = A_
            tmp[1:-1:2, :] = (A_[:-1, :] + A_[1:, :]) / 2
            tmp[-1, :] = A_[-1, :]
        if (d % 2 == 1):
            tmp[::2, :] = A_
            tmp[1::2, :] = (A_[:-1, :] + A_[1:, :]) / 2
        return tmp
    
    tic = time.time()
    if (max(d1, d2, d3, new_d1, new_d2) <= 2 * R) and (args.multi == 1) or (args.multi == 0):
        A1 = np.random.random((d1,R))
        A2 = np.random.random((d2,R))
        A3 = np.random.random((d3,R))
        Q1 = np.random.random((new_d1,R))
        Q2 = np.random.random((new_d2,R))
    else:
        mask_ = mask.copy()
        T_ = T.copy()
        C1_ = C1.copy()
        C2_ = C2.copy()
        if d1 >= 2 * R:
            mask_ = mask_[:d1//2, :, :]
            T_ = T_[:d1//2, :, :]
            C2_ = C2_[:d1//2, :, :]
        if d2 >= 2 * R:
            mask_ = mask_[:, ::2, :]
            T_ = T_[:, ::2, :]
            C1_ = C1_[:, ::2, :]
        if d3 >= 2 * R:
            mask_ = mask_[:, :, ::2]
            T_ = T_[:, :, ::2]
            C1_ = C1_[:, :, ::2]
            C2_ = C2_[:, :, ::2]
        if new_d1 >= 2 * R:
            C1_ = C1_[:new_d1//2, :, :]
        if new_d2 >= 2 * R:
            C2_ = C2_[:, ::2, :]
        A1_low, A2_low, A3_low, Q1_low, Q2_low, _, _ = myalgo(mask_, T_, C1_, C2_, R=R, iteration=20)

        if d1 >= 2 * R:
            A1 = np.random.random((d1,R))
            A1[:d1//2,:] = A1_low
        else:
            A1 = A1_low
        if d2 >= 2 * R:
            A2 = np.random.random((d2,R))
            A2 = interpolation(A2, A2_low)
        else:
            A2 = A2_low
        if d3 >= 2 * R:
            A3 = np.random.random((d3,R))
            A3 = interpolation(A3, A3_low)
        else:
            A3 = A3_low
        if new_d1 >= 2 * R:
            Q1 = np.random.random((new_d1,R))
            Q1[:new_d1//2,:] = Q1_low
        else:
            Q1 = Q1_low
        if new_d2 >= 2 * R:
            Q2 = np.random.random((new_d2,R))
            Q2 = interpolation(Q2, Q2_low)
        else:
            Q2 = Q2_low
        
    print (time.time() - tic)
    print ('---')
    print ('mode1:', d1, 'mode2:', d2, 'mode3:', d3, 'Q1 mode:', new_d1, 'Q2 mode:', new_d2)
        
    loss_func_list, loss_rec_list = [], []
    
    tic = time.time()
    for i in range(iteration): 

        LAMBDA = 1. * np.exp(-i / 20)
        # partial
        T_ = mask * T + (1-mask) * rec
        
        Q1 = optimize(Q1.T, (A3.T@A3)*(A2.T@A2), np.einsum('ijk,jr,kr->ri',C1,A2,A3,optimize=True), i).T

        Q2 = optimize(Q2.T, (A3.T@A3)*(A1.T@A1), np.einsum('ijk,ir,kr->rj',C2,A1,A3,optimize=True), i).T
        
        A1 = optimize(A1.T, LAMBDA * (Q2.T@Q2) * (A3.T@A3) + (A2.T@A2)*(A3.T@A3), LAMBDA * np.einsum('ijk,jr,kr->ri',C2,Q2,A3,optimize=True) +\
                    np.einsum('ijk,jr,kr->ri',T_,A2,A3,optimize=True), i).T
        
        A2 = optimize(A2.T, LAMBDA * (Q1.T@Q1)*(A3.T@A3) + (A1.T@A1)*(A3.T@A3), LAMBDA * np.einsum('ijk,ir,kr->rj',C1,Q1,A3,optimize=True) + \
                    np.einsum('ijk,ir,kr->rj',T_,A1,A3,optimize=True), i).T

        A3 = optimize(A3.T, LAMBDA * (Q1.T@Q1)*(A2.T@A2) + (A1.T@A1)*(A2.T@A2) + LAMBDA * (A1.T@A1)*(Q2.T@Q2), LAMBDA * np.einsum('ijk,ir,jr->rk',C1,Q1,A2,optimize=True) + \
                    LAMBDA * np.einsum('ijk,ir,jr->rk',C2,A1,Q2,optimize=True) + np.einsum('ijk,ir,jr->rk',T_,A1,A2,optimize=True), i).T
    
        norm1 = (A1**2).sum(axis=0) ** .5
        norm2 = (A2**2).sum(axis=0) ** .5
        norm3 = (A3**2).sum(axis=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)
        
        rec = np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True)

        LOSS1 = la.norm(np.einsum('ir,jr,kr->ijk',Q1,A2,A3,optimize=True)) / la.norm(C1)
        LOSS2 = la.norm(np.einsum('ir,jr,kr->ijk',A1,Q2,A3,optimize=True)) / la.norm(C2)
        LOSS3 = la.norm(mask * (rec - T)) / la.norm(mask * T)
        LOSS = LAMBDA * LOSS1 ** 2 + LAMBDA * LOSS2 ** 2 + LOSS3 ** 2

        recLOSS = 1 - la.norm(rec - T) / la.norm(T)

        print ('{}/{}'.format(i, iteration), 'loss:', LOSS, 'rec:', recLOSS, 'time: ', time.time() - tic)
        # print (i, time.time() - tic)
        tic = time.time()
        loss_func_list.append(LOSS)
        loss_rec_list.append(recLOSS)
        
        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3
        Q1 *= norm / norm1
        Q2 *= norm / norm2
        
    return A1, A2, A3, Q1, Q2, loss_func_list, loss_rec_list

if __name__ == '__main__':

    T = pickle.load(open('../data_synthetic/X.pkl', 'rb'))
    C1 = pickle.load(open('../data_synthetic/C1.pkl', 'rb'))
    C2 = pickle.load(open('../data_synthetic/C2.pkl', 'rb'))
    P1 = pickle.load(open('../data_synthetic/P1.pkl', 'rb'))
    P2 = pickle.load(open('../data_synthetic/P2.pkl', 'rb'))

    """
        First, sort on the categorical mode.
        The code provides a more efficient way than paper does.
        We directly sort in descent order at the beginning and pick [:d//2] during subsampling
    """

    import scipy.stats as ss
    C1_binary = C1 > 0

    # rank the coarse first mode
    rankIdx1 = ss.rankdata(C1_binary.sum(axis=1).sum(axis=1), method='ordinal')
    rankIdx1 = len(rankIdx1) - rankIdx1   

    C2_binary = C2 > 0

    # rank the fine first mode
    rankIdx3 = ss.rankdata(C2_binary.sum(axis=1).sum(axis=1), method='ordinal')
    rankIdx3 = len(rankIdx3) - rankIdx3  

    # apply ranking to all related mode
    T = T[rankIdx3, :, :]
    C1 = C1[rankIdx1, :, :]
    C2 = C2[rankIdx3, :, :]

    for j in range(1):
        np.random.seed(j)
        mask = np.random.random(T.shape) < args.partial

        A1, A2, A3, Q1, Q2, loss, rec = myalgo(mask, T, C1, C2, R=10, iteration=100)
        # pickle.dump([A1, A2, A3, Q1, Q2, loss, rec], open('synthetic_result/X_C1_C2_unknown_{}_jacobi_{}_multi_{}.pkl'.format(args.jacobi, args.multi, j), 'wb'))
