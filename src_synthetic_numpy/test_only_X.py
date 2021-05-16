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


def myalgo(mask, T, R, iteration):
    
    # set up
    d1, d2, d3 = T.shape
    
    def interpolation(A, A_):
        d, _ = A.shape
        tmp = np.zeros(A.shape)
        if (d % 2 == 0):
            tmp[::2, :] = A_
            tmp[1:-1:2, :] = (A_[:-1, :] + A_[1:, :]) / 2
            tmp[-1, :] = A_[-1, :]
        elif (d % 2 == 1):
            tmp[::2, :] = A_
            tmp[1::2, :] = (A_[:-1, :] + A_[1:, :]) / 2
        return tmp
    
    if (max(d1, d2, d3) <= 2 * R) and (args.multi == 1) or args.multi == 0:
        A1 = np.random.random((d1,R))
        A2 = np.random.random((d2,R))
        A3 = np.random.random((d3,R))
    else:

        # subsampling
        T_ = T.copy()
        mask_ = mask.copy()
        if d1 >= 2 * R:
            mask_ = mask_[:d1//2, :, :]
            T_ = T_[:d1//2, :, :]
        if d2 >= 2 * R:
            mask_ = mask_[:, ::2, :]
            T_ = T_[:, ::2, :]
        if d3 >= 2 * R:
            mask_ = mask_[:, :, ::2]
            T_ = T_[:, :, ::2]

        # solve low-resolution problem
        A1_low, A2_low, A3_low, _, _ = myalgo(mask_, T_, R=R, iteration=20)

        # interpolate
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
        
    print ('mode1:', d1, 'mode2:', d2, 'mode3:', d3)
    loss_list, rec_list = [], []
    
    tic = time.time()
    rec = np.zeros(T.shape)
    for i in range(iteration): 

        T_ = mask * T + (1 - mask) * rec
        
        # ALS
        A1 = optimize(A1.T, (A2.T@A2)*(A3.T@A3), np.einsum('ijk,jr,kr->ri',T_,A2,A3,optimize=True), i).T
        A2 = optimize(A2.T, (A1.T@A1)*(A3.T@A3), np.einsum('ijk,ir,kr->rj',T_,A1,A3,optimize=True), i).T
        A3 = optimize(A3.T, (A1.T@A1)*(A2.T@A2), np.einsum('ijk,ir,jr->rk',T_,A1,A2,optimize=True), i).T
        rec = np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True)

        # rescaling
        norm1 = (A1**2).sum(axis=0) ** .5
        norm2 = (A2**2).sum(axis=0) ** .5
        norm3 = (A3**2).sum(axis=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)

        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3
        
        LOSS = la.norm(mask * (rec - T)) ** 2
        recLOSS = 1 - la.norm(rec - T) / la.norm(T)
        if i % 10 == 0:
            print ('{}/{}'.format(i, iteration), 'loss:', LOSS, 'rec:', recLOSS, 'time: ', time.time() - tic)
        tic = time.time()

        loss_list.append(LOSS)
        rec_list.append(recLOSS)
        
    return A1, A2, A3, loss_list, rec_list

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
    C2_binary = C2 > 0

    # rank the fine first mode
    rankIdx3 = ss.rankdata(C2_binary.sum(axis=1).sum(axis=1), method='ordinal')
    rankIdx3 = len(rankIdx3) - rankIdx3 

    # apply ranking to all related mode
    T = T[rankIdx3, :, :]


    for j in range(1):
        np.random.seed(j)
        mask = np.random.random(T.shape) < args.partial

        A1, A2, A3, loss, rec = myalgo(mask, T, R=10, iteration=501)
        pickle.dump([A1, A2, A3, loss, rec], open('synthetic_result/X_{}_jacobi_{}_multi_{}.pkl'.format(args.jacobi, args.multi, j), 'wb'))
