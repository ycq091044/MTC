
import pickle

import numpy as np
from scipy import linalg as la
import time


def optimize(u, A, B):

    # u = np.linalg.solve(A + np.eye(A.shape[1]) * 1e-8, B)
    L = la.cholesky(A + np.eye(A.shape[1]) * 1e-8)
    y = la.solve_triangular(L.T, B, lower=True)
    u = la.solve_triangular(L, y, lower=False)
    return u


def CPD(mask, T, R, iteration):
    # time_std = [[time.time(),0]] ###########################
    loss_list, rec_list = [], []
    
    d1, d2, d3 = T.shape
    A1 = np.random.random((d1, R))
    A2 = np.random.random((d2, R))
    A3 = np.random.random((d3, R))
    
    tic = time.time()
    T_1 = np.transpose(T, (0,1,2)).reshape(d1,-1)
    T_2 = np.transpose(T, (1,0,2)).reshape(d2,-1)
    T_3 = np.transpose(T, (2,0,1)).reshape(d3,-1)
    toc1 = time.time()
    
    tic = time.time()

    # ALS
    for i in range(iteration):
        # ###########################
        # time_std.append([time.time(), time.time() - time_std[-1][0]])
        # if i > 0:
        #     estimate = sum([item[1] for item in time_std[2:]]) / i * 200
        #     std = np.std([item[1] for item in time_std[2:]])
        #     print (i, '{:.6} $\pm$ {:.4}'.format(estimate + time_std[1][1], std * 200))
        # ###########################
        
        A1 = optimize(A1.T, (A3.T@A3)*(A2.T@A2), np.einsum('ijk,jr,kr->ri',T,A2,A3,optimize=True)).T

        A2 = optimize(A2.T, (A1.T@A1)*(A3.T@A3), np.einsum('ijk,ir,kr->rj',T,A1,A3,optimize=True)).T
        
        A3 = optimize(A3.T, (A1.T@A1)*(A2.T@A2), np.einsum('ijk,ir,jr->rk',T,A1,A2,optimize=True)).T

        rec = np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True)
        LOSS = la.norm(mask * (rec - T))/ la.norm(mask * T)
        recLOSS = 1 - la.norm(rec - T) / la.norm(T)

        print (i, LOSS, recLOSS, 'time span: ', time.time() - tic)
        tic = time.time() 
        loss_list.append(LOSS)
        rec_list.append(recLOSS)

    return A1, A2, A3, loss_list, rec_list


if __name__ == '__main__':

    # GCSS large
    # T = pickle.load(open('../data_GCSS/X_google.pkl', 'rb'))
    # O1 = pickle.load(open('../data_GCSS/O1_google.pkl', 'rb'))
    # O2 = pickle.load(open('../data_GCSS/O2_google.pkl', 'rb'))
    # P1 = pickle.load(open('../data_GCSS/P1_google.pkl', 'rb'))
    # P2 = pickle.load(open('../data_GCSS/P2_google.pkl', 'rb'))

    # GCSS small
    T = pickle.load(open('../data_GCSS/X_small_google.pkl', 'rb'))
    O1 = pickle.load(open('../data_GCSS/O1_small_google.pkl', 'rb'))
    O2 = pickle.load(open('../data_GCSS/O2_small_google.pkl', 'rb'))
    P1 = pickle.load(open('../data_GCSS/P1_small_google.pkl', 'rb'))
    P2 = pickle.load(open('../data_GCSS/P2_small_google.pkl', 'rb'))

    for i in range(3):
        np.random.seed(i)
        mask = np.random.random(T.shape) > 0.95

        A1, A2, A3, _, _, loss, rec = CPD(mask, T, R=20, iteration=200)
        # pickle.dump([A1, A2, A3, loss, rec], open('../result_final_google/CPD_{}.pkl'.format(i), 'wb'))
