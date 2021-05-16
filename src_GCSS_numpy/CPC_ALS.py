import pickle

import numpy as np
from scipy import linalg as la
import time

parser = argparse.ArgumentParser()
parser.add_argument('--partial', type=float, default=0.95, help="partial observation")
args = parser.parse_args()

def optimize(u, A, B):

    # u = np.linalg.solve(A + np.eye(A.shape[1]) * 1e-8, B)
    L = la.cholesky(A_ + np.eye(A_.shape[1]) * 1e-8)
    y = la.solve_triangular(L.T, B, lower=True)
    u = la.solve_triangular(L, y, lower=False)
    return u


def CPC(mask, T, R, iteration):
    # time_std = [[time.time(),0]] ###########################
    loss_list, rec_list = [], []
    
    d1, d2, d3 = T.shape
    A1 = np.random.random((d1, R))
    A2 = np.random.random((d2, R))
    A3 = np.random.random((d3, R))
    
    tic = time.time()
    toc1 = time.time()
    
    tic = time.time()

    def Solve_Factor(Omega,A,RHS,num,reg):
        N = len(A)
        R = A[0].shape[1]
        lst_mat = []
        T_inds = "".join([chr(ord('a')+i) for i in range(Omega.ndim)])
        einstr=""
        for i in range(N):
            if i != num:
                einstr+=chr(ord('a')+i) + 'r' + ','
                lst_mat.append(A[i])
                einstr+=chr(ord('a')+i) + 'z' + ','
                lst_mat.append(A[i])
        einstr+= T_inds + "->"+chr(ord('a')+num)+'rz'
        lst_mat.append(Omega)
        P = np.einsum(einstr,*lst_mat,optimize=True)
        o = np.zeros_like(RHS)
        for j in range(A[num].shape[0]):
            o[j,:] = la.solve(P[j]+reg*np.eye(R),RHS[j,:])
        return o

    # ALS

    T_ = mask * T

    for i in range(iteration):
        # ###########################
        # time_std.append([time.time(), time.time() - time_std[-1][0]])
        # if i > 0:
        #     estimate = sum([item[1] for item in time_std[2:]]) / i * 200
        #     std = np.std([item[1] for item in time_std[2:]])
        #     print (i, '{:.6} $\pm$ {:.4}'.format(estimate + time_std[1][1], std * 200))
        # ###########################

        A1 = Solve_Factor(mask, [A1, A2, A3], np.einsum('ijk,jr,kr->ir',T_,A2,A3,optimize=True), 0, 1e-5)

        A2 = Solve_Factor(mask, [A1, A2, A3], np.einsum('ijk,ir,kr->jr',T_,A1,A3,optimize=True), 1, 1e-5)

        A3 = Solve_Factor(mask, [A1, A2, A3], np.einsum('ijk,ir,jr->kr',T_,A1,A2,optimize=True), 2, 1e-5)

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
        mask = np.random.random(T.shape) > args.partial

        A1, A2, A3, _, _, loss, rec = CPC(mask, T, R=20, iteration=200)
        # pickle.dump([A1, A2, A3, loss, rec], open('../result_final_google/CPC_{}.pkl'.format(i), 'wb'))
