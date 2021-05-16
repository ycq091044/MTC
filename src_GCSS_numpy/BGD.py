
import pickle
import numpy as np
from scipy import linalg as la
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--partial', type=float, default=0.95, help="partial observation")
args = parser.parse_args()

def GD(iteration, X, O1, O2, mask, R):
    # time_std = [[time.time(),0]] ###########################
        
    new_d1, d2, d3 = O1.shape
    d1, new_d2, d3 = O2.shape
    A1 = np.random.random((d1, R))
    A2 = np.random.random((d2, R))
    A3 = np.random.random((d3, R))
    tmp1 = np.random.random((new_d1, R))
    tmp2 = np.random.random((new_d2, R))
    
    # data prepare
    X_1 = X.reshape(d1, -1)
    X_2 = np.transpose(X, (1,0,2)).reshape(d2, -1)
    X_3 = np.transpose(X, (2,0,1)).reshape(d3, -1)
    M1 = mask.reshape(d1, -1)
    M2 = np.transpose(mask, (1,0,2)).reshape(d2, -1)
    M3 = np.transpose(mask, (2,0,1)).reshape(d3, -1)
    
    loss_list, rec_list = [], []

    def obj_func(X, coord, grad):
        X[coord] -= grad
        A1, A2, A3, tmp1, tmp2 = X

        rec = np.einsum('ir,jr,kr->ijk', A1, A2, A3, optimize=True)
        LOSS1 = la.norm(O1 - np.einsum('ir,jr,kr->ijk',tmp1,A2,A3,optimize=True)) / la.norm(O1)
        LOSS2 = la.norm(O2 - np.einsum('ir,jr,kr->ijk',A1,tmp2,A3,optimize=True)) / la.norm(O2)
        LOSS3 = la.norm(mask * (rec - T)) / la.norm(mask * T)

        LOSS = Lambda * LOSS1 ** 2 + Lambda * LOSS2 ** 2 + LOSS3 ** 2
        
        return LOSS

    def grad_tmp1(Lambda):
        return - np.einsum('ijk,jr,kr->ir',O1,A2,A3,optimize=True) + np.einsum('ar,br,bj,cr,cj->aj',tmp1,A2,A2,A3,A3,optimize=True)
        # return - O1_1 @ la.khatri_rao(A2, A3) + tmp1 @ ((A2.T @ A2) * (A3.T @ A3))
    
    def grad_tmp2(Lambda):
        return np.einsum('ijk,ir,kr->jr',O2,A1,A3,optimize=True) + np.einsum('ar,br,bj,cr,cj->aj',tmp2,A1,A1,A3,A3,optimize=True)
        # return - O2_2 @ la.khatri_rao(A1, A3) + tmp2 @ ((A1.T @ A1) * (A3.T @ A3))

    def grad_A1(Lambda):
        return - Lambda * np.einsum('ijk,jr,kr->ir',O2,tmp2,A3,optimize=True) + Lambda *  np.einsum('ar,br,bj,cr,cj->aj',A1,tmp2,tmp2,A3,A3,optimize=True)\
                            + (M1 * (A1 @ la.khatri_rao(A2, A3).T - X_1)) @ la.khatri_rao(A2, A3)
        # return - Lambda * O2_1 @ la.khatri_rao(tmp2, A3) + Lambda * A1 @ ((tmp2.T @ tmp2) * (A3.T @ A3)) \
        #                     + (M1 * (A1 @ la.khatri_rao(A2, A3).T - X_1)) @ la.khatri_rao(A2, A3)

    def grad_A2(Lambda):
        return - Lambda * np.einsum('ijk,ir,kr->jr',O1,tmp1,A3,optimize=True) + Lambda *  np.einsum('ar,br,bj,cr,cj->aj',A2,tmp1,tmp1,A3,A3,optimize=True)\
                            + (M2 * (A2 @ la.khatri_rao(A1, A3).T - X_2)) @ la.khatri_rao(A1, A3)
        # return - Lambda * O1_2 @ la.khatri_rao(tmp1, A3) + Lambda * A2 @ ((tmp1.T @ tmp1) * (A3.T @ A3)) \
                            # + (M2 * (A2 @ la.khatri_rao(A1, A3).T - X_2)) @ la.khatri_rao(A1, A3)

    def grad_A3(Lambda):
        return - Lambda * np.einsum('ijk,ir,jr->kr',O1,tmp1,A2,optimize=True) + Lambda *  np.einsum('ar,br,bj,cr,cj->aj',A3,tmp1,tmp1,A2,A2,optimize=True)\
                - Lambda * np.einsum('ijk,ir,jr->kr',O2,A1,tmp2,optimize=True) + Lambda *  np.einsum('ar,br,bj,cr,cj->aj',A3,A1,A1,tmp2,tmp2,optimize=True)\
                           + (M3 * (A3 @ la.khatri_rao(A1, A2).T - X_3)) @ TMP
        # return - Lambda * O1_3 @ TMP3 + Lambda * A3 @ ((tmp1.T @ tmp1) * (A2.T @ A2)) \
        #                     - Lambda * O2_3 @ TMP2 + Lambda * A3 @ ((A1.T @ A1) * (tmp2.T @ tmp2)) \
        #                     + (M3 * (A3 @ la.khatri_rao(A1, A2).T - X_3)) @ TMP

    def line_search(obj_func, X, coord, grad, alpha = [1e-8, 5e-8], visit = [-1, -1], depth=3):

        if depth == 0:
            return alpha[0]

        result = []
        for i, j in zip(alpha, visit):
            if j < 0:
                result.append(obj_func(X, coord, i * grad))
            else:
                result.append(j)
        
        mid = sum(alpha) / 2

        if result[0] >= result[1]:
            return line_search(obj_func, X, coord, grad, alpha = [mid, alpha[1]], visit = [-1, result[1]], depth=depth-1)
        else:
            return line_search(obj_func, X, coord, grad, alpha = [alpha[0], mid], visit = [result[0], -1], depth=depth-1)

    tic = time.time()
    for i in range(iteration):

        # ###########################
        # time_std.append([time.time(), time.time() - time_std[-1][0]])
        # if i > 0:
        #     estimate = sum([item[1] for item in time_std[2:]]) / i * 200
        #     std = np.std([item[1] for item in time_std[2:]])
        #     print (i, '{:.6} $\pm$ {:.4}'.format(estimate + time_std[1][1], std * 200))
        # ###########################

        Lambda = 1. * np.exp(-i/20)

        GRAD = grad_tmp1(Lambda)
        alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 3, GRAD)
        tmp1 -= alpha * GRAD
    
        GRAD = grad_tmp2(Lambda)
        alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 4, GRAD)
        tmp2 -= alpha * GRAD

        GRAD = grad_A1(Lambda)
        alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 0, GRAD)
        A1 -= alpha * GRAD

        GRAD = grad_A2(Lambda)
        alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 1, GRAD)
        A2 -= alpha * GRAD

        # TMP = la.khatri_rao(A1, A2)
        # TMP2 = la.khatri_rao(A1, tmp2)
        # TMP3 = la.khatri_rao(tmp1, A2)
        
        GRAD = grad_A3(Lambda)
        alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 2, GRAD)
        A3 -= alpha * GRAD
        
        rec = np.einsum('ir,jr,kr->ijk', A1, A2, A3, optimize=True)
        LOSS1 = la.norm(O1 - np.einsum('ir,jr,kr->ijk',tmp1,A2,A3,optimize=True)) / la.norm(O1)
        LOSS2 = la.norm(O2 - np.einsum('ir,jr,kr->ijk',A1,tmp2,A3,optimize=True)) / la.norm(O2)
        LOSS3 = la.norm(mask * (rec - T)) / la.norm(mask * T)

        LOSS = Lambda * LOSS1 ** 2 + Lambda * LOSS2 ** 2 + LOSS3 ** 2
        RecLOSS = 1 - la.norm(rec - T) / la.norm(T)
        print (i, LOSS, RecLOSS, 'time span: ', time.time() - tic)
        tic = time.time()

        loss_list.append(LOSS); rec_list.append(RecLOSS)

        # rescaling
        norm1 = (A1**2).sum(axis=0) ** .5
        norm2 = (A2**2).sum(axis=0) ** .5
        norm3 = (A3**2).sum(axis=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)

        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3
        tmp1 *= norm / norm1 
        tmp2 *= norm / norm2
        
    return A1, A2, A3, tmp1, tmp2, loss_list, rec_list


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

    T = np.transpose(T, (0,2,1))
    O1 = np.transpose(O1, (0,2,1))
    O2 = np.transpose(O2, (0,2,1))

    for j in range(1):
        np.random.seed(j)
        mask = np.random.random(T.shape) > args.partial

        A1, A2, A3, tmp1, tmp2, loss, rec = GD(400, T, O1, O2, mask, R=20)
        # pickle.dump([A1, A2, A3, tmp1, tmp2, loss, rec], open('result_google/BGD_{}_{}.pkl'.format(args.partial, j), 'wb'))
