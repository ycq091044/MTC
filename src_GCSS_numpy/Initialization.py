from scipy.optimize import nnls
import numpy as np
from scipy import linalg as la
import pickle
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--partial', type=float, default=0.95, help="partial observation")
parser.add_argument('--init', type=str, default='MTC', help="initialization method")
args = parser.parse_args()


def jacobi(u, A, b, mode):

    D = np.diag(A) + 1e-3
    R = A - np.diag(D)
    Z = np.diag(D**(-1)) @ b
    B = - np.diag(D**(-1)) @ R
    w = .2e-1
    for i in range(15):
        u = (1-w) * u + w * (Z + B @ u)
    
    return u

def optimize(u, weight, A, b, iteration):
    
    A_ = np.zeros(A[0].shape); b_ = np.zeros(b[0].shape)
    for i in range(len(A)):
        A_ += weight[i] * A[i]
        b_ += weight[i] * b[i]

    if iteration < 5:
        u1 = jacobi(u, A_, b_, 0)
    else:
        # u1 = la.solve(A_ + np.eye(A_.shape[1]) * 1e-8, b_)
        L = la.cholesky(A_ + np.eye(A_.shape[1]) * 1e-8)
        y = la.solve_triangular(L.T, b_, lower=True)
        u1 = la.solve_triangular(L, y, lower=False)
    return u1

def CPD(T, R, iteration, A1, A2, A3):

    def optimize2(u, A, B):
        from numpy import linalg as la
        # u1 = la.solve(A + np.eye(A.shape[1]) * 1e-8, B)
        L = la.cholesky(A + np.eye(A.shape[1]) * 1e-8)
        y = la.solve_triangular(L.T, B, lower=True)
        u1 = la.solve_triangular(L, y, lower=False)
        return u1

    d1, d2, d3 = T.shape
    
    tic = time.time()
    for i in range(iteration):
        
        A1 = optimize2(A1.T, (A3.T@A3)*(A2.T@A2), np.einsum('ijk,jr,kr->ri',T,A2,A3,optimize=True)).T

        A2 = optimize2(A2.T, (A1.T@A1)*(A3.T@A3), np.einsum('ijk,ir,kr->rj',T,A1,A3,optimize=True)).T
        
        A3 = optimize2(A3.T, (A1.T@A1)*(A2.T@A2), np.einsum('ijk,ir,jr->rk',T,A1,A2,optimize=True)).T
        
        recLOSS = la.norm(np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True))

        norm1 = (A1**2).sum(axis=0) ** .5
        norm2 = (A2**2).sum(axis=0) ** .5
        norm3 = (A3**2).sum(axis=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)

        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3

        print (i, recLOSS / la.norm(T), 'time span: ', time.time() - tic)
        tic = time.time() 
    return A1, A2, A3


def HOSVD(T, O1, O2, R):
    d1, d2, d3 = T.shape
    new_d1, _, _ = O1.shape
    _, _, new_d3 = O2.shape

    A1 = np.random.random((d1, R))
    A2 = np.random.random((d2, R))
    A3 = np.random.random((d3, R))
    tmp1 = np.random.random((new_d1, R))
    tmp3 = np.random.random((new_d3, R))

    """
    We implement the MTTKRP by both np.einsum and manually calculation,
    Remember when use np.einsum, the argument "optimize=True"
    """

    # HOOI
    for i in range(100):
        print (i)

        u, s, v = la.svd(np.einsum('ijk,jr,kr->ir',O1,A2,A3,optimize=True))
        # u, s, v = la.svd(O1_1 @ la.khatri_rao(A2, A3))
        tmp1 = u[:, :R]

        u, s, v = la.svd(np.einsum('ijk,ir,jr->kr',O1,tmp1,A2,optimize=True))
        # u, s, v = la.svd(O1_3 @ la.khatri_rao(tmp1, A2))
        A3 = u[:, :R]

        u, s, v = la.svd(np.einsum('ijk,ir,kr->jr',O1,tmp1,A3,optimize=True))
        # u, s, v = la.svd(O1_2 @ la.khatri_rao(tmp1, A3))
        A2 = u[:, :R]

        u, s, v = la.svd(np.einsum('ijk,jr,kr->ir',O2,A2,tmp3,optimize=True))
        # u, s, v = la.svd(O2_1 @ la.khatri_rao(tmp2, A3))
        A1 = u[:, :R]

        u, s, v = la.svd(np.einsum('ijk,ir,jr->kr',O2,A1,A2,optimize=True))
        # u, s, v = la.svd(O2_3 @ la.khatri_rao(A1, tmp2))
        tmp3 = u[:, :R]

        u, s, v = la.svd(np.einsum('ijk,ir,kr->jr',O2,A1,tmp3,optimize=True))
        # u, s, v = la.svd(O2_2 @ la.khatri_rao(A1, A3))
        A2 = u[:, :R]


    # CPD on Core
    Core = np.einsum('ijk,ia,jb,kc->abc',O1,tmp1,A2,A3,optimize=True)
    # after_tmp1 = (tmp1.T @ O1_1).reshape(R, d2, d3)
    # after_A2 = (A2.T @ np.transpose(after_tmp1, (1, 0, 2)).reshape(d2, -1)).reshape(R,R,d3)
    # after_A3 = (A3.T @ np.transpose(after_A2, (2, 1, 0)).reshape(d3, -1)).reshape(R,R,R)
    # Core = np.transpose(after_A3, (1,2,0))

    core1 = np.random.random((R,R))
    core2 = np.random.random((R,R))
    core3 = np.random.random((R,R))
    core1, core2, core3 = CPD(Core, R, 50, core1, core2, core3)

    # combine factors
    A1 = A1 @ core1
    tmp1 = tmp1 @ core1
    A2 = A2 @ core2
    A3 = A3 @ core3
    tmp3 = tmp3 @ core3

    return A1, A2, A3, tmp1, tmp3

def TendiB(T, O1, O2, R):
    d1, d2, d3 = T.shape
    new_d1, _, _ = O1.shape
    _, _, new_d3 = O2.shape

    A1 = np.random.random((d1, R))
    A2 = np.random.random((d2, R))
    A3 = np.random.random((d3, R))
    tmp1 = np.random.random((new_d1, R))
    tmp3 = np.random.random((new_d3, R))

    tmp1, A2, A3 = CPD(O1, R, 60, tmp1, A2, A3)
    A1, A2, tmp3 = CPD(O2, R, 60, A1, A2, tmp3)

    return A1, A2, A3, tmp1, tmp3

def myalgo(mask, T, O1, O2, R, iteration):

    # set up
    d1, d2, d3 = T.shape
    new_d1, _, _ = O1.shape
    _, _, new_d3 = O2.shape
    
    def interpolation(A, A_):
        d, _ = A.shape
        tmp = np.zeros(A.shape)
        if (d % 2 == 0):
            tmp[::2, :] = A_
            tmp[1:-1:2, :] = (A_[:-1, :] + A_[1:, :]) / 2
            tmp[-1, :] = A_[-1, :] / 2
        if (d % 2 == 1):
            tmp[::2, :] = A_
            tmp[1::2, :] = (A_[:-1, :] + A_[1:, :]) / 2
        return tmp

    if args.init == 'HOSVD':
        A1, A2, A3, tmp1, tmp3 = HOSVD(T, O1, O2, R)

    elif args.init == 'TendiB':
        A1, A2, A3, tmp1, tmp3 = TendiB(T, O1, O2, R)
    
    elif args.init == 'MRTL':

        if max(d1, d2, d3, new_d1, new_d3) < 2 * R:

            A1 = np.random.random((d1,R))
            A2 = np.random.random((d2,R))
            A3 = np.random.random((d3,R))
            tmp1 = np.random.random((new_d1,R))
            tmp3 = np.random.random((new_d3,R))

        ######## the recursion #######
        else:
            # subsampling
            mask_ = mask.copy()
            T_ = T.copy()
            O1_ = O1.copy()
            O2_ = O2.copy()
            if d1 >= 2 * R:
                mask_ = mask_[::2, :, :]; T_ = T_[::2, :, :]; O2_ = O2_[::2, :, :]
            if d2 >= 2 * R:
                mask_ = mask_[:, ::2, :]; T_ = T_[:, ::2, :]; O1_ = O1_[:, ::2, :]; O2_ = O2_[:, ::2, :]
            if d3 >= 2 * R:
                mask_ = mask_[:, :, ::2]; T_ = T_[:, :, ::2]; O1_ = O1_[:, :, ::2]
            if new_d1 >= 2 * R:
                O1_ = O1_[::2, :, :]
            if new_d3 >= 2 * R:
                O2_ = O2_[:, :, ::2]

            # solve low-resolution problem
            A1_, A2_, A3_, tmp1_, tmp3_, _, _ = myalgo(mask_, T_, O1_, O2_, R=R, iteration=10)

            # interpolate
            if d1 >= 2 * R:
                A1 = np.random.random((d1,R)); A1 = interpolation(A1, A1_)
            else:
                A1 = A1_
            if d2 >= 2 * R:
                A2 = np.random.random((d2,R)); A2 = interpolation(A2, A2_)
            else:
                A2 = A2_
            if d3 >= 2 * R:
                A3 = np.random.random((d3,R)); A3 = interpolation(A3, A3_)
            else:
                A3 = A3_
            if new_d1 >= 2 * R:
                tmp1 = np.random.random((new_d1,R)); tmp1 = interpolation(tmp1, tmp1_)
            else:
                tmp1 = tmp1_
            if new_d3 >= 2 * R:
                tmp3 = np.random.random((new_d3,R)); tmp3 = interpolation(tmp3, tmp3_)
            else:
                tmp3 = tmp3_

    elif args.init == 'MTC':

        # high-resolution
        if max(d1, d2, d3, new_d1, new_d3) < 2 * R:

            A1 = np.random.random((d1,R))
            A2 = np.random.random((d2,R))
            A3 = np.random.random((d3,R))
            tmp1 = np.random.random((new_d1,R))
            tmp3 = np.random.random((new_d3,R))

        ######## the recursion #######
        else:
            # subsampling
            mask_ = mask.copy()
            T_ = T.copy()
            O1_ = O1.copy()
            O2_ = O2.copy()
            if d1 >= 2 * R:
                mask_ = mask_[:d1//2, :, :]; T_ = T_[:d1//2, :, :]; O2_ = O2_[:d1//2, :, :]
            if d2 >= 2 * R:
                mask_ = mask_[:, :d2//2, :]; T_ = T_[:, :d2//2, :]; O1_ = O1_[:, :d2//2, :]; O2_ = O2_[:, :d2//2, :]
            if d3 >= 2 * R:
                mask_ = mask_[:, :, ::2]; T_ = T_[:, :, ::2]; O1_ = O1_[:, :, ::2]
            if new_d1 >= 2 * R:
                O1_ = O1_[:new_d1//2, :, :]
            if new_d3 >= 2 * R:
                O2_ = O2_[:, :, ::2]

            # solve low-resolution problem
            A1_, A2_, A3_, tmp1_, tmp3_, _, _ = myalgo(mask_, T_, O1_, O2_, R=R, iteration=10)

            # interpolate
            if d1 >= 2 * R:
                A1 = np.random.random((d1,R)); A1[:d1//2,:] = A1_
            else:
                A1 = A1_
            if d2 >= 2 * R:
                A2 = np.random.random((d2,R)); A2[:d2//2,:] = A2_
            else:
                A2 = A2_
            if d3 >= 2 * R:
                A3 = np.random.random((d3,R)); A3 = interpolation(A3, A3_)
            else:
                A3 = A3_
            if new_d1 >= 2 * R:
                tmp1 = np.random.random((new_d1,R)); tmp1[:new_d1//2,:] = tmp1_
            else:
                tmp1 = tmp1_
            if new_d3 >= 2 * R:
                tmp3 = np.random.random((new_d3,R)); tmp3 = interpolation(tmp3, tmp3_)
            else:
                tmp3 = tmp3_

    elif args.init == 'No':
        A1 = np.random.random((d1,R))
        A2 = np.random.random((d2,R))
        A3 = np.random.random((d3,R))
        tmp1 = np.random.random((new_d1,R))
        tmp3 = np.random.random((new_d3,R))


    print (d1, d2, d3, new_d1, new_d3, O1.shape, O2.shape)

    loss_list, rec_list = [], []
    
    tic = time.time()
    rec = np.zeros(mask.shape)
    for i in range(iteration): 
        LAMBDA = 1. * np.exp(-i / 20)

        # coupled ALS
        T_ = mask * T + (1-mask) * rec
        
        tmp1 = optimize(tmp1.T, [1.], [(A3.T@A3)*(A2.T@A2)], [np.einsum('ijk,jr,kr->ri',O1,A2,A3,optimize=True)], i).T

        tmp3 = optimize(tmp3.T, [1.], [(A2.T@A2)*(A1.T@A1)], [np.einsum('ijk,ir,jr->rk',O2,A1,A2,optimize=True)], i).T
        
        A1 = 1. * optimize(A1.T, [LAMBDA, 1.], [(tmp3.T@tmp3)*(A2.T@A2), (A3.T@A3)*(A2.T@A2)], \
                            [np.einsum('ijk,jr,kr->ri',O2,A2,tmp3,optimize=True), np.einsum('ijk,jr,kr->ri',T_,A2,A3,optimize=True)], i).T
        
        A3 = 1. * optimize(A3.T, [LAMBDA, 1.], [(A2.T@A2)*(tmp1.T@tmp1), (A2.T@A2)*(A1.T@A1)], \
                            [np.einsum('ijk,ir,jr->rk',O1,tmp1,A2,optimize=True), np.einsum('ijk,ir,jr->rk',T_,A1,A2,optimize=True)], i).T

        A2 = optimize(A2.T, [1., LAMBDA, LAMBDA], [(A3.T@A3)*(A1.T@A1), (A1.T@A1)*(tmp3.T@tmp3), (A3.T@A3)*(tmp1.T@tmp1)], \
                        [np.einsum('ijk,ir,kr->rj',T_,A1,A3,optimize=True), np.einsum('ijk,ir,kr->rj',O2,A1,tmp3,optimize=True), \
                        np.einsum('ijk,ir,kr->rj',O1,tmp1,A3,optimize=True)], i).T
    
        
        # output
        rec = np.einsum('ir,jr,kr->ijk', A1, A2, A3, optimize=True)
        LOSS1 = la.norm(O1 - np.einsum('ir,jr,kr->ijk',tmp1,A2,A3,optimize=True)) / la.norm(O1)
        LOSS2 = la.norm(O2 - np.einsum('ir,jr,kr->ijk',A1,A2,tmp3,optimize=True)) / la.norm(O2)
        LOSS3 = la.norm(mask * (rec - T)) / la.norm(mask * T)

        LOSS = LAMBDA * LOSS1 ** 2 + LAMBDA * LOSS2 ** 2 + LOSS3 ** 2
        recLOSS = 1 - la.norm(rec - T) / la.norm(T)

        print (i, LOSS, recLOSS, 'time span: ', time.time() - tic)
        # print (i, time.time() - tic)
        tic = time.time()
        loss_list.append(LOSS)
        rec_list.append(recLOSS)

        norm1 = (A1**2).sum(axis=0) ** .5
        norm2 = (A2**2).sum(axis=0) ** .5
        norm3 = (A3**2).sum(axis=0) ** .5
        # norm4 = (tmp1**2).sum(axis=0) ** .5
        # norm5 = (tmp2**2).sum(axis=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)
        
        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3
        tmp1 *= norm / norm1
        tmp3 *= norm / norm3
        
    return A1, A2, A3, tmp1, tmp3, loss_list, rec_list

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

    if args.init == 'MTC':
        """
        First, sort on the categorical mode.
        The code provides a more efficient way than paper does.
        We directly sort in descent order at the beginning and pick [:d//2] during subsampling
        """

        import scipy.stats as ss
        O1_binary = O1 > 0

        # rank the coarse first mode
        rankIdx1 = ss.rankdata(O1_binary.sum(axis=1).sum(axis=1), method='ordinal')
        rankIdx1 = len(rankIdx1) - rankIdx1   

        # rank the fine second mode
        rankIdx2 = ss.rankdata(O1_binary.sum(axis=0).sum(axis=1), method='ordinal')
        rankIdx2 = len(rankIdx2) - rankIdx2

        O2_binary = O2 > 0

        # rank the fine first mode
        rankIdx3 = ss.rankdata(O2_binary.sum(axis=1).sum(axis=1), method='ordinal')
        rankIdx3 = len(rankIdx3) - rankIdx3   

        # apply ranking to all related mode
        T = T[rankIdx3, :, :][:, rankIdx2, :]
        O1 = O1[rankIdx1, :, :][:, rankIdx2, :]
        O2 = O2[rankIdx3, :, :][:, rankIdx2, :]

    for i in range(3):
        np.random.seed(i)
        mask = np.random.random(T.shape) > args.partial

        A1, A2, A3, _, _, loss, rec = myalgo(mask, T, O1, O2, R=20, iteration=30)
        # pickle.dump([A1, A2, A3, loss, rec], open('result_google/{}_Initialization_{}.pkl'.format(args.init,i), 'wb'))
