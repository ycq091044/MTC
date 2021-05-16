import numpy as np
import pickle
import time
import argparse
import torch
import os

def optimize(A, B):
    """
    least square optimizer:
        A @ u.T = B
    """
    # L = torch.cholesky(A + torch.eye(A.shape[1]).to(device) * 1e-8)
    # y = torch.triangular_solve(B, L.T, upper=True)[0]
    # u = torch.triangular_solve(y, L, upper=False)[0]
    u, _ = torch.solve(B, A)
    return u


def OracleCPD(T, R, iteration, device, A1=None, A2=None, A3=None):
    """
    OracleCPD model:
        we call the <optimize> for each least square sub-problem
    """

    # preparation
    loss_list, rec_list = [], []
    d1, d2, d3 = T.shape

    if A1 is None:
        A1 = torch.randn(d1, R).to(device)
        A2 = torch.randn(d2, R).to(device)
        A3 = torch.randn(d3, R).to(device)
    
    # ALS 
    tic = time.time()
    for i in range(iteration):

        # sub-iteration
        A1 = optimize((A3.T@A3)*(A2.T@A2), torch.einsum('ijk,jr,kr->ri',T,A2,A3)).T
        A2 = optimize((A1.T@A1)*(A3.T@A3), torch.einsum('ijk,ir,kr->rj',T,A1,A3)).T
        A3 = optimize((A1.T@A1)*(A2.T@A2), torch.einsum('ijk,ir,jr->rk',T,A1,A2)).T

        # loss 
        rec = torch.einsum('ir,jr,kr->ijk',A1,A2,A3)
        LOSS = torch.norm(rec - T)/ torch.norm(T)
        recLOSS = 1 - torch.norm(rec - T) / torch.norm(T)
        if i % 20 == 0:
            print ('{}/{}'.format(i, iteration), 'loss:', LOSS.item(), 'rec:', recLOSS.item(), 'time span:', time.time() - tic)

        # collect loss
        tic = time.time() 
        loss_list.append(LOSS)
        rec_list.append(recLOSS)

    return A1, A2, A3, loss_list, rec_list


def maskOptimizer(Omega, A, RHS, num, reg, device):
    """
    masked least square optimizer:
        A @ u.T = Omega * RHS
        number: which factor
        reg: 2-norm regulizer
    """
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
    P = torch.einsum(einstr,*lst_mat)
    o = torch.zeros_like(RHS)
    for j in range(A[num].shape[0]):
        o[j,:] = torch.inverse(P[j]+reg*torch.eye(R).to(device)) @ RHS[j,:]
    return o


def CPC_ALS(T, mask, R, iteration, device):
    """
    CPC-ALS:
        call <maskOptimizer> for the sub-iteration
    """

    # preparation
    loss_list, rec_list = [], []
    d1, d2, d3 = T.shape

    A1 = torch.FloatTensor(np.random.random((d1, R))).to(device)
    A2 = torch.FloatTensor(np.random.random((d2, R))).to(device)
    A3 = torch.FloatTensor(np.random.random((d3, R))).to(device)

    T_ = mask * T

    # ALS
    tic = time.time()
    for i in range(5):

        # sub-iteration
        A1 = maskOptimizer(mask, [A1, A2, A3], torch.einsum('ijk,jr,kr->ir',T_,A2,A3), 0, 1e-5, device)
        A2 = maskOptimizer(mask, [A1, A2, A3], torch.einsum('ijk,ir,kr->jr',T_,A1,A3), 1, 1e-5, device)
        A3 = maskOptimizer(mask, [A1, A2, A3], torch.einsum('ijk,ir,jr->kr',T_,A1,A2), 2, 1e-5, device)

        # loss 
        rec = torch.einsum('ir,jr,kr->ijk',A1,A2,A3)
        LOSS = torch.norm(mask * (rec - T))/ torch.norm(mask * T)
        recLOSS = 1 - torch.norm(rec - T) / torch.norm(T)
        print ('{}/{}'.format(i, iteration), 'loss:', LOSS.item(), 'rec:', recLOSS.item(), 'time span:', time.time() - tic)
        
        # collect loss
        tic = time.time() 
        loss_list.append(LOSS)
        rec_list.append(recLOSS)

    return A1, A2, A3, loss_list, rec_list


def BGD(X, O1, O2, mask, R, iteration, device):
    """
    block gradient descent (BGD):
        notation mapping (to the paper):
            A1 -> U, A2 -> V, A3 -> W, tmp1 -> Q1, tmp2 -> Q2
            O1 -> C1, O2 -> C2
    """

    # preparation
    new_d1, d2, d3 = O1.shape
    d1, new_d2, d3 = O2.shape
    A1 = torch.FloatTensor(np.random.random((d1,R))).to(device)
    A2 = torch.FloatTensor(np.random.random((d2,R))).to(device)
    A3 = torch.FloatTensor(np.random.random((d3,R))).to(device)
    tmp1 = torch.FloatTensor(np.random.random((new_d1,R))).to(device)
    tmp2 = torch.FloatTensor(np.random.random((new_d2,R))).to(device)
    loss_list, rec_list = [], []
    
    # intermediate constant data (for fast gradient computation)
    X_1 = X.reshape(d1, -1)
    X_2 = X.permute(1,0,2).reshape(d2, -1)
    X_3 = X.permute(2,0,1).reshape(d3, -1)
    M1 = mask.reshape(d1, -1)
    M2 = mask.permute(1,0,2).reshape(d2, -1)
    M3 = mask.permute(2,0,1).reshape(d3, -1)

    def obj_func(X, coord, grad):
        """
        X = (A1, A2, A3, tmp1, tmp2)
        coord decides which one get the gradient
        """
        X[coord] -= grad
        A1, A2, A3, tmp1, tmp2 = X

        rec = torch.einsum('ir,jr,kr->ijk', A1, A2, A3)
        LOSS1 = torch.norm(O1 - torch.einsum('ir,jr,kr->ijk',tmp1,A2,A3)) / torch.norm(O1)
        LOSS2 = torch.norm(O2 - torch.einsum('ir,jr,kr->ijk',A1,tmp2,A3)) / torch.norm(O2)
        LOSS3 = torch.norm(mask * (rec - X)) / torch.norm(mask * X)
        LOSS = Lambda * LOSS1 ** 2 + Lambda * LOSS2 ** 2 + LOSS3 ** 2
        
        return LOSS

    # Kharti-rao product
    def kr(A, B):
        return torch.cat([torch.outer(A[:,i], B[:,i]).view(-1, 1) for i in range(A.shape[1])], dim=1)

    def grad_tmp1(Lambda):
        return - torch.einsum('ijk,jr,kr->ir',O1,A2,A3) + torch.einsum('ar,br,bj,cr,cj->aj',tmp1,A2,A2,A3,A3)
    
    def grad_tmp2(Lambda):
        return - torch.einsum('ijk,ir,kr->jr',O2,A1,A3) + torch.einsum('ar,br,bj,cr,cj->aj',tmp2,A1,A1,A3,A3)

    def grad_A1(Lambda):
        return - Lambda * torch.einsum('ijk,jr,kr->ir',O2,tmp2,A3) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A1,tmp2,tmp2,A3,A3)\
                            + (M1 * (A1 @ kr(A2, A3).T - X_1)) @ kr(A2, A3)
    def grad_A2(Lambda):
        return - Lambda * torch.einsum('ijk,ir,kr->jr',O1,tmp1,A3) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A2,tmp1,tmp1,A3,A3)\
                            + (M2 * (A2 @ kr(A1, A3).T - X_2)) @ kr(A1, A3)
    def grad_A3(Lambda):
        return - Lambda * torch.einsum('ijk,ir,jr->kr',O1,tmp1,A2) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A3,tmp1,tmp1,A2,A2)\
                - Lambda * torch.einsum('ijk,ir,jr->kr',O2,A1,tmp2) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A3,A1,A1,tmp2,tmp2)\
                           + (M3 * (A3 @ kr(A1, A2).T - X_3)) @ kr(A1, A2)

    def line_search(obj_func, X, coord, grad, alpha=[1e-9, 3e-7], visit=[-1, -1], depth=3):

        if depth == 0:
            return alpha[0]

        result = []
        for i, j in zip(alpha, visit):
            result.append(obj_func(X, coord, i * grad)) if j < 0 else result.append(j)
        mid = sum(alpha) / 2

        if result[0] >= result[1]:
            return line_search(obj_func, X, coord, grad, alpha=[mid, alpha[1]], visit=[-1, result[1]], depth=depth-1)
        else:
            return line_search(obj_func, X, coord, grad, alpha=[alpha[0], mid], visit=[result[0], -1], depth=depth-1)


    alpha = 1e-8
    
    # start gradient descent
    tic = time.time()
    for i in range(iteration):
        Lambda = max(1. * np.exp(-i / 50), 1e-3)

        GRAD = grad_tmp1(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 3, GRAD)
        tmp1 -= alpha * GRAD

        GRAD = grad_tmp2(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 4, GRAD)
        tmp2 -= alpha * GRAD

        GRAD = grad_A1(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 0, GRAD)
        A1 -= alpha * GRAD

        GRAD = grad_A2(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 1, GRAD)
        A2 -= alpha * GRAD
        
        GRAD = grad_A3(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 2, GRAD)
        A3 -= alpha * GRAD
        
        # loss
        rec = torch.einsum('ir,jr,kr->ijk', A1, A2, A3)
        LOSS1 = torch.norm(O1 - torch.einsum('ir,jr,kr->ijk',tmp1,A2,A3)) / torch.norm(O1)
        LOSS2 = torch.norm(O2 - torch.einsum('ir,jr,kr->ijk',A1,tmp2,A3)) / torch.norm(O2)
        LOSS3 = torch.norm(mask * (rec - X)) / torch.norm(mask * X)
        LOSS = Lambda * LOSS1 ** 2 + Lambda * LOSS2 ** 2 + LOSS3 ** 2
        RecLOSS = 1 - torch.norm(rec - X) / torch.norm(X)
        if i % 20 == 0:
            print ('{}/{}'.format(i, iteration), 'loss:', LOSS.item(), 'rec:', RecLOSS.item(), 'time span: ', time.time() - tic)

        # collect loss
        tic = time.time()

        loss_list.append(LOSS); 
        rec_list.append(RecLOSS)

        # rescaling
        norm1 = (A1**2).sum(dim=0) ** .5
        norm2 = (A2**2).sum(dim=0) ** .5
        norm3 = (A3**2).sum(dim=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)

        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3
        tmp1 *= norm / norm1 
        tmp2 *= norm / norm2
        
    return A1, A2, A3, tmp1, tmp2, loss_list, rec_list


def CMTF(X, O1, O2, mask, R, iteration, device):
    """
    Coupled Matrix Tensor Factorization (CMTF):
        notation mapping (to the paper):
            A1 -> U, A2 -> V, A3 -> W, tmp1 -> Q1, tmp2 -> Q2
            O1 -> C1, O2 -> C2
    """

    # preparation
    new_d1, d2, d3 = O1.shape
    d1, new_d2, d3 = O2.shape
    A1 = torch.FloatTensor(np.random.random((d1,R))).to(device)
    A2 = torch.FloatTensor(np.random.random((d2,R))).to(device)
    A3 = torch.FloatTensor(np.random.random((d3,R))).to(device)
    tmp1 = torch.FloatTensor(np.random.random((new_d1,R))).to(device)
    tmp2 = torch.FloatTensor(np.random.random((new_d2,R))).to(device)
    loss_list, rec_list = [], []
    
    # intermediate constant data (for fast gradient computation)
    X_1 = X.reshape(d1, -1)
    X_2 = X.permute(1,0,2).reshape(d2, -1)
    X_3 = X.permute(2,0,1).reshape(d3, -1)
    M1 = mask.reshape(d1, -1)
    M2 = mask.permute(1,0,2).reshape(d2, -1)
    M3 = mask.permute(2,0,1).reshape(d3, -1)

    def obj_func(X, coord, grad):
        """
        X = (A1, A2, A3, tmp1, tmp2)
        coord decides which one get the gradient
        """
        X[coord] -= grad
        A1, A2, A3, tmp1, tmp2 = X

        rec = torch.einsum('ir,jr,kr->ijk', A1, A2, A3)
        LOSS1 = torch.norm(O1 - torch.einsum('ir,jr,kr->ijk',tmp1,A2,A3)) / torch.norm(O1)
        LOSS2 = torch.norm(O2 - torch.einsum('ir,jr,kr->ijk',A1,tmp2,A3)) / torch.norm(O2)
        LOSS3 = torch.norm(mask * (rec - X)) / torch.norm(mask * X)
        LOSS = Lambda * LOSS1 ** 2 + Lambda * LOSS2 ** 2 + LOSS3 ** 2
        
        return LOSS

    # Kharti-rao product
    def kr(A, B):
        return torch.cat([torch.outer(A[:,i], B[:,i]).view(-1, 1) for i in range(A.shape[1])], dim=1)

    def grad_tmp1(Lambda):
        return - torch.einsum('ijk,jr,kr->ir',O1,A2,A3) + torch.einsum('ar,br,bj,cr,cj->aj',tmp1,A2,A2,A3,A3)
    
    def grad_tmp2(Lambda):
        return - torch.einsum('ijk,ir,kr->jr',O2,A1,A3) + torch.einsum('ar,br,bj,cr,cj->aj',tmp2,A1,A1,A3,A3)

    def grad_A1(Lambda):
        return - Lambda * torch.einsum('ijk,jr,kr->ir',O2,tmp2,A3) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A1,tmp2,tmp2,A3,A3)\
                            + (M1 * (A1 @ kr(A2, A3).T - X_1)) @ kr(A2, A3)
    def grad_A2(Lambda):
        return - Lambda * torch.einsum('ijk,ir,kr->jr',O1,tmp1,A3) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A2,tmp1,tmp1,A3,A3)\
                            + (M2 * (A2 @ kr(A1, A3).T - X_2)) @ kr(A1, A3)
    def grad_A3(Lambda):
        return - Lambda * torch.einsum('ijk,ir,jr->kr',O1,tmp1,A2) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A3,tmp1,tmp1,A2,A2)\
                - Lambda * torch.einsum('ijk,ir,jr->kr',O2,A1,tmp2) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A3,A1,A1,tmp2,tmp2)\
                           + (M3 * (A3 @ kr(A1, A2).T - X_3)) @ kr(A1, A2)

    def line_search(obj_func, X, coord, grad, alpha=[1e-9, 3e-7], visit=[-1, -1], depth=3):

        if depth == 0:
            return alpha[0]

        result = []
        for i, j in zip(alpha, visit):
            result.append(obj_func(X, coord, i * grad)) if j < 0 else result.append(j)
        mid = sum(alpha) / 2

        if result[0] >= result[1]:
            return line_search(obj_func, X, coord, grad, alpha=[mid, alpha[1]], visit=[-1, result[1]], depth=depth-1)
        else:
            return line_search(obj_func, X, coord, grad, alpha=[alpha[0], mid], visit=[result[0], -1], depth=depth-1)

    alpha = 2e-8

    # start gradient descent at once
    tic = time.time()
    for i in range(iteration):

        Lambda = max(1. * np.exp(-i / 50), 1e-3)

        GRAD_tmp1 = grad_tmp1(Lambda)
        GRAD_tmp2 = grad_tmp2(Lambda)
        GRAD_A1 = grad_A1(Lambda)
        GRAD_A2 = grad_A2(Lambda)
        GRAD_A3 = grad_A3(Lambda)

        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 3, GRAD_tmp1)
        tmp1 -= alpha * GRAD_tmp1
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 4, GRAD_tmp2)
        tmp2 -= alpha * GRAD_tmp2
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 0, GRAD_A1)
        A1 -= alpha * GRAD_A1
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 1, GRAD_A2)
        A2 -= alpha * GRAD_A2
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 2, GRAD_A3)
        A3 -= alpha * GRAD_A3
        
        # loss
        rec = torch.einsum('ir,jr,kr->ijk', A1, A2, A3)
        LOSS1 = torch.norm(O1 - torch.einsum('ir,jr,kr->ijk',tmp1,A2,A3)) / torch.norm(O1)
        LOSS2 = torch.norm(O2 - torch.einsum('ir,jr,kr->ijk',A1,tmp2,A3)) / torch.norm(O2)
        LOSS3 = torch.norm(mask * (rec - X)) / torch.norm(mask * X)
        LOSS = Lambda * LOSS1 ** 2 + Lambda * LOSS2 ** 2 + LOSS3 ** 2
        RecLOSS = 1 - torch.norm(rec - X) / torch.norm(X)
        if i % 20 == 0:
            print ('{}/{}'.format(i, iteration), 'loss:', LOSS.item(), 'rec:', RecLOSS.item(), 'time span: ', time.time() - tic)

        # collect loss   
        tic = time.time()
        loss_list.append(LOSS)
        rec_list.append(RecLOSS)

        # rescaling
        norm1 = (A1**2).sum(dim=0) ** .5
        norm2 = (A2**2).sum(dim=0) ** .5
        norm3 = (A3**2).sum(dim=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)

        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3
        tmp1 *= norm / norm1 
        tmp2 *= norm / norm2
        
    return A1, A2, A3, tmp1, tmp2, loss_list, rec_list


def PREMA_init(X, O1, O2, R, device):
    # preparation
    d1, d2, d3 = X.shape
    new_d1, _, _ = O1.shape
    _, new_d2, _ = O2.shape

    A1 = torch.FloatTensor(np.random.random((d1,R))).to(device)
    A2 = torch.FloatTensor(np.random.random((d2,R))).to(device)
    A3 = torch.FloatTensor(np.random.random((d3,R))).to(device)
    tmp1 = torch.FloatTensor(np.random.random((new_d1,R))).to(device)
    tmp2 = torch.FloatTensor(np.random.random((new_d2,R))).to(device)

    # do cross CPD
    tmp1, A2, A3, _, _ = OracleCPD(O1, R, 10, device, tmp1, A2, A3)
    A1, tmp2, A3, _, _ = OracleCPD(O2, R, 10, device, A1, tmp2, A3)
    # tmp1, A2, A3, _, _ = OracleCPD(O1, R, 5, device, tmp1, A2, A3)
    # A1, tmp2, A3, _, _ = OracleCPD(O2, R, 5, device, A1, tmp2, A3)

    return A1, A2, A3, tmp1, tmp2

def PREMA(X, O1, O2, mask, R, iteration, device):
    """
    PREMA:
        notation mapping (to the paper):
            A1 -> U, A2 -> V, A3 -> W, tmp1 -> Q1, tmp2 -> Q2
            O1 -> C1, O2 -> C2
    """

    # preparation
    new_d1, d2, d3 = O1.shape
    d1, new_d2, d3 = O2.shape
    A1, A2, A3, tmp1, tmp2 = PREMA_init(X, O1, O2, R, device)
    print ('finish initialization')
    print ()
    loss_list, rec_list = [], []
    
    # intermediate constant data (for fast gradient computation)
    X_1 = X.reshape(d1, -1)
    X_2 = X.permute(1,0,2).reshape(d2, -1)
    X_3 = X.permute(2,0,1).reshape(d3, -1)
    M1 = mask.reshape(d1, -1)
    M2 = mask.permute(1,0,2).reshape(d2, -1)
    M3 = mask.permute(2,0,1).reshape(d3, -1)

    def obj_func(X, coord, grad):
        """
        X = (A1, A2, A3, tmp1, tmp2)
        coord decides which one get the gradient
        """
        X[coord] -= grad
        A1, A2, A3, tmp1, tmp2 = X

        rec = torch.einsum('ir,jr,kr->ijk', A1, A2, A3)
        LOSS1 = torch.norm(O1 - torch.einsum('ir,jr,kr->ijk',tmp1,A2,A3)) / torch.norm(O1)
        LOSS2 = torch.norm(O2 - torch.einsum('ir,jr,kr->ijk',A1,tmp2,A3)) / torch.norm(O2)
        LOSS3 = torch.norm(mask * (rec - X)) / torch.norm(mask * X)
        LOSS = Lambda * LOSS1 ** 2 + Lambda * LOSS2 ** 2 + LOSS3 ** 2
        
        return LOSS

    # Kharti-rao product
    def kr(A, B):
        return torch.cat([torch.outer(A[:,i], B[:,i]).view(-1, 1) for i in range(A.shape[1])], dim=1)

    def grad_tmp1(Lambda):
        return - torch.einsum('ijk,jr,kr->ir',O1,A2,A3) + torch.einsum('ar,br,bj,cr,cj->aj',tmp1,A2,A2,A3,A3)
    
    def grad_tmp2(Lambda):
        return - torch.einsum('ijk,ir,kr->jr',O2,A1,A3) + torch.einsum('ar,br,bj,cr,cj->aj',tmp2,A1,A1,A3,A3)

    def grad_A1(Lambda):
        return - Lambda * torch.einsum('ijk,jr,kr->ir',O2,tmp2,A3) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A1,tmp2,tmp2,A3,A3)\
                            + (M1 * (A1 @ kr(A2, A3).T - X_1)) @ kr(A2, A3) + beta * torch.ones((d1, new_d1)).to(device) @ tmp1
    def grad_A2(Lambda):
        return - Lambda * torch.einsum('ijk,ir,kr->jr',O1,tmp1,A3) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A2,tmp1,tmp1,A3,A3)\
                            + (M2 * (A2 @ kr(A1, A3).T - X_2)) @ kr(A1, A3) + beta * torch.ones((d2, new_d2)).to(device) @ tmp2
    def grad_A3(Lambda):
        return - Lambda * torch.einsum('ijk,ir,jr->kr',O1,tmp1,A2) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A3,tmp1,tmp1,A2,A2)\
                - Lambda * torch.einsum('ijk,ir,jr->kr',O2,A1,tmp2) + Lambda * torch.einsum('ar,br,bj,cr,cj->aj',A3,A1,A1,tmp2,tmp2)\
                           + (M3 * (A3 @ kr(A1, A2).T - X_3)) @ kr(A1, A2)

    def line_search(obj_func, X, coord, grad, alpha=[1e-9, 3e-7], visit=[-1, -1], depth=3):

        if depth == 0:
            return alpha[0]

        result = []
        for i, j in zip(alpha, visit):
            result.append(obj_func(X, coord, i * grad)) if j < 0 else result.append(j)
        mid = sum(alpha) / 2

        if result[0] >= result[1]:
            return line_search(obj_func, X, coord, grad, alpha=[mid, alpha[1]], visit=[-1, result[1]], depth=depth-1)
        else:
            return line_search(obj_func, X, coord, grad, alpha=[alpha[0], mid], visit=[result[0], -1], depth=depth-1)


    alpha = 2e-8
    beta = 1e-3
    
    # start gradient descent
    tic = time.time()
    for i in range(iteration):
        Lambda = max(1. * np.exp(-i / 50), 1e-3)

        GRAD = grad_tmp1(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 3, GRAD)
        tmp1 -= alpha * GRAD

        GRAD = grad_tmp2(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 4, GRAD)
        tmp2 -= alpha * GRAD

        GRAD = grad_A1(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 0, GRAD)
        A1 -= alpha * GRAD

        GRAD = grad_A2(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 1, GRAD)
        A2 -= alpha * GRAD
        
        GRAD = grad_A3(Lambda)
        # alpha = line_search(obj_func, [A1, A2, A3, tmp1, tmp2], 2, GRAD)
        A3 -= alpha * GRAD
        
        # loss
        rec = torch.einsum('ir,jr,kr->ijk', A1, A2, A3)
        LOSS1 = torch.norm(O1 - torch.einsum('ir,jr,kr->ijk',tmp1,A2,A3)) / torch.norm(O1)
        LOSS2 = torch.norm(O2 - torch.einsum('ir,jr,kr->ijk',A1,tmp2,A3)) / torch.norm(O2)
        LOSS3 = torch.norm(mask * (rec - X)) / torch.norm(mask * X)
        LOSS = Lambda * LOSS1 ** 2 + Lambda * LOSS2 ** 2 + LOSS3 ** 2
        RecLOSS = 1 - torch.norm(rec - X) / torch.norm(X)
        if i % 20 == 0:
            print ('{}/{}'.format(i, iteration), 'loss:', LOSS.item(), 'rec:', RecLOSS.item(), 'time span: ', time.time() - tic)

        # collect loss
        tic = time.time()

        loss_list.append(LOSS); 
        rec_list.append(RecLOSS)

        # rescaling
        norm1 = (A1**2).sum(dim=0) ** .5
        norm2 = (A2**2).sum(dim=0) ** .5
        norm3 = (A3**2).sum(dim=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)

        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3
        tmp1 *= norm / norm1 
        tmp2 *= norm / norm2
        
    return A1, A2, A3, tmp1, tmp2, loss_list, rec_list


def jacobi(u, A, b, device):
    """
        A @ u.T = b, given initial u
    """
    # matrix preprocessing
    D = torch.diag(A).to(device) + 1e-5
    R = A - torch.diag(D)
    Z = torch.diag(D**(-1)) @ b
    B = - torch.diag(D**(-1)) @ R

    w = 2e-2
    for i in range(15):
        u = (1-w) * u + w * (Z + B @ u)
    return u

def twoStageOptimizer(u, weight, A, b, iteration, J, device):
    """
        <weight, A> @ u.T = <weight, b>
    """

    # get cumulative <weight, A> and <weight, b>
    A_ = torch.zeros(A[0].shape).to(device)
    b_ = torch.zeros(b[0].shape).to(device)
    for i in range(len(A)):
        A_ += weight[i] * A[i]
        b_ += weight[i] * b[i]

    # two stage optimization
    if (iteration < 1) and (J == 1):
        u = jacobi(u, A_, b_, device)
    else:
        u = torch.solve(b_, A_ + torch.eye(A_.shape[1]).to(device) * 1e-5)[0]
        # L = torch.cholesky(A_ + torch.eye(A_.shape[1]).to(device) * 1e-5)
        # y = torch.triangular_solve(b_, L.T, upper=True)[0]
        # u1 = torch.triangular_solve(y, L, upper=False)[0]
    return u

def interpolation(A, A_, device):
    """
    interp func for continuous mode
    """
    d, _ = A.shape
    tmp = torch.zeros(A.shape).to(device)
    if (d % 2 == 0):
        tmp[::2, :] = A_
        tmp[1:-1:2, :] = (A_[:-1, :] + A_[1:, :]) / 2
        tmp[-1, :] = A_[-1, :] / 2
    if (d % 2 == 1):
        tmp[::2, :] = A_
        tmp[1::2, :] = (A_[:-1, :] + A_[1:, :]) / 2
    return tmp


# [O1, O2], [[P11, P12], None], ['110', '010'], mask * T


def MTC(T, O1, O2, mask, R, iteration, device, J, M):
    """
        J in {0, 1}: jacobi iteration or not
        M in {0, 1}: muiltiresolution or not

        This model follows recursion:
            - if some requirment is met (in the most coarse level or M=0)
                we randomly initialize the factor
            - else
                (1) subsample d_t+1 to d_t;
                (2) solve a d_t problem;
                (3) interpolate from d_t to d_t+1

    """

    # preparation
    d1, d2, d3 = T.shape
    new_d1, _, _ = O1.shape
    _, _, new_d3 = O2.shape
    loss_list, rec_list = [], []
    
    if (max(d1, d2, d3, new_d1, new_d3) < 2 * R) and (M == 1) or M == 0:
        A1 = torch.FloatTensor(np.random.random((d1,R))).to(device)
        A2 = torch.FloatTensor(np.random.random((d2,R))).to(device)
        A3 = torch.FloatTensor(np.random.random((d3,R))).to(device)
        tmp1 = torch.FloatTensor(np.random.random((new_d1,R))).to(device)
        tmp3 = torch.FloatTensor(np.random.random((new_d3,R))).to(device)

    else:
        #--- subsampling ---#
        mask_ = mask.clone()
        T_ = T.clone()
        O1_ = O1.clone()
        O2_ = O2.clone()
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

        #--- solve d_t problem ---#
        A1_, A2_, A3_, tmp1_, tmp3_, _, _ = MTC(T_, O1_, O2_, mask_, R, 10, device, J, M)

        #--- interpolate ---#
        if d1 >= 2 * R:
            A1 = torch.FloatTensor(np.random.random((d1,R))).to(device); A1[:d1//2,:] = A1_
        else:
            A1 = A1_
        if d2 >= 2 * R:
            A2 = torch.FloatTensor(np.random.random((d2,R))).to(device); A2[:d2//2,:] = A2_
        else:
            A2 = A2_
        if d3 >= 2 * R:
            A3 = torch.FloatTensor(np.random.random((d3,R))).to(device); A3 = interpolation(A3, A3_, device)
        else:
            A3 = A3_
        if new_d1 >= 2 * R:
            tmp1 = torch.FloatTensor(np.random.random((new_d1,R))).to(device); tmp1[:new_d1//2,:] = tmp1_
        else:
            tmp1 = tmp1_
        if new_d3 >= 2 * R:
            tmp3 = torch.FloatTensor(np.random.random((new_d3,R))).to(device); tmp3 = interpolation(tmp3, tmp3_, device)
        else:
            tmp3 = tmp3_
        
    print ()
    print ('-- new resolution -- (d1, d2, d3, coarse_d1, coarse_d2)', d1, d2, d3, new_d1, new_d3)
    
    # coupled ALS
    tic = time.time()
    rec = torch.zeros(mask.shape).to(device)
    for i in range(iteration): 
        LAMBDA = max(1. * np.exp(-i / 50), 1e-2)

        T_ = mask * T + (1-mask) * rec
        
        # sub-iteration
        tmp1 = twoStageOptimizer(tmp1.T, [1.], [(A3.T@A3)*(A2.T@A2)], [torch.einsum('ijk,jr,kr->ri',O1,A2,A3)], i, J, device).T
        tmp3 = twoStageOptimizer(tmp3.T, [1.], [(A2.T@A2)*(A1.T@A1)], [torch.einsum('ijk,ir,jr->rk',O2,A1,A2)], i, J, device).T
        A1 = twoStageOptimizer(A1.T, [LAMBDA, 1.], [(tmp3.T@tmp3)*(A2.T@A2), (A3.T@A3)*(A2.T@A2)], \
                            [torch.einsum('ijk,jr,kr->ri',O2,A2,tmp3), torch.einsum('ijk,jr,kr->ri',T_,A2,A3)], i, J, device).T
        A3 = twoStageOptimizer(A3.T, [LAMBDA, 1.], [(A2.T@A2)*(tmp1.T@tmp1), (A2.T@A2)*(A1.T@A1)], \
                            [torch.einsum('ijk,ir,jr->rk',O1,tmp1,A2), torch.einsum('ijk,ir,jr->rk',T_,A1,A2)], i, J, device).T
        A2 = twoStageOptimizer(A2.T, [1., LAMBDA, LAMBDA], [(A3.T@A3)*(A1.T@A1), (A1.T@A1)*(tmp3.T@tmp3), (A3.T@A3)*(tmp1.T@tmp1)], \
                        [torch.einsum('ijk,ir,kr->rj',T_,A1,A3), torch.einsum('ijk,ir,kr->rj',O2,A1,tmp3), \
                        torch.einsum('ijk,ir,kr->rj',O1,tmp1,A3)], i, J, device).T
        
        # loss
        rec = torch.einsum('ir,jr,kr->ijk', A1, A2, A3)
        LOSS1 = torch.norm(O1 - torch.einsum('ir,jr,kr->ijk',tmp1,A2,A3)) / torch.norm(O1)
        LOSS2 = torch.norm(O2 - torch.einsum('ir,jr,kr->ijk',A1,A2,tmp3)) / torch.norm(O2)
        LOSS3 = torch.norm(mask * (rec - T)) / torch.norm(mask * T)
        LOSS = LAMBDA * LOSS1 ** 2 + LAMBDA * LOSS2 ** 2 + LOSS3 ** 2
        recLOSS = 1 - torch.norm(rec - T) / torch.norm(T)
        if i % 20 == 0:
            print ('{}/{}'.format(i, iteration), 'loss:', LOSS.item(), 'rec:', recLOSS.item(), 'time span:', time.time() - tic)

        # collect loss
        tic = time.time()
        loss_list.append(LOSS)
        rec_list.append(recLOSS)
        
        # rescaling
        norm1 = (A1**2).sum(dim=0) ** .5
        norm2 = (A2**2).sum(dim=0) ** .5
        norm3 = (A3**2).sum(dim=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)

        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3
        tmp1 *= norm / norm1
        tmp3 *= norm / norm3
        
    return A1, A2, A3, tmp1, tmp3, loss_list, rec_list


def MTC_P2(T, O1, O2, mask, P2, R, iteration, device, J, M):
    """
        J in {0, 1}: jacobi iteration or not
        M in {0, 1}: muiltiresolution or not

        This model follows recursion:
            - if some requirment is met (in the most coarse level or M=0)
                we randomly initialize the factor
            - else
                (1) subsample d_t+1 to d_t;
                (2) solve a d_t problem;
                (3) interpolate from d_t to d_t+1

    """

    # preparation
    d1, d2, d3 = T.shape
    new_d1, _, _ = O1.shape
    _, _, new_d3 = O2.shape
    loss_list, rec_list = [], []
    
    if (max(d1, d2, new_d1, new_d3) < 2 * R) and (M == 1) or M == 0:
        A1 = torch.FloatTensor(np.random.random((d1,R))).to(device)
        A2 = torch.FloatTensor(np.random.random((d2,R))).to(device)
        A3 = torch.FloatTensor(np.random.random((d3,R))).to(device)
        tmp1 = torch.FloatTensor(np.random.random((new_d1,R))).to(device)
        tmp3 = P2 @ A3

    else:
        #--- subsampling ---#
        mask_ = mask.clone()
        T_ = T.clone()
        O1_ = O1.clone()
        O2_ = O2.clone()
        P2_ = P2.clone()
        if d1 >= 2 * R:
            mask_ = mask_[:d1//2, :, :]; T_ = T_[:d1//2, :, :]; O2_ = O2_[:d1//2, :, :]
        if d2 >= 2 * R:
            mask_ = mask_[:, :d2//2, :]; T_ = T_[:, :d2//2, :]; O1_ = O1_[:, :d2//2, :]; O2_ = O2_[:, :d2//2, :]
        if new_d1 >= 2 * R:
            O1_ = O1_[:new_d1//2, :, :]
        if new_d3 >= 2 * R:
            mask_ = mask_[:, :, ::2]; T_ = T_[:, :, ::2]; O1_ = O1_[:, :, ::2]; O2_ = O2_[:, :, ::2]; P2_ = P2[::2, ::2]

        #--- solve d_t problem ---#
        A1_, A2_, A3_, tmp1_, tmp3_, _, _ = MTC_P2(T_, O1_, O2_, mask_, P2_, R, 10, device, J, M)

        #--- interpolate ---#
        if d1 >= 2 * R:
            A1 = torch.FloatTensor(np.random.random((d1,R))).to(device); A1[:d1//2,:] = A1_
        else:
            A1 = A1_
        if d2 >= 2 * R:
            A2 = torch.FloatTensor(np.random.random((d2,R))).to(device); A2[:d2//2,:] = A2_
        else:
            A2 = A2_
        if new_d3 >= 2 * R:
            A3 = torch.FloatTensor(np.random.random((d3,R))).to(device); A3 = interpolation(A3, A3_, device); tmp3 = P2 @ A3
        else:
            A3 = A3_; tmp3 = tmp3_ # we can also do "tmp3 = P2 @ A3_"
        if new_d1 >= 2 * R:
            tmp1 = torch.FloatTensor(np.random.random((new_d1,R))).to(device); tmp1[:new_d1//2,:] = tmp1_
        else:
            tmp1 = tmp1_
        
    print ()
    print ('-- new resolution -- (d1, d2, d3, coarse_d1, coarse_d2)', d1, d2, d3, new_d1, new_d3)
    
    # coupled-ALS
    tic = time.time()
    rec = torch.zeros(mask.shape).to(device)
    for i in range(iteration): 
        LAMBDA = max(1. * np.exp(-i / 50), 1e-2)

        # sub-iteration
        T_ = mask * T + (1-mask) * rec
        tmp1 = twoStageOptimizer(tmp1.T, [1.], [(A3.T@A3)*(A2.T@A2)], [torch.einsum('ijk,jr,kr->ri',O1,A2,A3)], i, J, device).T
        # tmp3 = optimize(tmp3.T, [1.], [(A2.T@A2)*(A1.T@A1)], [torch.einsum('ijk,ir,jr->rk',O2,A1,A2)], i).T
        A1 = twoStageOptimizer(A1.T, [LAMBDA, 1.], [(tmp3.T@tmp3)*(A2.T@A2), (A3.T@A3)*(A2.T@A2)], \
                            [torch.einsum('ijk,jr,kr->ri',O2,A2,tmp3), torch.einsum('ijk,jr,kr->ri',T_,A2,A3)], i, J, device).T
        A3 = twoStageOptimizer(A3.T, [LAMBDA, 1.], [(A2.T@A2)*(tmp1.T@tmp1), (A2.T@A2)*(A1.T@A1)], \
                            [torch.einsum('ijk,ir,jr->rk',O1,tmp1,A2), torch.einsum('ijk,ir,jr->rk',T_,A1,A2)], i, J, device).T
        tmp3 = P2 @ A3
        A2 = twoStageOptimizer(A2.T, [1., LAMBDA, LAMBDA], [(A3.T@A3)*(A1.T@A1), (A1.T@A1)*(tmp3.T@tmp3), (A3.T@A3)*(tmp1.T@tmp1)], \
                        [torch.einsum('ijk,ir,kr->rj',T_,A1,A3), torch.einsum('ijk,ir,kr->rj',O2,A1,tmp3), \
                        torch.einsum('ijk,ir,kr->rj',O1,tmp1,A3)], i, J, device).T
    
        # loss
        rec = torch.einsum('ir,jr,kr->ijk', A1, A2, A3)
        LOSS1 = torch.norm(O1 - torch.einsum('ir,jr,kr->ijk',tmp1,A2,A3)) / torch.norm(O1)
        LOSS2 = torch.norm(O2 - torch.einsum('ir,jr,kr->ijk',A1,A2,tmp3)) / torch.norm(O2)
        LOSS3 = torch.norm(mask * (rec - T)) / torch.norm(mask * T)
        LOSS = LAMBDA * LOSS1 ** 2 + LAMBDA * LOSS2 ** 2 + LOSS3 ** 2
        recLOSS = 1 - torch.norm(rec - T) / torch.norm(T)
        if i % 20 == 0:
            print ('{}/{}'.format(i, iteration), 'loss:', LOSS.item(), 'rec:', recLOSS.item(), 'time span:', time.time() - tic)
        
        # collect loss
        tic = time.time()
        loss_list.append(LOSS)
        rec_list.append(recLOSS)
        
        # rescaling
        norm1 = (A1**2).sum(dim=0) ** .5
        norm2 = (A2**2).sum(dim=0) ** .5
        norm3 = (A3**2).sum(dim=0) ** .5
        norm = (norm1 * norm2 * norm3) ** (1/3.)

        A1 *= norm / norm1
        A2 *= norm / norm2
        A3 *= norm / norm3
        tmp1 *= norm / norm1
        tmp3 *= norm / norm3
        
    return A1, A2, A3, tmp1, tmp3, loss_list, rec_list
