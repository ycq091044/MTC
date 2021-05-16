import numpy as np
import pickle
import time
import argparse
from model import OracleCPD, BGD, PREMA, CMTF, CPC_ALS, MTC, MTC_P2
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='MTC', help="model name")
parser.add_argument('--partial', type=float, default=0.95, help="partial observation")
parser.add_argument('--jacobi', type=int, default=0, help="jacobi (1) or not (0)")
parser.add_argument('--multi', type=int, default=0, help="multi (1) or not (0)")
args = parser.parse_args()

# loading data

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

"""
    First, sort on the categorical mode.
    The code provides a more efficient way than paper does.
    We directly sort in descent order at the beginning and pick [:d//2] during subsampling
"""

if 'MTC' in args.model:
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

elif args.model in ['BGD', 'PREMA', 'CMTF']:
    T = np.transpose(T, (0,2,1))
    O1 = np.transpose(O1, (0,2,1))
    O2 = np.transpose(O2, (0,2,1))




if __name__ == '__main__':

    # set random seed
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # pick the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = torch.FloatTensor(T).to(device)
    O1 = torch.FloatTensor(O1).to(device)
    O2 = torch.FloatTensor(O2).to(device)
    P2 = torch.FloatTensor(P2).to(device)
    mask = torch.FloatTensor(np.random.random(T.shape) < args.partial).to(device)

    iteration = 201
    R = 10

    if args.model == 'OracleCPD':
        A1, A2, A3, loss, rec = OracleCPD(T, R, iteration, device)
    
    elif args.model == 'CPC-ALS':
        A1, A2, A3, loss, rec = CPC_ALS(T, mask, R, iteration, device)

    elif args.model == 'BGD':
        A1, A2, A3, _, _, loss, rec = BGD(T, O1, O2, mask, R, iteration, device)

    elif args.model == 'PREMA':
        A1, A2, A3, _, _, loss, rec = PREMA(T, O1, O2, mask, R, iteration, device)

    elif args.model == 'CMTF':
        A1, A2, A3, _, _, loss, rec = CMTF(T, O1, O2, mask, R, iteration, device)

    elif args.model == 'MTC':
        A1, A2, A3, _, _, loss, rec = MTC(T, O1, O2, mask, R, iteration, device, args.jacobi, args.multi)

    elif args.model == 'MTC-P2':
        A1, A2, A3, _, _, loss, rec = MTC_P2(T, O1, O2, mask, P2, R, iteration, device, args.jacobi, args.multi)
    
    else:
        print ('sorry, the model name is incorrect. Please choose from the following list:')
        print ('1. OracleCPD, 2. CPC-ALS, 3. BGD, 4. PREMA, 5. CMTF, 6. MTC, 7. MTC-P2')
