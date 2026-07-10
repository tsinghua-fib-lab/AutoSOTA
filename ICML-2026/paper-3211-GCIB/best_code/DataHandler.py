import pickle
import numpy as np
from numpy import ndarray, dtype, signedinteger
from numpy._typing import _32Bit
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix, issparse
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from typing import List, Tuple, Any
import time

if args.device == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"


class DataHandler:
    def __init__(self):
        self.tstLoader = None
        self.trnLoader = None
        self.allOneAdj = None
        self.torchBiAdj = None
        self.tstMat = None
        self.trnMat = None
        if args.data == 'Beibei':
            predir = './Datasets/Beibei/'
            self.behaviortype = ['buy', 'pv', 'cart']
        elif args.data == 'Taobao':
            predir = './Datasets/Taobao/'
            self.behaviortype = ['buy', 'pv', 'cart']
        elif args.data == 'Tmall':
            predir = './Datasets/Tmall/'
            self.behaviortype = ['buy', 'cart', 'click', 'collect']
        elif args.data == 'yelp':
            predir = './Datasets/Yelp/'
            self.behaviortype = ['pos', 'neutral', 'neg', 'tip']
        elif args.data == 'ml10m':
            predir = './Datasets/ML10M/'
            self.behaviortype = ['pos', 'neutral', 'neg']
        self.predir = predir
        if args.data == 'Beibei' or args.data == 'Taobao' or args.data == 'Tmall':
            # target is buy
            self.trnfile = predir + 'buytrain_matrix.pkl'
            self.tstfile = predir + 'buytest_matrix.pkl'
        else:
            # target is pos
            self.trnfile = predir + 'pos_matrix.pkl'
            self.tstfile = predir + 'test_matrix.pkl'
        
        for type in self.behaviortype:
            if type == 'buy' or type == 'pos':
                continue
            elif type == 'cart':
                self.cartfile = predir + 'cart_matrix.pkl'
            elif type == 'pv':
                self.pvfile = predir + 'pv_matrix.pkl'
            elif type == 'click':
                self.clickfile = predir + 'click_matrix.pkl'
            elif type == 'collect':
                self.collectfile = predir + 'collect_matrix.pkl'
            elif type == 'neutral':
                self.neutralfile = predir + 'neutral_matrix.pkl'
            elif type == 'neg':
                self.negfile = predir + 'neg_matrix.pkl'
            elif type == 'tip':
                self.tipfile = predir + 'tip_matrix.pkl'




    def loadFile(self):
        # data是原始的交互数据集 见dataprocess.py 
        pass

    def loadOneFile(self, filename, datatype="origin") -> sp.coo_matrix:
        with open(filename, 'rb') as fs:
            # data是处理过的稀疏矩阵，pkl文件中是scipy.sparse._csr.csr_matrix类型 astype转换为numpy或者string类型,并把稀疏矩阵非0处赋值为true再转成np.float32
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) is not coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        # csr_matrix.sum() 返回的是np.matrix  sum不改变shape
        degree = np.array(mat.sum(axis=-1))
        ## reshape 将degree从二维转为一维
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt1 = np.reshape(np.power(degree, -1), [-1])
        ## 将无穷值设置为0
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrt1[np.isinf(dInvSqrt)] = 0.0
        ## 度的对角矩阵
        dInvSqrtMat = sp.diags(dInvSqrt)
        dInvSqr1tMat = sp.diags(dInvSqrt1)
        ## D-0.5 * A * D-0.5,   D-1 * A
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo(), dInvSqr1tMat.dot(mat).tocoo()

    def makeTorchAdj(self, mat) -> torch.sparse_coo:
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # 加上单位矩阵
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat, matD_1A = self.normalizeAdj(mat)

        # make cuda tensor
        ## np.vstack(tup:sequence of ndarrays,), t.from_numpy()
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        ## torch.Size类
        shape = torch.Size(mat.shape)
        ##  sparse(idx,val,size)   size (list, tuple, or torch.Size, optional)
        # return torch.sparse.FloatTensor(idxs, vals, shape).to(device)
        idxs1 = torch.from_numpy(np.vstack([matD_1A.row, matD_1A.col]).astype(np.int64))
        vals1 = torch.from_numpy(matD_1A.data.astype(np.float32))
        ## torch.Size类
        shape1 = torch.Size(matD_1A.shape)

        return torch.sparse_coo_tensor(idxs, vals, shape).to(device), torch.sparse_coo_tensor(idxs1, vals1, shape1).to(device)

    def makeTorchAdjNosym(self, mat: sp.coo_matrix) -> torch.sparse_coo:
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse_coo_tensor(idxs, vals, shape).to(device)

    def makesparsetensor(self, mat: sp.csr_matrix) -> torch.sparse_coo:
        idx = torch.from_numpy(np.concatenate((mat.row, mat.col), axis=0).astype(np.int64))
        val = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse_coo_tensor(idx, val, shape).to(device)

    # 全1的邻接矩阵 为了sum邻居的emb
    def makeAllOne(self, torchAdj):
        idxs = torchAdj._indices()
        vals = torch.ones_like(torchAdj._values())
        shape = torchAdj.shape
        # return torch.sparse.FloatTensor(idxs, vals, shape).to(device)
        return torch.sparse_coo_tensor(idxs, vals, shape).to(device)


    def LoadData(self):
        self.trnMat = self.loadOneFile(self.trnfile)
        self.tstMat = self.loadOneFile(self.tstfile)
        # self.pvMat = self.loadOneFile(self.pvfile)
        # self.cartMat = self.loadOneFile(self.cartfile)

        args.user, args.item = self.trnMat.shape

        self.torchBuyAdj, _ = self.makeTorchAdj(self.trnMat)
        self.allOneAdj = self.makeAllOne(self.torchBuyAdj)

        self.auxAdj_list = []
        self.aux_allOneAdj_list = []
        sum_adj = sp.coo_matrix((args.user, args.item))
        sum_adj += self.trnMat
        for type in self.behaviortype:
            if type == 'buy' or type == 'pos':
                continue
            elif type == 'cart':
                self.cartMat = self.loadOneFile(self.cartfile)
                sum_adj += self.cartMat
                self.torchcartAdj, _ = self.makeTorchAdj(self.cartMat)
                self.auxAdj_list.append(self.torchcartAdj)
                self.aux_allOneAdj_list.append(self.makeAllOne(self.torchcartAdj))
            elif type == 'pv':
                self.pvMat = self.loadOneFile(self.pvfile)
                sum_adj += self.pvMat
                self.torchpvAdj, _ = self.makeTorchAdj(self.pvMat)
                self.auxAdj_list.append(self.torchpvAdj)
                self.aux_allOneAdj_list.append(self.makeAllOne(self.torchpvAdj))
            elif type == 'click':
                self.clickMat = self.loadOneFile(self.clickfile)
                sum_adj += self.clickMat
                self.torchclickAdj, _ = self.makeTorchAdj(self.clickMat)
                self.auxAdj_list.append(self.torchclickAdj)
                self.aux_allOneAdj_list.append(self.makeAllOne(self.torchclickAdj))
            elif type == 'collect':
                self.collectMat = self.loadOneFile(self.collectfile)
                sum_adj += self.collectMat
                self.torchcollectAdj, _ = self.makeTorchAdj(self.collectMat)
                self.auxAdj_list.append(self.torchcollectAdj)
                self.aux_allOneAdj_list.append(self.makeAllOne(self.torchcollectAdj))
            elif type == 'neutral':
                self.neutralMat = self.loadOneFile(self.neutralfile)
                sum_adj += self.neutralMat
                self.torchneutralAdj, _ = self.makeTorchAdj(self.neutralMat)
                self.auxAdj_list.append(self.torchneutralAdj)
                self.aux_allOneAdj_list.append(self.makeAllOne(self.torchneutralAdj))
            elif type == 'neg':
                self.negMat = self.loadOneFile(self.negfile)
                sum_adj += self.negMat
                self.torchnegAdj, _ = self.makeTorchAdj(self.negMat)
                self.auxAdj_list.append(self.torchnegAdj)
                self.aux_allOneAdj_list.append(self.makeAllOne(self.torchnegAdj))
            elif type == 'tip':
                self.tipMat = self.loadOneFile(self.tipfile)
                sum_adj += self.tipMat
                self.torchtipAdj, _ = self.makeTorchAdj(self.tipMat)
                self.auxAdj_list.append(self.torchtipAdj)
                self.aux_allOneAdj_list.append(self.makeAllOne(self.torchtipAdj))

        self.sum_auxAdj = torch.zeros_like(self.auxAdj_list[0])
        for adj in self.auxAdj_list:
            self.sum_auxAdj += adj

        sum_adj.sum_duplicates()
        sum_adj = sum_adj.tocoo()
        sum_adj = coo_matrix((np.ones_like(sum_adj.data), (sum_adj.row, sum_adj.col)),shape=(args.user, args.item))
        self.torchSumAdj, _ = self.makeTorchAdj(sum_adj)


        

        trnData = TrnData(self.trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=4)

        # 正常的测试集
        tstData = TstData(self.tstMat, self.trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=4)


class TrnData(data.Dataset):
    def __init__(self, coomat, mask=None):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.mask = mask

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if self.mask is not None:
            return self.rows[idx], self.cols[idx], self.negs[idx], self.mask[idx]
        else:
            return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(data.Dataset):
    def __init__(self, tstmat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0
        ## 列表乘法
        tstLocs = [None] * tstmat.shape[0]
        tstUsrs = set()
        for i in range(len(tstmat.data)):
            row = tstmat.row[i]
            col = tstmat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs  # test users
        self.tstLocs = tstLocs  # test users'items

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        ## reshape 是拉成一维  因为dataloader返回batch的时候会升维度 变成二维；如果不拉成一维度 就会从二维变成三维；切片操作不改变维度
        # return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
        return self.tstUsrs[idx], self.csrmat[self.tstUsrs[idx]].toarray().flatten()