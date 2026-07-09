import torch

import numpy as np
import scipy as sp

from scipy.sparse import csc_matrix,csr_matrix,coo_matrix

def csc_torch_to_scipy(A):
    return csc_matrix((A.values().detach().numpy(),A.row_indices().numpy(),A.ccol_indices().numpy()),shape=A.size())

def csc_scipy_to_torch(A):
    return torch.sparse_csc_tensor(torch.tensor(A.indptr,dtype=torch.int64),torch.tensor(A.indices,dtype=torch.int64),torch.tensor(A.data),size=np.shape(A),dtype=torch.float64)


def coo_torch_to_scipy(A):
    # A must be coalesced
    return coo_matrix((A.values().detach().numpy(),(A._indices()[0,:].numpy(),A._indices()[1,:].numpy())),shape=A.size())

def coo_scipy_to_torch(A):
    i = torch.tensor(np.vstack((A.row,A.col)),dtype=torch.long)
    v = torch.tensor(A.data,dtype=torch.float64)
    return torch.sparse_coo_tensor(i,v,size=np.shape(A))

def initialize_torch_from_npz(filename):
    data = np.load(filename, allow_pickle=True)
    Qnp, qnp, Gnp, hnp, Anp, bnp = data["Q"][()], data["q"][()], data["G"][()], data["h"][()], data["A"][()], data["b"][()]

    Qnp, Gnp = Qnp.tocoo(), Gnp.tocoo()
    __to_coo = lambda M: torch.sparse_coo_tensor(torch.stack([torch.tensor(M.row), torch.tensor(M.col)]),
                                                 torch.tensor(M.data), M.shape, dtype=torch.float64, requires_grad=True)
    Q, G = __to_coo(Qnp).to_sparse_csc(), __to_coo(Gnp).to_sparse_csc()

    q = torch.tensor(qnp, requires_grad=True)
    h = torch.tensor(hnp, requires_grad=True)
    if Anp is not None:
        Anp = Anp.tocoo()
        A = __to_coo(Anp).to_sparse_csc()
        b = torch.tensor(bnp, requires_grad=True)

    return Q,q,G,h,A,b

class sparse_row_norm(torch.autograd.Function):
    '''
    Differentiate sparse row-wise 1 or 2 norm. This is not supported in torch 2.3.1 or external sparse_torch?
    '''

    @staticmethod
    def forward(ctx, A, p):
        '''
        '''

        assert(p == 1 or p == 2)

        ctx.p = p

        A = csc_torch_to_scipy(A) # TODO : convert to list of sparse matrices if sparse
        ctx.A = A

        N = np.expand_dims(sp.sparse.linalg.norm(A,ord=p,axis=1),-1)
        ctx.N = N

        return torch.tensor(N,dtype=torch.float64)

    @staticmethod
    def backward(ctx,grad_output):
        '''
        dN/dA ; gradient is sparse
        '''

        if ctx.p == 1:
            dN = ctx.A.sign().multiply(grad_output.numpy())
        elif ctx.p == 2:
            dN = ctx.A.multiply(grad_output.numpy() * np.power(ctx.N,-1))

        # return torch.tensor(dN.todense(),dtype=torch.float64), None # TODO : sparse or not?
        return csc_scipy_to_torch(csc_matrix(dN)), None


# TODO : note these functions are not optimized, but for the problems that are computationally feasible, these seem to be fine
class sparse_row_normalize(torch.autograd.Function):
    '''
    Differentiate sparse row normalization. Insane this is not supported...
    '''

    @staticmethod
    def forward(ctx, A, N):
        '''
        '''

        N = N.numpy()
        ctx.N = N
        ctx.A = csc_torch_to_scipy(A)

        A = csr_matrix(ctx.A)
        A.data /= N[A.nonzero()[0],0] # normalize rows in-place
        A = csc_matrix(A)

        return csc_scipy_to_torch(A)

    @staticmethod
    def backward(ctx,grad_output):
        '''
        '''

        if grad_output.layout is torch.strided:
            dL = grad_output.numpy()
        else:
            dL = grad_output.to_dense().numpy()

        dA_norm = np.multiply(np.power(ctx.N, -1), dL)
        dN = -np.multiply(np.expand_dims(np.diag(ctx.A @ dL.T), -1), np.power(ctx.N, -2))

        return torch.tensor(dA_norm, dtype=torch.float64), torch.tensor(dN, dtype=torch.float64)

def test_normalization():

    # p = 2
    p = 1

    rng = np.random.default_rng()
    S = sp.sparse.random(3, 4, density=0.5, random_state=rng, format="csc")

    S = csc_scipy_to_torch(S)
    S.requires_grad_()

    D = S.clone().detach().to_dense().numpy()
    D = torch.tensor(D,dtype=torch.float64,requires_grad=True)

    print("Inputs: ")
    print(D)
    print(S.to_dense())
    print("\n")

    row_norm = sparse_row_norm.apply
    row_normalize = sparse_row_normalize.apply

    db = torch.rand(3,1,dtype=torch.float64,requires_grad=True)
    sb = db.clone().detach().requires_grad_()

    N_D = torch.unsqueeze(torch.linalg.vector_norm(D, ord=p, dim=-1),-1)
    D_new = torch.div(D, N_D)
    db_new = torch.div(db, N_D)

    N_S = row_norm(S,p)
    S_new = row_normalize(S,N_S)
    sb_new = torch.div(sb, N_S)

    # S = S.to_sparse_coo()
    # mat = torch_sparse.SparseTensor(row=S._indices()[0, :], col=S._indices()[1, :], value=S._values())

    print("Check outputs: ")
    print(N_D)
    print(N_S)
    print(db)
    print(sb)
    print("Check outputs: ")
    print("\n")

    print("Check gradients: ")
    l_D = D_new
    l_S = S_new
    l_D.backward(torch.ones(3,4))
    l_S.backward(torch.ones(3,4))
    print("Check gradients: ")
    print(D.grad)
    print(S.grad)

    # print("Check gradients: ")
    # l_db = db_new
    # l_sb = sb_new
    # l_db.backward(torch.ones(3,1))
    # l_sb.backward(torch.ones(3,1))
    # print(sb.grad)
    # print(db.grad)

    return

# test_normalization()
