"""
Paper: MetaSymNet: A Tree-like Symbol Network with Adaptive Architecture and
Activation Functions
DL ACM: https://dl.acm.org/doi/10.1609/aaai.v39i25.34915
Code slightly modified from https://github.com/1716757342/MetSymNet @ f26c357
"""

import copy

import numpy as np
import numpy.typing as npt
import sympy as sp
from scipy.optimize import minimize

max_val = 10e10
S = [
    "+",
    "*",
    "/",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
]  # Library of candidate activation functions
# S = ['+', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt']
# S = ['+', '*', 'sin', 'exp', 'log','sqrt']
# S = ['+', '*','sin','cos']
# S = ['+', '*','sin']
# S = ['+','*','/','sin','cos','exp']
SX = []


# for i in range(len(X)):
#     S.append('x'+str(i))
def div(x1, x2, pwd=1):
    if pwd == 1:
        aa = x1 / (x2 + (x2 / abs(x2)) * 1e-5)
        aa[np.argwhere(aa >= max_val)] = max_val
        aa[np.argwhere(aa <= -max_val)] = -max_val
        return aa
    if pwd == 0:
        return x1 / x2


def log_s(x1, pwd=1):
    if pwd == 1:
        aa = np.log(abs(x1) + 1e-5)
        # aa[np.argwhere(abs(x1) <= 0.001)] = -100
        aa[np.argwhere(aa >= max_val)] = max_val
        aa[np.argwhere(aa <= -max_val)] = -max_val

        # x1[np.argwhere(x1 <= 0.00001)] = 0.00001
        # aa = np.log(abs(x1))
        # aa[np.argwhere(abs(x1) <= 0.001)] = -100
        # aa[np.argwhere(aa >= max_val)] = max_val
        # aa[np.argwhere(aa <= -max_val)] = -max_val
        return aa

    if pwd == 0:
        return np.log(x1)


def sqrt_s(x1, pwd=1):
    if pwd == 1:
        aa = np.sqrt(abs(x1) + 10e-5)
        aa[np.argwhere(aa >= max_val)] = max_val
        return aa
    if pwd == 0:
        return np.sqrt(abs(x1))


def exp_s(x, pwd=1):
    if pwd == 1:
        aa = np.exp(np.clip(x, -999999, np.log(max_val)))
        # aa = np.exp(x)
        # aa[np.argwhere(aa >= max_val)] = max_val
        return aa
    if pwd == 0:
        return np.exp(x)


def mul(x1, x2, pwd=1):
    if pwd == 1:
        aa = x1 * x2
        aa[np.argwhere(aa >= max_val)] = max_val
        aa[np.argwhere(aa <= -max_val)] = -max_val
        return aa
    if pwd == 0:
        return x1 * x2


def forward(s, z, x, xx):
    global z22
    z22 = 0
    if s in ["+", "-", "*", "/"]:
        if s == "+":
            z22 = z + x
        if s == "-":
            z22 = z - x
        if s == "*":
            z22 = z * x
        if s == "/":
            z22 = z / x
    if "x" in s and "e" not in s:
        z22 = xx[int(s[1]) - 1]
    if s in ["sin", "cos", "log", "exp", "sqrt"]:
        if s == "exp":
            z22 = sp.exp(z)
        if s == "cos":
            z22 = sp.cos(z)
        if s == "sin":
            z22 = sp.sin(z)
        if s == "log":
            z22 = sp.log(z)
        if s == "sqrt":
            z22 = sp.sqrt(z)
    return z22


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


global LL
LL = []


def pretravale(treenode):  ## Returns a preorder traversal
    if treenode is None:
        return None
    # print(treenode.val)
    LL.append(treenode.val)
    pretravale(treenode.left)
    pretravale(treenode.right)
    return LL


def pro_t(
    l, X, C
):  ## The forward propagation function gives l: network preorder traversal and all the optimization constants C.

    num = X.shape[1]
    stack = []
    global value
    value = []
    nn = len(np.argwhere(np.array(l) == "s"))  ## The number of s nodes in the network
    nx = len(np.argwhere(np.array(l) == "x"))  ## The number of x nodes in the network
    A = C[
        0 : nn * (len(S) + 2)
    ]  ## Extract the parameters corresponding to the symbol s (selection parameter, amplitude parameter and bias parameter)
    B = C[
        nn * (len(S) + 2) :
    ]  ## The parameters corresponding to the symbol x are extracted
    # print('len B ',len(B))
    # print(x)
    # print(x2)
    # print(np.stack((x, x2),axis=0))
    for i in range(len(l)):  #### computer the first order traversal.
        s = l[-(i + 1)]
        if (
            s == "x"
        ):  ## Take out the parameters to be optimized corresponding to each x node.

            OUTX = (
                lambda a1: a1[0]
                * (  #### Decide whice x to choose.
                    np.dot(softmax(a1[1 : len(SX) + 1], c=2), X)
                )
                + a1[-1]
            )  ##Calculate the metafunction of variables with bias

            vx = OUTX(
                B[(nx - 1) * (len(SX) + 2) : (nx) * (len(SX) + 2)]
            )  # Extraction obtains the selection parameters for each x node
            # vx = OUTX(B[(nx - 1) * (4):(nx) * (4)])  #Extraction obtains the selection parameters for each x node
            nx = nx - 1
            stack.append(vx)
            value.append(vx)
        if s == "s":
            # print(f"shape(X): {X.shape}, len(S): {len(S)}, num: {num}")
            OUT1 = lambda a1: (
                a1[0]
                * (
                    np.dot(
                        softmax(a1[1 : len(S) + 1], c=2),
                        sy(stack.pop(), stack.pop(), X).reshape(len(S), num),
                    )
                )
                + a1[len(S) + 1]
            )

            v = OUT1(A[(nn - 1) * (len(S) + 2) : (nn) * (len(S) + 2)])
            nn = nn - 1
            stack.append(v)
            value.append(vx)
    return stack[0]


def all_farward(A, l2, X):
    global s, stack1, v1, D, B
    stack1 = []
    D = np.array(A[0 : int(len(A) / 2)])  ## Extract W
    B = np.array(A[int(len(A) / 2) : len(A)])  ## Extract B
    # print(A)
    # print(B)
    # print(D)
    for i in range(len(l2)):
        s = l2[-(i + 1)]
        c = D[-(i + 1)]
        b = B[-(i + 1)]
        if "x" in s and "e" not in s:
            stack1.append(X[int(s[1]) - 1] * c + b)  ##
        if s in ["sin", "cos", "log", "exp", "sqrt"]:
            if s == "exp":
                v1 = exp_s(stack1.pop())
            if s == "log":
                v1 = log_s(stack1.pop())
            if s == "cos":
                v1 = np.cos(stack1.pop())
            if s == "sin":
                v1 = np.sin(stack1.pop())
            if s == "sqrt":
                v1 = sqrt_s(stack1.pop())
            stack1.append(v1 * c + b)
        if s in ["+", "-", "*", "/"]:
            if s == "+":
                v1 = stack1.pop() + stack1.pop()
            if s == "-":
                v1 = stack1.pop() - stack1.pop()
            if s == "*":
                v1 = mul(stack1.pop(), stack1.pop())
            if s == "/":
                v1 = div(stack1.pop(), stack1.pop())
            stack1.append(v1 * c + b)
    return stack1[0]


def r2(EEe, yy_1):
    return 1 - (np.sum((yy_1 - EEe) ** 2)) / (np.sum((yy_1 - np.mean(yy_1)) ** 2))


global C
C = 10


def softmax(x, c=C):
    global C
    # print('ccccccccc',C)
    x = x - max(x)
    # C = C + 0.01/9
    return np.array((np.exp(C * x) / np.sum(np.exp(C * x))))


def sy_symbol(s, x, x_1, xx):
    global vsy
    if s in ["sin", "cos", "log", "exp", "sqrt"]:
        if s == "exp":
            vsy = exp_s(x)
        if s == "log":
            vsy = log_s(x)
        if s == "cos":
            vsy = np.cos(x)
        if s == "sin":
            vsy = np.sin(x)
        if s == "sqrt":
            vsy = sqrt_s(x)
    if s in ["+", "-", "*", "/"]:
        if s == "+":
            vsy = x + x_1
        if s == "-":
            vsy = x - x_1
        if s == "*":
            vsy = x * x_1
        if s == "/":
            vsy = div(x, x_1)
    # if 'x' in s and 'e' not in s:
    #     vsy = xx[int(s[1])-1]
    return vsy


def sy(x, x_1, xx_1):  ###### sy() for multiple dimensions
    sy_list = []
    for i in range(len(S)):
        sy_list.append(sy_symbol(S[i], x, x_1, xx_1))
    out = np.array(sy_list)
    # print(f"sy: len(out)={len(out)}")
    return out


# n_v = 2
def LR(l, C, cc1=0, cc2=1, lam=1):
    stack = []
    nn = len(np.argwhere(np.array(l) == "s"))
    nx = len(np.argwhere(np.array(l) == "x"))
    A = C[0 : nn * (len(S) + 2)]
    B = C[nn * (len(S) + 2) :]
    for i in range(len(l)):  ####Compute the preorder traversal
        s = l[-(i + 1)]
        if s == "s":
            ##L22 penalty, which guarantees a protrusion after softmax
            # L22 = lambda a1: -np.sum((softmax(a1[1:len(S) + 1])) * np.log2((softmax(a1[1:len(S) + 1]))))
            L22 = (
                lambda a1: -(np.log2(np.max(softmax(a1[1 : len(S) + 1])))) / nn
            )  ##Select out the selection parameters to find the entropy loss
            v = cc2 * L22(A[(nn - 1) * (len(S) + 2) : (nn) * (len(S) + 2)])
            nn = nn - 1
            stack.append(v)
        if s == "x":
            # Lx22 = lambda a1: -np.sum(softmax(a1) * log_di(softmax(a1),3))
            # Lx22 = lambda a1: -np.sum(softmax(a1) * np.log2(softmax(a1)))
            Lx22 = lambda a1: -(
                1 * np.log2(np.max(softmax(a1[1 : len(SX) + 1])))
            ) / len(SX)
            v = cc2 * Lx22(B[(nx - 1) * (len(SX) + 2) : (nx) * (len(SX) + 2)])
            nx = nx - 1
            stack.append(1 * v)
    return lam * np.sum(stack)


def find_two_largest_indices(nums):
    if len(nums) < 2:
        raise ValueError("List must contain at least two elements.")

    # Initialize the indices of the largest and second largest values
    if nums[0] > nums[1]:
        max_idx, second_max_idx = 0, 1
    else:
        max_idx, second_max_idx = 1, 0

    # Iterate over the list and update the indices of the largest and next largest values
    for i in range(2, len(nums)):
        if nums[i] > nums[max_idx]:
            second_max_idx = max_idx
            max_idx = i
        elif nums[i] > nums[second_max_idx]:
            second_max_idx = i

    return max_idx, second_max_idx


def mean_squared_error(vector, array):  ### 求 MSE
    if vector.ndim != 1 or array.ndim != 2:
        raise ValueError(
            "Input vector must be one-dimensional and input array must be two-dimensional."
        )

    vector = vector.reshape(1, -1)  # 将 vector 从 (N,) 转变为 (1, N)

    # Calculating the difference
    difference = vector - array
    # Calculate the squared difference
    squared_difference = difference**2
    # Calculate the mean squared error for each vector
    mse = squared_difference.mean(axis=1)

    return mse


def E2N(L):  #### The expression is completed as a network
    stack1 = []
    for i in range(len(L)):
        s = L[i]
        if "x" in s and "e" not in s:
            s_add = ["x"]
            stack1 = stack1 + s_add
        if s in ["sin", "cos", "log", "exp", "sqrt"]:
            s_add = ["s", "x"]
            stack1 = stack1 + s_add
        if s in ["+", "-", "*", "/"]:
            s_add = ["s"]
            stack1 = stack1 + s_add
    return stack1


"""
def f(x_1, x_2):
    # F =2 * np.sin(x_1)*cos(x_2)
    F = x_1**4 - x_1**3 + 0.5 * x_2**2 - x_2
    return F
"""


def metasymnet(
    X: npt.NDArray,
    Y: npt.NDArray,
    netstruct: list[str] = ["s", "s", "s", "x", "x", "x", "s", "s", "x", "x", "x"],
    iters: int = 100,
):
    # BFGS Optimization
    # num = 40  ## Number of sampling points
    #
    # dim = 2
    # X = np.random.rand(dim, num) * 4 - 2
    # # Variables are defined using loops and dynamic variable name generation
    # for i in range(dim):
    #     globals()[f"x_{i+1}"] = X[i]
    # y_1 = 0.0
    # # print(x_1,x_2)
    # y_1 = f(x_1, x_2)
    #
    # Y = copy.deepcopy(y_1)

    print(f"X.shape={X.shape}")
    dim = X.shape[0]
    num = X.shape[1]
    print(f"dim: {dim}, num: {num}")

    for i in range(dim):
        globals()[f"x_{i+1}"] = X[i]

    y_1 = copy.deepcopy(Y)
    print(f"Y.shape={Y.shape}")

    for i in range(len(X)):
        # S.append('x'+str(i+1))
        SX.append("x" + str(i + 1))

    ## Initialize the network structure
    l = copy.deepcopy(netstruct)
    # l = ['s', 's', 'x', 'x', 's', 'x', 'x']
    # l = ['s','s','s','x','x','x','s','x','x']
    # l = ["s", "s", "s", "x", "x", "x", "s", "s", "x", "x", "x"]
    # l = ['s', 's', 's', 'x', 'x', 's', 'x', 'x', 's', 's', 'x', 'x', 's', 'x', 'x']
    # l = ['s','x','x']

    ITE = iters
    gobal_best = -100
    for j in range(ITE):
        s_n = len(
            np.argwhere(np.array(l) == "s")
        )  # The number of S nodes in the network
        x_n = len(
            np.argwhere(np.array(l) == "x")
        )  # Number of x-variable nodes in the network (number of leaf nodes)

        out3 = lambda a2: pro_t(l, X, a2)
        fun = lambda a2: np.mean((y_1 - out3(a2)) ** 2)
        fun_loss = lambda a2: fun(a2) + 0.2 * LR(l, a2, lam=1)
        # fun_loss = lambda a2 : fun(a2)

        print("\n===> L-BFGS_B optimization:")
        # print(" g    f(x1,x2)     x1      x2  ")
        # print("===  ==========  ======  ======")
        # loss_best = 100
        r_best = -100
        a_best = np.random.randn((len(S) + 2) * s_n + (len(SX) + 2) * x_n)
        counter = 0
        threthold = 0.999999
        for i in range(2):

            a = list(range(len(l)))
            stack_t = []
            ##### Building binary trees The third parameter of each tree is a constant. The second value of the node is the value that is not multiplied by a constant
            for i in range(len(l)):  ## Build a binary tree network (only S and SX)
                st = l[-(i + 1)]
                if st == "x":
                    a[-(i + 1)] = TreeNode("x")
                    stack_t.append(a[-(i + 1)])
                if st == "s":
                    a[-(i + 1)] = TreeNode("s")
                    a[-(i + 1)].left = stack_t.pop()
                    a[-(i + 1)].right = stack_t.pop()
                    stack_t.append(a[-(i + 1)])
            counter = counter + 1
            print("Iter : ", counter)  ## Output number of iterations
            Z_all = np.random.randn(
                (len(S) + 2) * s_n + (len(SX) + 2) * x_n
            )  ## Initializing parameters

            res = minimize(
                fun_loss, Z_all, method="L-BFGS-B"
            )  ## Numerical optimization
            # print('dddd',value)
            loss = fun(res.x)  ## The optimized parameters are extracted
            R2 = 1 - (loss / (np.sum((Y - np.mean(Y)) ** 2)))  ## Calculate r2
            if R2 >= r_best:
                a_best = copy.deepcopy(res.x)
                r_best = copy.deepcopy(R2)
                print("r_best", r_best)

            if r_best >= threthold:
                print("end_r_best", r_best)
                break
        L = []
        NN = 0
        NX = 0
        a_best_A = copy.deepcopy(
            a_best[0 : s_n * (len(S) + 2)]
        )  # The parameters of the best S-node obtained earlier are extracted

        a_best_B = copy.deepcopy(
            a_best[s_n * (len(S) + 2) :]
        )  # The parameters of the best x-node obtained earlier are extracted
        for i in range(len(l)):
            if l[i] == "s":
                aa = np.array(
                    copy.deepcopy(a_best_A[NN * (len(S) + 2) : (NN + 1) * (len(S) + 2)])
                )
                # print(NN*(len(S)+2),(NN+1)*(len(S)+2))
                # print(aa)
                az = list(copy.deepcopy(aa[1:-1]))
                # print(az)
                ss = S[
                    az.index(max(az))
                ]  ## Extract the basic symbols that the current S-node (PANGU meta-function) will evolve.
                L.append(ss)
                NN = NN + 1
            if l[i] == "x":
                P_x_all = a_best_B[NX * (len(SX) + 2) : (NX + 1) * (len(SX) + 2)]
                P_x_select = a_best_B[
                    NX * (len(SX) + 2) + 1 : (NX + 1) * (len(SX) + 2) - 1
                ]
                P_x_select_s = softmax(P_x_select, c=2)
                ax = list(
                    copy.deepcopy(P_x_select)
                )  ## Extract the variable selection parameters
                t2 = find_two_largest_indices(ax)  ## Select the first two variables

                X_l = P_x_all[0] * P_x_select_s[t2[0]] * X[t2[0]] + P_x_all[-1]
                X_r = P_x_all[0] * P_x_select_s[t2[1]] * X[t2[1]] + P_x_all[-1]
                out_lr = sy(
                    X_l, X_r, X
                )  ### xl and xr are fed into the activation function library for numerical operations
                # print(np.array(value[i]))
                # print(out_lr)
                sel_mse = list(mean_squared_error(np.array(value[i]), out_lr))
                # print('666666',sel_mse)
                sx = S[sel_mse.index(min(sel_mse))]
                if sx in ["+", "-", "*", "/"]:
                    sx_add = [sx, SX[t2[0]], SX[t2[1]]]
                if sx in ["sin", "cos", "exp", "sqrt", "log"]:
                    sx_add = [sx, SX[t2[0]], SX[t2[1]]]
                if "x" in sx and "e" not in sx:
                    sx_add = [sx]
                # print('44444',L)
                L = L + sx_add
                # print('55555',L)
                NX = NX + 1
        print("L", L)
        stack_pro = []
        ###### Building a binary tree replaces each node of the tree with a good one for later optimization.
        a = list(range(len(L)))
        for i in range(
            len(L)
        ):  ## An expression binary tree is built based on the preorder traversal of the evolved expression obtained above
            st = L[-(i + 1)]
            if "x" in st and "e" not in st:
                a[-(i + 1)] = TreeNode(L[-(i + 1)])
                a[-(i + 1)].left = None
                a[-(i + 1)].right = None
                stack_t.append(a[-(i + 1)])
            else:
                if L[-(i + 1)] in ["+", "*", "/"]:
                    a[-(i + 1)] = TreeNode(L[-(i + 1)])
                    a[-(i + 1)].left = stack_t.pop()
                    a[-(i + 1)].right = stack_t.pop()
                    stack_t.append(a[-(i + 1)])
                if L[-(i + 1)] in ["sin", "exp", "cos", "log", "sqrt"]:
                    a[-(i + 1)] = TreeNode(L[-(i + 1)])
                    a[-(i + 1)].left = stack_t.pop()
                    a[-(i + 1)].right = stack_t.pop()
                    a[-(i + 1)].right = None
                    stack_t.append(a[-(i + 1)])
                if (
                    "x" in L[-(i + 1)] and "e" not in L[-(i + 1)]
                ):  ## Determine whether s evolves into a variable
                    a[-(i + 1)] = TreeNode(L[-(i + 1)])
                    a[-(i + 1)].left = None
                    a[-(i + 1)].right = None
                    stack_t.append(a[-(i + 1)])

        l_out = []
        # print('888888',a[0])
        l_out = pretravale(
            a[0]
        )  ## Returns a preorder traversal of the best expression after evolution
        print("l_out", l_out)
        #### The binary tree of expressions evolved into expressions has been established.
        LL = []
        # r_best = -100
        # l_out = ['*', 'exp', '*', 'x', 'x', 'sin', 'x2']
        fun_out = lambda aa: all_farward(aa, l_out, X)
        LOSS = lambda aa: np.mean((y_1 - fun_out(aa)) ** 2)  ## Loss function
        a2_best = np.random.randn(
            len(l_out) * 2
        )  ## Initialize w and b of the optimal expression and refine
        r2_best = -100
        for i in range(2):
            Z_c = np.random.randn(len(l_out) * 2)
            res2 = minimize(LOSS, Z_c, method="L-BFGS-B")
            loss = LOSS(res2.x)
            R2 = 1 - (loss / (np.mean((Y - np.mean(Y)) ** 2)))
            if R2 >= r2_best:
                r2_best = R2
                a2_best = res2.x

        #### The recovery expression is reduced with sympy.
        XS = []
        for i in range(dim):
            globals()[f"x{i+1}"] = sp.Symbol("x" + str(i + 1))
            XS.append(globals()[f"x{i+1}"])

        #### Calculate the final output sympy type and further simplify.
        stack_pro1 = []
        for k in range(len(l_out)):
            sol = l_out[-(k + 1)]
            D2 = a2_best[0 : int(len(a2_best) / 2)]
            B2 = a2_best[int(len(a2_best) / 2) : len(a2_best)]
            d1 = D2[-(k + 1)]
            b1 = B2[-(k + 1)]
            if (
                "x" in sol and "e" not in sol
            ):  ### The reason we use and 'e' not in so is to rule out the effect of exp
                v = d1 * XS[int(sol[1]) - 1] + b1
                stack_pro1.append(v)
            if sol in ["+", "*", "/"]:
                v = d1 * forward(sol, stack_pro1.pop(), stack_pro1.pop(), X) + b1
                stack_pro1.append(v)
            if sol in ["sin", "exp", "cos", "log", "sqrt"]:
                v = d1 * forward(sol, stack_pro1.pop(), X[0], X) + b1
                stack_pro1.append(v)
        # print(stack_pro1[0])
        print("The highest $R^2$ is: ", r_best)
        print("----------The expression for the prediction---------- ")
        expr_out = sp.simplify(stack_pro1[0])
        print(expr_out)

        if r2_best >= gobal_best:
            gobal_best = r2_best
            l_out_best = copy.deepcopy(l_out)
        l = E2N(l_out)  #### Complete the network
        print(l)

    return expr_out
