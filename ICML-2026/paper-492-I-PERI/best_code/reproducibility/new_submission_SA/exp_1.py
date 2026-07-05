import numpy as np
from sklearn.linear_model import LinearRegression
from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm, DtDynamicSCM, create_random_linear_dt_dynamic_scm_from_ftadmg
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag, FtDirectedAcyclicGraph
from pyciphod.utils.scms.scm import create_random_linear_scm_from_admg, LinearSCM

def create_dag():
    g = FtDirectedAcyclicGraph()

    X2 = DTimeVar("X", 2)

    Y3 =  DTimeVar("Y", 3)

    U2 = DTimeVar("U", 2)

    Z2 = DTimeVar("Z", 2)

    W2 = DTimeVar("W", 2)

    g.add_vertices([X2, Y3, U2, Z2, W2])



    g.add_directed_edge(X2, Y3)

    g.add_directed_edge(Z2, X2)

    g.add_directed_edge(W2, Y3)

    g.add_directed_edge(U2, Y3)
    g.add_directed_edge(U2, X2)
    g.add_directed_edge(U2, Z2)
    g.add_directed_edge(U2, W2)

    return g

def create_ft_dag():
    g = FtDirectedAcyclicGraph()

    X3 =  DTimeVar("X", 3)
    X2 = DTimeVar("X", 2)
    X1 = DTimeVar("X", 1)

    Y3 =  DTimeVar("Y", 3)
    Y2 = DTimeVar("Y", 2)
    Y1 = DTimeVar("Y", 1)

    U3 =  DTimeVar("U", 3)
    U2 = DTimeVar("U", 2)
    U1 = DTimeVar("U", 1)

    Z3 =  DTimeVar("Z", 3)
    Z2 = DTimeVar("Z", 2)
    Z1 = DTimeVar("Z", 1)

    W3 =  DTimeVar("W", 3)
    W2 = DTimeVar("W", 2)
    W1 = DTimeVar("W", 1)

    g.add_vertices([X1, X2, X3, Y1, Y2, Y3, U1, U2, U3, Z1, Z2, Z3, W1, W2, W3])

    # g.add_directed_edge(Y1, Y2)
    # g.add_directed_edge(Y2, Y3)
    g.add_directed_edge(Z1, Z2)
    g.add_directed_edge(Z2, Z3)
    g.add_directed_edge(W1, W2)
    g.add_directed_edge(W2, W3)
    g.add_directed_edge(U1, U2)
    g.add_directed_edge(U2, U3)

    # g.add_directed_edge(U1, Y2)
    # g.add_directed_edge(U1, X2)
    # g.add_directed_edge(U1, Z2)
    # g.add_directed_edge(U1, W2)
    # g.add_directed_edge(U2, Y3)
    # g.add_directed_edge(U2, X3)
    # g.add_directed_edge(U2, Z3)
    # g.add_directed_edge(U2, W3)


    g.add_directed_edge(X1, Y1)
    g.add_directed_edge(X2, Y2)
    g.add_directed_edge(X3, Y3)
    g.add_directed_edge(X1, Y2)
    g.add_directed_edge(X2, Y3)

    g.add_directed_edge(X1, Z1)
    g.add_directed_edge(X2, Z2)
    g.add_directed_edge(X3, Z3)
    g.add_directed_edge(X1, Z2)
    g.add_directed_edge(X2, Z3)

    # g.add_directed_edge(Z1, X1)
    # g.add_directed_edge(Z2, X2)
    # g.add_directed_edge(Z3, X3)
    # g.add_directed_edge(Z1, X2)
    # g.add_directed_edge(Z2, X3)



    g.add_directed_edge(W1, Y1)
    g.add_directed_edge(W2, Y2)
    g.add_directed_edge(W3, Y3)
    g.add_directed_edge(W1, Y2)
    g.add_directed_edge(W2, Y3)

    g.add_directed_edge(U3, Y3)
    g.add_directed_edge(U3, X3)
    g.add_directed_edge(U3, Z3)
    g.add_directed_edge(U3, W3)
    g.add_directed_edge(U2, Y2)
    g.add_directed_edge(U2, X2)
    g.add_directed_edge(U2, Z2)
    g.add_directed_edge(U2, W2)
    g.add_directed_edge(U1, Y1)
    g.add_directed_edge(U1, X1)
    g.add_directed_edge(U1, Z1)
    g.add_directed_edge(U1, W1)
    return g


def to_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def two_steps_lr(y, x, z, w, cw, cy):
    y = np.asarray(y).ravel()
    x = to_2d(x)
    z = to_2d(z)
    w = to_2d(w)
    if len(cw)>0:
        cw = to_2d(cw)
    if len(cy) > 0:
        cy = to_2d(cy)

    # Stage 1
    stage1 = LinearRegression()
    if len(cw)>0:
        cov1 = np.hstack([x, z, cw])
    else:
        cov1 = np.hstack([x, z])
    stage1.fit(cov1, w)
    w_hat = stage1.predict(cov1)

    # Stage 2
    stage2 = LinearRegression()
    if len(cy) > 0:
        cov2 = np.hstack([x, cy, w_hat])
    else:
        cov2 = np.hstack([x, w_hat])
    stage2.fit(cov2, y)

    beta_iv = stage2.coef_[0]
    return beta_iv


def one_steps_lr(y, x, c):
    y = np.asarray(y).ravel()
    x = to_2d(x)
    c = to_2d(c)

    lr_model = LinearRegression()
    cov = np.hstack([x, c])
    lr_model.fit(cov, y)
    beta = lr_model.coef_[0]
    return beta


def run_single_iteration(seed: int, type_dag="ftdag"):
    x_t_1 = DTimeVar("X", 2)
    y_t = DTimeVar("Y", 3)
    if type_dag == "ftdag":
        w_set = [
            DTimeVar("W", 1),
            DTimeVar("W", 2),
            DTimeVar("W", 3)
        ]
        z_set = [
            DTimeVar("Z", 1),
            DTimeVar("Z", 2),
            DTimeVar("Z", 3)
        ]
        cw_set = [
            DTimeVar("X", 1),
            DTimeVar("Z", 1),
            DTimeVar("W", 1),
            DTimeVar("Y", 1),
            DTimeVar("X", 3),
        ]
        cy_set = [
            DTimeVar("X", 1),
            DTimeVar("Z", 1),
            DTimeVar("W", 1),
            DTimeVar("Y", 1),
            DTimeVar("X", 3),
        ]

        u_set = [
            DTimeVar("U", 1),
            DTimeVar("U", 2),
            DTimeVar("U", 3)
        ]
    else:
        w_set = [
            DTimeVar("W", 2),
        ]
        z_set = [
            DTimeVar("Z", 2),
        ]
        cw_set = [
        ]
        cy_set = [
        ]

        u_set = [
            DTimeVar("U", 2),
        ]

    dag = create_ft_dag()

    # scm = create_random_linear_dt_dynamic_scm_from_ftadmg(
    #     ftadmg=dag,
    #     causally_stationary=True,
    #     u_dist=np.random.normal,
    #     seed=seed,
    # )
    scm, coefficients, intercepts = create_random_linear_scm_from_admg(admg=dag, seed=seed)
    coefficients[(x_t_1, y_t)] = -0.8
    for ut in u_set:
        for zt in z_set:
            if (ut, zt) in coefficients.keys():
                step = np.random.uniform(0.01, 0.4)
                coefficients[(ut, zt)] = 0.5 + step
        for wt in w_set:
            if (ut, wt) in coefficients.keys():
                step = np.random.uniform(0.01, 0.4)
                coefficients[(ut, wt)] = 0.5 + step

    # for zt in z_set:
    #     if (x_t_1, zt) in coefficients.keys():
    #         coefficients[(x_t_1, zt)] = 1
    # for wt in w_set:
    #     if (wt, y_t) in coefficients.keys():
    #         coefficients[(wt, y_t)] = 1
    scm = LinearSCM(v=scm.v, u=scm.u, coefficients=coefficients, intercepts=intercepts, u_dist=scm._u_dist_map)


    beta_gt = coefficients[(x_t_1, y_t)]

    df = scm.generate_data(
        n_samples=10000,
        include_latent=False,
        seed=seed,
    )

    y_ar = df[y_t].to_numpy()
    x_ar = df[x_t_1].to_numpy()
    z_ar = df[z_set].to_numpy()
    w_ar = df[w_set].to_numpy()
    cy_ar = df[cy_set].to_numpy()
    cw_ar = df[cw_set].to_numpy()
    cu_ar = df[cy_set + u_set].to_numpy()
    # u_ar = df[u_set].to_numpy()

    beta_hat_prox = two_steps_lr(y_ar, x_ar, z_ar, w_ar, cw_ar, cy_ar)
    beta_hat_prox = float(beta_hat_prox)

    beta_hat = one_steps_lr(y_ar, x_ar, cu_ar)
    beta_hat = float(beta_hat)

    return beta_hat_prox, beta_hat, beta_gt


if __name__ == '__main__':
    beta_hat_prox_list = list()
    beta_hat_list = list()
    beta_gt_list = list()
    for i in range(10):
        beta_h_prox, beta_h, beta_t = run_single_iteration(seed=i, type_dag="ftdag")
        beta_hat_prox_list.append(beta_h_prox)
        beta_hat_list.append(beta_h)
        beta_gt_list.append(beta_t)
    print(beta_hat_prox_list)
    print(beta_hat_list)
    print(beta_gt_list)

    beta_gt_list = np.array(beta_gt_list)
    beta_hat_prox_list = np.array(beta_hat_prox_list)
    errors = beta_gt_list - beta_hat_prox_list
    rmse = np.sqrt(np.mean(errors ** 2))
    var = np.var(errors)
    print(rmse, var)

    beta_hat_list = np.array(beta_hat_list)
    errors = beta_gt_list - beta_hat_list
    rmse = np.sqrt(np.mean(errors ** 2))
    var = np.var(errors)
    print(rmse, var)

