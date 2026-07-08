import PI_class_EnbPI_journal as EnbPI  # For me
import utils_EnbPI_journal as util
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Lasso, RidgeCV, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import math
import warnings
import itertools
import importlib
import time
import os
import sys
import keras
warnings.filterwarnings("ignore")

'''Sec 5.1. Figure 1'''


def DGP_and_run_detailed(param, test_style, save_fig, first_run=True):
    '''
        # A. No Change point, detailed look at each case.
    '''
    # NOTE: "first_run" means we have NOT generated "Data_dc" and "beta_star".
    # But as I sometimes need to tune parameters while keeping them the same, I will just save them to save time
    # Case (1) No Change point, for detailed plot
    # A. Create Data
    Ttot, stronglymixing, high_dim, curr_fX, tseries, current_regr = param[test_style]
    importlib.reload(sys.modules['utils_EnbPI_journal'])
    if first_run:
        Data_dc = util.DGP(curr_fX, T_tot=Ttot, tseries=tseries,
                           high_dim=high_dim, change_points=False, stronglymixing=stronglymixing)
        with open(f'Data_nochangepts_{test_style}.p', 'wb') as fp:
            pickle.dump(Data_dc, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'Data_nochangepts_{test_style}.p', 'rb') as fp:
            Data_dc = pickle.load(fp)
    util.quick_plt(Data_dc, current_regr, tseries, stronglymixing)
    # B. Start fitting
    train_frac = 0.5
    if tseries:
        Data_dc_tseries = {}
        for key in Data_dc.keys():
            Data_dc_tseries[key] = Data_dc[key][:Ttot - 100]
        regr_no_change = util.split_and_train(Data_dc_tseries, train_frac=train_frac,
                                              mathcalA=current_regr, alpha=0.05, itrial=0, return_full=True)
        Data_dc = Data_dc_tseries
    else:
        regr_no_change = util.split_and_train(Data_dc, train_frac=train_frac,
                                              mathcalA=current_regr, alpha=0.05, itrial=0, return_full=True)
    # C. Visualize results
    # Note, below is because except for AR(1) epsilon_t, the computation of optimal \beta^* is the same
    if stronglymixing:
        first_run = True
    else:
        if test_style == 'linear':
            first_run = True
        else:
            first_run = False
    util.visualize_everything(Data_dc, regr_no_change,
                              train_frac=train_frac, save_fig=save_fig, tseries=tseries, stronglymixing=stronglymixing, first_run=first_run)
    return [Data_dc, regr_no_change]


def cond_coverage(param, test_style, first_run):
    # Store the resulting interval lower ends by train_size in a dictionary, since we wish to have short interval (but sometimes intervals are too wide)
    importlib.reload(sys.modules['utils_EnbPI_journal'])  # For me
    test_style = 'nonlinear'
    Ttot, stronglymixing, high_dim, curr_fX, tseries, current_regr = param[test_style]
    cond_cov_res = {}
    train_frac_ls = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for train_frac in train_frac_ls:
        T = int(Ttot * train_frac)
        # First train a model with the first T data
        Data_dc = util.DGP(curr_fX, T_tot=Ttot, tseries=tseries,
                           high_dim=high_dim, change_points=False, stronglymixing=stronglymixing)
        Data_dc = {key: Data_dc[key][:T + 1] for key in Data_dc.keys()}
        alpha = 0.1
        regr_no_change = util.split_and_train(Data_dc, train_frac=T / (T + 1),
                                              mathcalA=current_regr, alpha=alpha, itrial=0, return_full=True)
        # Then just return the residual and prediction on X_{T+1}.
        current_mod = regr_no_change[1]
        fhat_X = current_mod.Ensemble_pred_interval_centers[0]
        resid = current_mod.Ensemble_online_resid[:T]
        beta_hat_bin = current_mod.beta_hat_bins[0]
        PI_lower = fhat_X + np.percentile(resid, math.ceil(100 * beta_hat_bin))
        PI_upper = fhat_X + \
            np.percentile(resid, math.ceil(100 * (1 - alpha + beta_hat_bin)))
        # Then generate \{\eps_{T+1,i}\}_{i=1}^{T_1} depending on test style
        if stronglymixing:
            Finv = util.F_inv_stronglymixingDGP
            rho = 0.6
        else:
            Finv = util.F_inv
            rho = 0
        fX = Data_dc['f(X)'][-1]
        first_run = True
        if first_run:
            eps_T_p1_ls = []
            # NOTE: doing so takes some time (3min for AR(1)), so I will store the results.
            T1 = 100  # just examine performance on the next 100 points
            for i in range(T1):
                U = np.random.uniform(size=T + 1)
                Errs = np.zeros(T + 1)
                Finv_vec = np.vectorize(Finv)
                U_inv = Finv_vec(U)
                Errs[0] = U_inv[0]
                for j in range(1, T + 1):
                    Errs[j] = rho * Errs[j - 1] + U_inv[j]
                eps_T_p1_ls.append(Errs[-1])
            eps_T_p1_ls = np.array(eps_T_p1_ls)
            with open(f'cond_cov_resid_{test_style}_{train_frac}.p', 'wb') as fp:
                pickle.dump(eps_T_p1_ls, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'cond_cov_resid_{test_style}_{train_frac}.p', 'rb') as fp:
                eps_T_p1_ls = pickle.load(fp)
        Y_cond = fX + eps_T_p1_ls
        plt.rcParams.update({'font.size': 16,
                             'axes.titlesize': 16, 'axes.labelsize': 16,
                             'legend.fontsize': 12})
        print(f'Results with {train_frac*100}% as training data')
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(Y_cond, color='orange',
                label=r'$f(X_{T+1})+\{\epsilon_{T+1,i}\}_{i=1}^{T_1}$')
        ax.plot(np.repeat(fhat_X, T1), color='blue', label=r'$\hat{Y}_{T+1}$')
        ax.set_ylim(np.min([PI_lower / 2, PI_lower * 2]), PI_upper * 1.5)
        ax.fill_between(range(T1),
                        np.repeat(PI_lower, T1), np.repeat(PI_upper, T1), color='blue', alpha=0.2)
        ax.set_xlabel('Prediction Time Index')
        # ax.set_title(r'Conditional Coverage at $T+1$ is ' +
        #              f'{np.round(np.mean((Y_cond >= PI_lower) & (Y_cond <= PI_upper)),2)}')
        # ax.legend(loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.4))
        ax.legend(loc='upper center', ncol=2)
        plt.show()
        cond_cov_res[train_frac] = [
            fig, fX, eps_T_p1_ls, fhat_X, PI_lower, PI_upper]
    # 1. Save ptwise result
    fig.savefig(f'Simulation/EmpvsActual_CondPtwiseWidth_nochangepts_{regr_name}{tse}{strongm}.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)  # Save the last figure, as it has the shortest width
    with open(f'cond_cov_LARGE_results{test_style}.p', 'wb') as fp:
        pickle.dump(cond_cov_res, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # 2. Plot average coverage
    with open(f'cond_cov_LARGE_results{test_style}.p', 'rb') as fp:
        cond_cov_res = pickle.load(fp)
    mean_width_dic = {}
    mean_cov_dic = {}
    for train_size in cond_cov_res.keys():
        _, fX, eps_T_p1_ls, fhat_X, PI_lower, PI_upper = cond_cov_res[train_size]
        Y_cond = fX + eps_T_p1_ls
        mean_width_dic[train_size] = PI_upper - PI_lower
        mean_cov_dic[train_size] = np.round(
            np.mean((Y_cond >= PI_lower) & (Y_cond <= PI_upper)), 2)
    if stronglymixing:
        savename = 'beta_star_stronglymixing.p'
    else:
        savename = 'beta_star_nostronglymixing.p'
    with open(savename, 'rb') as fp:
        beta_star = pickle.load(fp)
    importlib.reload(sys.modules['utils_EnbPI_journal'])  # For me
    fig_cond_cov = util.EmpvsActual_AveWidth(
        beta_star, alpha, mean_width_dic, mean_cov_dic, stronglymixing, cond_cov=True)
    regr_name = current_regr.__class__.__name__
    tse = '_tseries' if tseries else ''
    strongm = '_mixing' if stronglymixing else ''
    fig_cond_cov.savefig(
        f'Simulation/EmpvsActual_CondAveResult_nochangepts_{regr_name}{tse}{strongm}.pdf', dpi=300, bbox_inches='tight',
        pad_inches=0)
    # 3. Lastly, plot histogram of residuals
    fig_hist, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(resid, bins=15, kde=True, ax=ax,
                 label=r'LOO residuals $\{\hat{\epsilon_t}\}_{t=1}^T$')
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0.2))
    ax.set_xlabel('Residual Value')
    ax.axes.get_yaxis().set_visible(False)
    fig_hist.savefig(
        f'Simulation/EmpvsActual_CondHist_nochangepts_{regr_name}{tse}{strongm}.pdf', dpi=300, bbox_inches='tight',
        pad_inches=0)


def get_ave_figure(result_ave, save_fig):
    '''
        # B. No Change point, average cov & width over T
    '''
    importlib.reload(sys.modules['utils_EnbPI_journal'])  # For me
    if stronglymixing:
        savename = 'beta_star_stronglymixing.p'
    else:
        savename = 'beta_star_nostronglymixing.p'
    with open(savename, 'rb') as fp:
        beta_star = pickle.load(fp)
    tot = len(train_frac_ls)
    fig_avewidth = util.EmpvsActual_AveWidth(beta_star, alpha=0.05,
                                             mean_width_dic={train_frac_ls[i]: np.array(result_ave['width'])[
                                                 i] for i in range(tot)},
                                             mean_cov_dic={train_frac_ls[i]: np.array(result_ave['coverage'])[i] for i in range(tot)}, stronglymixing=stronglymixing)
    regr_name = current_regr.__class__.__name__
    tse = '_tseries' if tseries else ''
    strongm = '_mixing' if stronglymixing else ''
    if save_fig:
        fig_avewidth.savefig(
            f'Simulation/EmpvsActual_AveResult_nochangepts_{regr_name}{tse}{strongm}.pdf', dpi=300, bbox_inches='tight',
            pad_inches=0)

# This dictionary contains parameters "Ttot, stronglymixing, high_dim, curr_fX, tseries, current_regr"
# We have 2100 Because the first 100 points do NOT have feature = d, which is 100
# NOTE: lasso parameter affects fitted values. Too large alpha make predictions almost flat (undesirable)


param = {'linear': [2000, False, False, util.True_mod_linear_pre, False, LinearRegression(fit_intercept=False)],
         'lasso': [2000, False, True, util.True_mod_lasso_pre, True, Lasso(alpha=1)],
         'nonlinear': [2100, True, True, util.True_mod_nonlinear_pre, True, util.keras_mod()]}


for test_style in param.keys():
    # for test_style in ['nonlinear']:
    # A. No Change point, detailed look at each case.
    save_fig = True  # Set as "False" if just trying things out
    # NOTE: set this to be False once we have ran each "test_style" once.
    first_run = False
    Data_dc, regr_no_change = DGP_and_run_detailed(
        param, test_style, save_fig, first_run)
    # B. No Change point, average cov & width over T
    train_frac_ls = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    regr_no_change_Train = pd.DataFrame()
    _, stronglymixing, _, _, tseries, current_regr = param[test_style]
    # See if width decreases as T increases. If NOT, run the 'Tsmall' and 'Tlarge' below
    with open(f'Data_nochangepts_{test_style}.p', 'rb') as fp:
        Data_dc = pickle.load(fp)
    for train_frac in train_frac_ls:
        regr_no_change_train = util.split_and_train(Data_dc, train_frac=train_frac,
                                                    mathcalA=current_regr, alpha=0.05, itrial=0)
        regr_no_change_Train = pd.concat(
            [regr_no_change_Train, regr_no_change_train])
    save_fig = True  # Set as "False" if just trying things out
    get_ave_figure(regr_no_change_Train, save_fig)


'''Sec 5.1 Figure 2'''


def helix(r, H, stronglymixing=True):
    n = 1000
    # Plot a helix along the x-axis
    theta_max = 8 * np.pi
    theta = np.linspace(0, theta_max, n)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = H*theta
    X = np.array(np.c_[x, y, z])
    eps = 0.0001
    FX = X[:, 0]*np.sqrt(np.abs(X[:, 1]))/np.sqrt(X[:, 2]+eps)
    rho_dict = {True: 0.6, False: 0}
    rho = rho_dict[stronglymixing]
    Errs = np.zeros(n)
    U = np.random.uniform(size=n)
    # AR(1) error or just N(0,0.1) error
    Errs[0] = util.F_inv_stronglymixingDGP(U[0])
    for i in range(1, n):
        Errs[i] = rho*Errs[i-1]+util.F_inv_stronglymixingDGP(U[i])
    Y = FX+Errs
    # Sometimes, I may cut a bit of the initial data if they differ too much from the rest
    start = 0
    return {'Y': Y[start:], 'X': X[start:], 'f(X)': FX[start:], 'Eps': Errs[start:], 'Helix': [theta[start:], x[start:], y[start:], z[start:]]}


def plt_helix(Data_helix):
    plt.rcParams.update({'font.size': 18})
    # Original Helix
    theta, x, y, z = Data_helix['Helix']
    Y = Data_helix['Y']
    fig = plt.figure(figsize=(10, 5))
    ax_helix = fig.add_subplot(121, projection='3d')  # Original Helix
    ax_helix.scatter(x, y, z, color=cm.Paired(Y), lw=2)
    ax_Y = fig.add_subplot(122)  # Y
    ax_Y.plot(range(len(Y)), Y, lw=1)


# Generate Helix Data
r = 10  # radius size
H = 3  # height
Data_helix = helix(r, H, stronglymixing=True)

plt_helix(Data_helix)


def plt_helix_with_PI(Data_helix, EnbPI_result, regr_name):
    # see https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # for heatmap
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 20, 'legend.fontsize': 22})
    # Original Helix
    theta, x, y, z = Data_helix['Helix']
    Y = Data_helix['Y']
    X = Data_helix['X']
    FX = Data_helix['f(X)']
    # Fitted Values
    Yhat = EnbPI_result.Ensemble_pred_interval_centers
    PI_ends = EnbPI_result.Ensemble_pred_interval_ends
    upper_Y, lower_Y = PI_ends['upper'], PI_ends['lower']
    Ttot = len(FX)
    T1 = len(upper_Y)
    T = Ttot-T1
    fig = plt.figure(figsize=(24, 7), constrained_layout=True)
    gs = fig.add_gridspec(5, 8)
    ax_Y = fig.add_subplot(gs[0:2, :])  # Y_t with prediction intervals
    # ax_Y.set_title(f'Prediction Intervals by EnbPI {regr_name}')
    ax_helixupper = fig.add_subplot(gs[2:, 0:2], projection='3d')
    ax_helixupper.set_title(r'Colored by interval upper end')
    ax_helix = fig.add_subplot(gs[2:, 2:4], projection='3d')  # Original Helix
    ax_helix.set_title(r'Colored by $Y_t$')
    ax_helixpred = fig.add_subplot(gs[2:, 4:6], projection='3d')
    ax_helixpred.set_title(r'Colored by $\hat{Y_t}$')
    ax_helixlower = fig.add_subplot(gs[2:, 6:8], projection='3d')
    ax_helixlower.set_title(r'Colored by interval lower end')
    # Plot original helix
    ax_helix.scatter(x[T:], y[T:], z[T:], color=cm.Paired(Y[T:]), lw=2)
    ax_helixpred.scatter(x[T:], y[T:], z[T:], color=cm.Paired(Yhat), lw=2)
    ax_helixupper.scatter(x[T:], y[T:], z[T:], color=cm.Paired(upper_Y), lw=2)
    ax_helixlower.scatter(x[T:], y[T:], z[T:], color=cm.Paired(lower_Y), lw=2)
    for ax in [ax_helix, ax_helixpred, ax_helixupper, ax_helixlower]:
        ax.set_xlabel(r'$r\cos(\theta_t)$')
        ax.set_ylabel(r'$r\sin(\theta_t)$')
        ax.set_zlabel(r'$H\theta_t$')
        ax.zaxis.labelpad = 8
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
    # Plot actual Y with upper and lower PI
    Y_upper = np.append(EnbPI_result.Ensemble_train_interval_centers, upper_Y)
    Y_lower = np.append(EnbPI_result.Ensemble_train_interval_centers, lower_Y)
    Yhat = np.append(EnbPI_result.Ensemble_train_interval_centers, Yhat)
    ax_Y.plot(range(Ttot), FX,
              label=r'$\frac{r\cos(\theta_t)\cdot \sqrt{|r\sin(\theta_t)|}}{\sqrt{H\theta_t+\epsilon}}$', color='black', lw=1)
    ax_Y.plot(range(Ttot), Yhat, label=r'$\hat{Y_t}$', color='blue', lw=2, alpha=0.8)
    ax_Y.plot(range(Ttot), Y, label=r'$Y_t$', color='orange', lw=2, alpha=0.8)
    ax_Y.fill_between(range(Ttot), Y_lower, Y_upper, color='blue', alpha=0.2)
    ax_Y.axvline(x=T, linewidth=2, color='red')
    ax_Y.text(T-35, -12, r'$T=500$', color='red')
    ax_Y.set_xlabel(r'Time Index $t$')
    fig.tight_layout(pad=3)
    ax_Y.legend(ncol=3, loc='upper right', bbox_to_anchor=(1, 1.05))
    fig.tight_layout()
    plt.show()
    return fig


fig_helix_RF = plt_helix_with_PI(Data_helix, EnbPI_result_RF, regr_name='Random Forest')

# Run below when I want to fine tune plot and the EnbPI has been fitted
# Fit Model
importlib.reload(sys.modules['util_EnbPI_journal'])
importlib.reload(sys.modules['PI_class_EnbPI_journal'])
train_frac = 0.5
alpha = 0.05
random_forest = RandomForestRegressor(n_estimators=100, criterion='mse',
                                      bootstrap=False, n_jobs=-1)
_, EnbPI_result_RF = util.split_and_train(Data_helix, train_frac=train_frac,
                                          mathcalA=random_forest, alpha=alpha, itrial=0, return_full=True)
fig_helix_RF = plt_helix_with_PI(Data_helix, EnbPI_result_RF, regr_name='Random Forest')

save_fig = True
if save_fig:
    fig_helix_RF.savefig(
        f'Simulation/Helix_RF.pdf', dpi=300, bbox_inches='tight',
        pad_inches=0)


'''Sec 6. Figure 7'''


def change_refit(param, test_style, save_fig, first_run=True):
    # Assume perfect knowledge of change time (T*) and also assume we collect observation for some time (e.g. NOT as long as old T, since we may NOT afford to do so).
    # Basically run EnbPI on two chunks of data and concatenate results
    # YET, the prediction period T*+1,...,T*+T/2 will still be predicted by pre-change model (o/w no PI created for these); we just deal with T*+T/2+1,...,T+T1 with new model
    importlib.reload(sys.modules['utils_EnbPI_journal'])  # For me
    Ttot, stronglymixing, high_dim, curr_fX, curr_fX_post, tseries, current_regr = param[
        test_style]
    change_pts = True
    change_frac = 0.6
    train_frac = 0.3
    if first_run:
        Data_dc_change = util.DGP(curr_fX, curr_fX_post, T_tot=Ttot, tseries=tseries,
                                  high_dim=high_dim, change_points=change_pts, change_frac=change_frac)
        with open(f'Data_changepts_{test_style}.p', 'wb') as fp:
            pickle.dump(Data_dc_change, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'Data_changepts_{test_style}.p', 'rb') as fp:
            Data_dc_change = pickle.load(fp)
    if tseries:
        # Because the first 100 points are not needed for time-seires data
        Ttot -= 100
    Tstar = math.ceil(Ttot * change_frac)  # change point
    # length after change point before refitting
    Tnew = math.ceil(Ttot * train_frac / 10)
    util.quick_plt(Data_dc_change, current_regr, tseries,
                   stronglymixing, change_points=True, args=[Tstar, Tnew])
    Data_dc_prechange = {}
    Data_dc_postchange = {}
    for key in Data_dc_change.keys():
        # Overlap with length Tnew
        Data_dc_prechange[key] = Data_dc_change[key][:Tstar + Tnew]
        Data_dc_postchange[key] = Data_dc_change[key][Tstar:]
    # return [Data_dc_prechange, Data_dc_postchange]
    # e.g. change train_frac so still use Ttot*train_frac to train pre-change model and
    # Ttot*train_frac/2 to train post-change model
    print('Fit Pre-change Model')
    # return_full is TRUE below because we need much information to compute the actual coverage
    regr_prechange = util.split_and_train(Data_dc_prechange, train_frac=(train_frac * Ttot) / (Tstar + Tnew),
                                          mathcalA=current_regr, alpha=0.05, itrial=0, return_full=True)
    print('Fit Post-change Model')
    regr_postchange = util.split_and_train(Data_dc_postchange, train_frac=Tnew / (Ttot - Tstar),
                                           mathcalA=current_regr, alpha=0.05, itrial=0, return_full=True)
    # Merge some fields together
    # Note, because of the T/2 overlap after Tstar, I use POST-change residuals
    # Since they will be plotted later.
    Ttrain = math.ceil(Ttot * train_frac)
    regr_change_refit = regr_prechange.copy()
    # Note, interval centers & ends are ONLY for prediction, so have length T1 (wrt to data being used)
    # Online resid include training LOO ones as well, so have length T+T1 (wrt to data being used)
    regr_change_refit[1].Ensemble_pred_interval_centers = np.append(
        regr_prechange[1].Ensemble_pred_interval_centers, regr_postchange[1].Ensemble_pred_interval_centers)
    regr_change_refit[1].Ensemble_pred_interval_ends = np.r_[
        regr_prechange[1].Ensemble_pred_interval_ends, regr_postchange[1].Ensemble_pred_interval_ends]
    if stronglymixing:
        savename = 'beta_star_stronglymixing.p'
    else:
        savename = 'beta_star_nostronglymixing.p'
    with open(savename, 'rb') as fp:
        beta_star = pickle.load(fp)
    FX = Data_dc_change['f(X)']  # Include training data
    # Only for T+1,...,T+T1
    FXhat = regr_change_refit[1].Ensemble_pred_interval_centers
    # Only for T+1,...,T+T1
    PI = regr_change_refit[1].Ensemble_pred_interval_ends
    # Needed because PI has been concatenated
    PI = pd.DataFrame(PI, columns=['lower', 'upper'])
    Y_predict = Data_dc_change['Y'][math.ceil(len(FX) * train_frac):]
    fig_ptwisewidth = util.EmpvsActual_PtwiseWidth_simple(
        beta_star, 0.05, FX[-len(FXhat):], FXhat, PI, Y_predict, stronglymixing, change_pts=True, args=[Tstar - Ttot, Tnew])
    if save_fig:
        if change_pts:
            string = '_changepts'
        regr_name = current_regr.__class__.__name__
        tse = '_tseries' if tseries else ''
        strongm = '_mixing' if stronglymixing else ''
        fig_ptwisewidth.savefig(
            f'Simulation/EmpvsActual_PtwiseWidth{string}_{regr_name}{tse}{strongm}_simple.pdf', dpi=300, bbox_inches='tight',
            pad_inches=0)
    return [Data_dc_change, regr_change_refit]

    # # NOTE: Below are needed ONLY if we want to examine:
    # # A. Average width
    # # B. Marginal Coverage
    # # Note, we include POST-CHANGE residuals below, NOT because these are used to build prediction intervals for Tstar,...,Tstart+T/2, but because we want to compare CDF/PDF figures with the earlier "change_norefit"
    # regr_change_refit[1].Ensemble_online_resid = np.append(
    #     regr_prechange[1].Ensemble_online_resid[:-Tnew], regr_postchange[1].Ensemble_online_resid)
    # # Update average width in the end,
    # # BUT only look at parts where there are no change points
    # width_pre = np.diff(regr_change_refit[1].Ensemble_pred_interval_ends[:Tstar-Ttrain])
    # width_post = np.diff(regr_change_refit[1].Ensemble_pred_interval_ends[Tstar-int(Tnew/2):])
    # regr_change_refit[0]['width'] = np.mean(np.append(width_pre, width_pre))
    # # Coverage updated took some time, as I was confused how long are prechange and postchange sequence
    # regr_change_refit[0]['coverage'] = ((regr_change_refit[1].Ensemble_pred_interval_centers+regr_change_refit[1].Ensemble_pred_interval_ends[:, 0] <= Data_dc_change['Y'][Ttrain:]) &
    #                                     (regr_change_refit[1].Ensemble_pred_interval_centers+regr_change_refit[1].Ensemble_pred_interval_ends[:, 1] >= Data_dc_change['Y'][Ttrain:])).mean()
    # print(f"After Refitting:\n Mean width is {regr_change_refit[0]['width']}")
    # if return_full:
    #     return regr_change_refit
    # else:
    #     return regr_change_refit[0]


# This dictionary contains parameters "Ttot, stronglymixing, high_dim, curr_fX,curr_fX_post, tseries, current_regr"
importlib.reload(sys.modules['utils_EnbPI_journal'])  # For me
param_changepts = {'linear': [2000, False, False, util.True_mod_linear_pre, util.True_mod_linear_post, False, LinearRegression(fit_intercept=False)],
                   'lasso': [2000, False, True, util.True_mod_lasso_pre, util.True_mod_lasso_post, False, Lasso(alpha=0.01)],
                   'nonlinear': [2000, False, False, util.True_mod_nonlinear_pre, util.True_mod_nonlinear_post, False, util.keras_mod()],
                   'tseries_nostronglymixing': [2100, False, False, util.True_mod_nonlinear_pre, util.True_mod_nonlinear_post, True, util.keras_mod()],
                   'tseries_stronglymixing': [2100, True, False, util.True_mod_nonlinear_pre, util.True_mod_nonlinear_post, True, util.keras_mod()]}
for test_style in param_changepts.keys():
    # test_style = 'linear'
    save_fig = True  # Set as "False" if just trying things out
    first_run = False
    Data_dc_change, regr_change_refit = change_refit(
        param_changepts, test_style, save_fig, first_run)
