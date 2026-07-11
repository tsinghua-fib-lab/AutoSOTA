import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import warnings
import torch.nn as nn
import torch.optim as optim
from DeepKnockoffs.mmd import mix_rbf_mmd2_loss
warnings.filterwarnings('ignore')

def covariance_diff_biased(X, Xk, SigmaHat, Mask, scale=1.0):
    """ Second-order loss function, as described in deep knockoffs manuscript
    :param X: input data
    :param Xk: generated knockoffs
    :param SigmaHat: target covariance matrix
    :param Mask: masking the diagonal of Cov(X,Xk)
    :param scale: scaling the loss function
    :return: second-order loss function
    """

    # Center X,Xk
    mX  = X  - torch.mean(X,0,keepdim=True)
    mXk = Xk - torch.mean(Xk,0,keepdim=True)
    # Compute covariance matrices
    SXkXk = torch.mm(torch.t(mXk),mXk)/mXk.shape[0]
    SXXk  = torch.mm(torch.t(mX),mXk)/mXk.shape[0]

    # Compute loss
    T  = (SigmaHat-SXkXk).pow(2).sum() / scale
    T += (Mask*(SigmaHat-SXXk)).pow(2).sum() / scale
    return T

def create_checkpoint_name(pars):
    """ Defines the filename of the network
    :param pars: training hyper-parameters
    :return: filename composed of the hyper-parameters
    """

    checkpoint_name = 'net'
    for key, value in pars.items():
        checkpoint_name += '_' + key
        if key == 'alphas':
            for i in range(len(pars['alphas'])):
                checkpoint_name += '_' + str(pars['alphas'][i])
        else:
            checkpoint_name += '_' + str(value)
    return checkpoint_name

def save_checkpoint(state, filename):
    """ Saves the most updatated network to filename and store the previous
    machine in filename + _prev.pth.tar' file
    :param state: training state of the machine
    :filename: filename to save the current machine
    """

    # keep the previous model
    if os.path.isfile(filename):
        os.rename(filename, filename + '_prev.pth.tar')
    # save new model
    torch.save(state, filename)

def gen_batches(n_samples, batch_size, n_reps):
    """ Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    batches = []
    for rep_id in range(n_reps):
        idx = np.random.permutation(n_samples)
        for i in range(0, math.floor(n_samples/batch_size)*batch_size, batch_size):
            window = np.arange(i,i+batch_size)
            new_batch = idx[window]
            batches += [new_batch]
    return(batches)

class Net(nn.Module):
    """ Deep knockoff network
    """
    def __init__(self, p, dim_h, family="continuous"):
        """ Constructor
        :param p: dimensions of data
        :param dim_h: width of the network (~6 layers are fixed)
        :param family: data type, either "continuous" or "binary"
        """
        super(Net, self).__init__()

        self.p = p
        self.dim_h = dim_h
        if (family=="continuous"):
            self.main = nn.Sequential(
                nn.Linear(2*self.p, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.p),
            )
        elif (family=="binary"):
            self.main = nn.Sequential(
                nn.Linear(2*self.p, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.p),
                nn.Sigmoid(),
                nn.BatchNorm1d(self.p, eps=1e-02),
            )
        else:
            sys.exit("Error: unknown family");

    def forward(self, x, noise):
        """ Sample knockoff copies of the data
        :param x: input data
        :param noise: random noise seed
        :returns the constructed knockoffs
        """
        x_cat = torch.cat((x,noise),1)
        x_cat[:,0::2] = x
        x_cat[:,1::2] = noise
        return self.main(x_cat)

def norm(X, p=2):
    if(p==np.inf):
        return(torch.max(torch.abs(X)))
    else:
        return(torch.norm(X,p))

class KnockoffMachine:
    """ Deep Knockoff machine
    """
    def __init__(self, pars, checkpoint_name=None, logs_name=None):
        """ Constructor
        :param pars: dictionary containing the following keys
                'family': data type, either "continuous" or "binary"
                'p': dimensions of data
                'epochs': number of training epochs
                'epoch_length': number of iterations over the full data per epoch
                'batch_size': batch size
                'test_size': size of test set
                'lr': learning rate for main training loop
                'lr_milestones': when to decrease learning rate, unused when equals to number of epochs
                'dim_h': width of the network
                'target_corr': target correlation between variables and knockoffs
                'LAMBDA': penalty encouraging second-order knockoffs
                'DELTA': decorrelation penalty hyper-parameter
                'GAMMA': penalty for MMD distance
                'alphas': kernel widths for the MMD measure (uniform weights)
        :param checkpoint_name: location to save the machine
        :param logs_name: location to save the logfile
        """
        # architecture parameters
        self.p = pars['p']
        self.dim_h = pars['dim_h']
        self.family = pars['family']

        # optimization parameters
        self.epochs = pars['epochs']
        self.epoch_length = pars['epoch_length']
        self.batch_size = pars['batch_size']
        self.test_size = pars['test_size']
        self.lr = pars['lr']
        self.lr_milestones = pars['lr_milestones']

        # loss function parameters
        self.alphas = pars['alphas']
        self.target_corr = torch.from_numpy(pars['target_corr']).float()
        self.DELTA = pars['DELTA']
        self.GAMMA = pars['GAMMA']
        self.LAMBDA = pars['LAMBDA']

        # noise seed
        self.noise_std = 1.0
        self.dim_noise = self.p

        # higher-order discrepency function
        self.matching_loss = mix_rbf_mmd2_loss
        self.matching_param = self.alphas

        # Normalize learning rate to avoid numerical issues
        self.lr = self.lr / np.max([self.DELTA, self.GAMMA, self.GAMMA, self.LAMBDA, 1.0])

        self.pars = pars
        if checkpoint_name == None:
            self.checkpoint_name = None
            self.best_checkpoint_name = None
        else:
            self.checkpoint_name = checkpoint_name + "_checkpoint.pth.tar"
            self.best_checkpoint_name = checkpoint_name + "_best.pth.tar"

        if logs_name == None:
            self.logs_name = None
        else:
            self.logs_name = logs_name

        self.resume_epoch = 0

        # init the network
        self.net = Net(self.p, self.dim_h, family=self.family)

    def compute_diagnostics(self, X, Xk, noise, test=False):
        """ Evaluates the different components of the loss function
        :param X: input data
        :param Xk: knockoffs of X
        :param noise: allocated tensor that is used to sample the noise seed
        :param test: compute the components of the loss on train (False) or test (True)
        :return diagnostics: a dictionary containing the following keys:
                 'Mean' : distance between the means of X and Xk
                 'Corr-Diag': correlation between X and Xk
                 'Corr-Full: ||Cov(X,X) - Cov(Xk,Xk)||_F^2 / ||Cov(X,X)||_F^2
                 'Corr-Swap: ||M(Cov(X,X) - Cov(Xk,Xk))||_F^2 / ||Cov(X,X)||_F^2
                             where M is a mask that excludes the diagonal
                 'Loss': the value of the loss function
                 'MMD-Full': discrepancy between (X',Xk') and (Xk'',X'')
                 'MMD-Swap': discrepancy between (X',Xk') and (X'',Xk'')_swap(s)
        """
        # Initialize dictionary of diagnostics
        diagnostics = dict()
        if test:
            diagnostics["Data"] = "test"
        else:
            diagnostics["Data"] = "train"

        ##############################
        # Second-order moments
        ##############################

        # Difference in means
        D_mean = X.mean(0) - Xk.mean(0)
        D_mean = (D_mean*D_mean).mean()
        diagnostics["Mean"] = D_mean.data.cpu().item()

        # Center and scale X, Xk
        mX = X - torch.mean(X,0,keepdim=True)
        mXk = Xk - torch.mean(Xk,0,keepdim=True)
        scaleX  = (mX*mX).mean(0,keepdim=True)
        scaleXk = (mXk*mXk).mean(0,keepdim=True)

        # Correlation between X and Xk
        scaleX[scaleX==0] = 1.0   # Prevent division by 0
        scaleXk[scaleXk==0] = 1.0 # Prevent division by 0
        mXs  = mX  / torch.sqrt(scaleX)
        mXks = mXk / torch.sqrt(scaleXk)
        corr = (mXs*mXks).mean()
        diagnostics["Corr-Diag"] = corr.data.cpu().item()

        # Cov(Xk,Xk)
        Sigma = torch.mm(torch.t(mXs),mXs)/mXs.shape[0]
        Sigma_ko = torch.mm(torch.t(mXks),mXks)/mXk.shape[0]
        DK_2 = norm(Sigma_ko-Sigma) / norm(Sigma)
        diagnostics["Corr-Full"] = DK_2.data.cpu().item()

        # Cov(Xk,X) excluding the diagonal elements
        SigIntra_est = torch.mm(torch.t(mXks),mXs)/mXk.shape[0]
        DS_2 = norm(self.Mask*(SigIntra_est-Sigma)) / norm(Sigma)
        diagnostics["Corr-Swap"] = DS_2.data.cpu().item()

        ##############################
        # Loss function
        ##############################
        _, loss_display, mmd_full, mmd_swap = self.loss(X[:noise.shape[0]], Xk[:noise.shape[0]], test=True)
        diagnostics["Loss"]  = loss_display.data.cpu().item()
        diagnostics["MMD-Full"] = mmd_full.data.cpu().item()
        diagnostics["MMD-Swap"] = mmd_swap.data.cpu().item()

        # Return dictionary of diagnostics
        return diagnostics

    def loss(self, X, Xk, test=False):
        """ Evaluates the loss function
        :param X: input data
        :param Xk: knockoffs of X
        :param test: evaluate the MMD, regardless the value of GAMMA
        :return loss: the value of the effective loss function
                loss_display: a copy of the loss variable that will be used for display
                mmd_full: discrepancy between (X',Xk') and (Xk'',X'')
                mmd_swap: discrepancy between (X',Xk') and (X'',Xk'')_swap(s)
        """

        # Divide the observations into two disjoint batches
        n = int(X.shape[0]/2)
        X1,Xk1 = X[:n], Xk[:n]
        X2,Xk2 = X[n:(2*n)], Xk[n:(2*n)]

        # Joint variables
        Z1 = torch.cat((X1,Xk1),1)
        Z2 = torch.cat((Xk2,X2),1)
        Z3 = torch.cat((X2,Xk2),1).clone()
        swap_inds = np.where(np.random.binomial(1,0.5,size=self.p))[0]
        Z3[:,swap_inds] = Xk2[:,swap_inds]
        Z3[:,swap_inds+self.p] = X2[:,swap_inds]

        # Compute the discrepancy between (X,Xk) and (Xk,X)
        mmd_full = 0.0
        # Compute the discrepancy between (X,Xk) and (X,Xk)_s
        mmd_swap = 0.0
        if(self.GAMMA>0 or test):
            # Evaluate the MMD by following section 4.3 in
            # Li et al. "Generative Moment Matching Networks". Link to
            # the manuscript -- https://arxiv.org/pdf/1502.02761.pdf
            mmd_full = self.matching_loss(Z1, Z2, self.matching_param)
            mmd_swap = self.matching_loss(Z1, Z3, self.matching_param)

        # Match first two moments
        loss_moments = 0.0
        if self.LAMBDA>0:
            # First moment
            D_mean = X.mean(0) - Xk.mean(0)
            loss_1m = D_mean.pow(2).sum()
            # Second moments
            loss_2m = covariance_diff_biased(X, Xk, self.SigmaHat, self.Mask, scale=self.Sigma_norm)
            # Combine moments
            loss_moments = loss_1m + loss_2m

        # Penalize correlations between variables and knockoffs
        loss_corr = 0.0
        if self.DELTA>0:
            # Center X and Xk
            mX  = X  - torch.mean(X,0,keepdim=True)
            mXk = Xk - torch.mean(Xk,0,keepdim=True)
            # Correlation between X and Xk
            eps = 1e-3
            scaleX  = mX.pow(2).mean(0,keepdim=True)
            scaleXk = mXk.pow(2).mean(0,keepdim=True)
            mXs  = mX / (eps+torch.sqrt(scaleX))
            mXks = mXk / (eps+torch.sqrt(scaleXk))
            corr_XXk = (mXs*mXks).mean(0)
            loss_corr = (corr_XXk-self.target_corr).pow(2).mean()

        # Combine the loss functions
        loss = self.GAMMA*mmd_full + self.GAMMA*mmd_swap + self.LAMBDA*loss_moments + self.DELTA*loss_corr
        loss_display = loss
        return loss, loss_display, mmd_full, mmd_swap


    def train(self, X_in, resume = False):
        """ Fit the machine to the training data
        :param X_in: input data
        :param resume: proceed the training by loading the last checkpoint
        """

        # Divide data into training/test set
        X = torch.from_numpy(X_in[self.test_size:]).float()
        if(self.test_size>0):
            X_test = torch.from_numpy(X_in[:self.test_size]).float()
        else:
            X_test = torch.zeros(0, self.p)

        # used to compute statistics and diagnostics
        self.SigmaHat = np.cov(X,rowvar=False)
        self.SigmaHat = torch.from_numpy(self.SigmaHat).float()
        self.Mask = torch.ones(self.p, self.p) - torch.eye(self.p)

        # allocate a matrix for the noise realization
        noise = torch.zeros(self.batch_size,self.dim_noise)
        noise_test = torch.zeros(X_test.shape[0],self.dim_noise)
        use_cuda = torch.cuda.is_available()

        if resume == True:  # load the last checkpoint
            self.load(self.checkpoint_name)
            self.net.train()
        else:  # start learning from scratch
            self.net.train()
            # Define the optimization method
            self.net_optim = optim.SGD(self.net.parameters(), lr = self.lr, momentum=0.9)
            # Define the scheduler
            self.net_sched = optim.lr_scheduler.MultiStepLR(self.net_optim, gamma=0.1,
                                                            milestones=self.lr_milestones)

        # bandwidth parameters of the Gaussian kernel
        self.matching_param = self.alphas

        # move data to GPU if available
        if use_cuda:
            self.SigmaHat = self.SigmaHat.cuda()
            self.Mask = self.Mask.cuda()
            self.net = self.net.cuda()
            X = X.cuda()
            X_test = X_test.cuda()
            noise = noise.cuda()
            noise_test = noise_test.cuda()
            self.target_corr = self.target_corr.cuda()

        Xk = 0*X
        self.Sigma_norm = self.SigmaHat.pow(2).sum()
        self.Sigma_norm_cross = (self.Mask*self.SigmaHat).pow(2).sum()

        # Store diagnostics
        diagnostics = pd.DataFrame()
        losses_test = []

        # main training loop
        for epoch in range(self.resume_epoch, self.epochs):
            # prepare for training phase
            self.net.train()
            # update the learning rate scheduler
            self.net_sched.step()
            # divide the data into batches
            batches = gen_batches(X.size(0), self.batch_size, self.epoch_length)

            losses = []
            losses_dist_swap = []
            losses_dist_full = []

            for batch in batches:
                # Extract data for this batch
                X_batch  = X[batch,:]

                self.net_optim.zero_grad()

                # Run the network
                Xk_batch = self.net(X_batch, self.noise_std*noise.normal_())

                # Compute the loss function
                loss, loss_display, mmd_full, mmd_swap = self.loss(X_batch, Xk_batch)

                # Compute the gradient
                loss.backward()

                # Take a gradient step
                self.net_optim.step()

                # Save history
                losses.append(loss_display.data.cpu().item())
                if self.GAMMA>0:
                    losses_dist_swap.append(mmd_swap.data.cpu().item())
                    losses_dist_full.append(mmd_full.data.cpu().item())

                # Save the knockoffs
                Xk[batch, :] = Xk_batch.data

            ##############################
            # Compute diagnostics
            ##############################

            # Prepare for testing phase
            self.net.eval()

            # Evaluate the diagnostics on the training data, the following
            # function recomputes the loss on the training data
            diagnostics_train = self.compute_diagnostics(X, Xk, noise, test=False)
            diagnostics_train["Loss"] = np.mean(losses)
            if(self.GAMMA>0 and self.GAMMA>0):
                diagnostics_train["MMD-Full"] = np.mean(losses_dist_full)
                diagnostics_train["MMD-Swap"] = np.mean(losses_dist_swap)
            diagnostics_train["Epoch"] = epoch
            diagnostics = diagnostics.append(diagnostics_train, ignore_index=True)

            # Evaluate the diagnostics on the test data if available
            if(self.test_size>0):
                Xk_test = self.net(X_test, self.noise_std*noise_test.normal_())
                diagnostics_test = self.compute_diagnostics(X_test, Xk_test, noise_test, test=True)
            else:
                diagnostics_test = {key:np.nan for key in diagnostics_train.keys()}
            diagnostics_test["Epoch"] = epoch
            diagnostics = diagnostics.append(diagnostics_test, ignore_index=True)

            # If the test loss is at a minimum, save the machine to
            # the location pointed by best_checkpoint_name
            losses_test.append(diagnostics_test["Loss"])
            if((self.test_size>0) and (diagnostics_test["Loss"] == np.min(losses_test)) and \
               (self.best_checkpoint_name is not None)):
                best_machine = True
                save_checkpoint({
                    'epochs': epoch+1,
                    'pars'  : self.pars,
                    'state_dict': self.net.state_dict(),
                    'optimizer' : self.net_optim.state_dict(),
                    'scheduler' : self.net_sched.state_dict(),
                }, self.best_checkpoint_name)
            else:
                best_machine = False

            ##############################
            # Print progress
            ##############################
            if(self.test_size>0):
                print("[%4d/%4d], Loss: (%.4f, %.4f)" %
                      (epoch + 1, self.epochs, diagnostics_train["Loss"], diagnostics_test["Loss"]), end=", ")
                print("MMD: (%.4f,%.4f)" %
                      (diagnostics_train["MMD-Full"]+diagnostics_train["MMD-Swap"], 
                       diagnostics_test["MMD-Full"]+diagnostics_test["MMD-Swap"]), end=", ")
                print("Cov: (%.3f,%.3f)" %
                      (diagnostics_train["Corr-Full"]+diagnostics_train["Corr-Swap"], 
                       diagnostics_test["Corr-Full"]+diagnostics_test["Corr-Swap"]), end=", ")
                print("Decorr: (%.3f,%.3f)" %
                      (diagnostics_train["Corr-Diag"], diagnostics_test["Corr-Diag"]), end="")
                if best_machine:
                    print(" *", end="")
            else:
                print("[%4d/%4d], Loss: %.4f" %
                      (epoch + 1, self.epochs, diagnostics_train["Loss"]), end=", ")
                print("MMD: %.4f" %
                      (diagnostics_train["MMD-Full"] + diagnostics_train["MMD-Swap"]), end=", ")
                print("Cov: %.3f" %
                      (diagnostics_train["Corr-Full"] + diagnostics_train["Corr-Swap"]), end=", ")
                print("Decorr: %.3f" %
                      (diagnostics_train["Corr-Diag"]), end="")
                
            print("")
            sys.stdout.flush()

            # Save diagnostics to logfile
            if self.logs_name is not None:
                diagnostics.to_csv(self.logs_name, sep=" ", index=False)

            # Save the current machine to location checkpoint_name
            if self.checkpoint_name is not None:
                save_checkpoint({
                    'epochs': epoch+1,
                    'pars'  : self.pars,
                    'state_dict': self.net.state_dict(),
                    'optimizer' : self.net_optim.state_dict(),
                    'scheduler' : self.net_sched.state_dict(),
                }, self.checkpoint_name)

    def load(self, checkpoint_name):
        """ Load a machine from a stored checkpoint
        :param checkpoint_name: checkpoint name of a trained machine
        """
        filename = checkpoint_name + "_checkpoint.pth.tar"

        flag = 1
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            sys.stdout.flush()
            try:
                checkpoint = torch.load(filename, map_location='cpu')
            except:
                print("error loading saved model, trying the previous version")
                sys.stdout.flush()
                flag = 0

            if flag == 0:
                try:
                    checkpoint = torch.load(filename + '_prev.pth.tar', map_location='cpu')
                    flag = 1
                except:
                    print("error loading prev model, starting from scratch")
                    sys.stdout.flush()
                    flag = 0
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            sys.stdout.flush()
            flag = 0

        if flag == 1:
                self.net.load_state_dict(checkpoint['state_dict'])
                if torch.cuda.is_available():
                    self.net = self.net.cuda()

                self.net_optim = optim.SGD(self.net.parameters(), lr = self.lr, momentum=0.9)
                self.net_optim.load_state_dict(checkpoint['optimizer'])
                self.net_sched = optim.lr_scheduler.MultiStepLR(self.net_optim, gamma=0.1,
                                                                milestones=self.lr_milestones)
                self.resume_epoch = checkpoint['epochs']

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(filename, checkpoint['epochs']))
                sys.stdout.flush()
        else:
            self.net.train()
            self.net_optim = optim.SGD(self.net.parameters(), lr = self.lr, momentum=0.9)
            self.net_sched = optim.lr_scheduler.MultiStepLR(self.net_optim, gamma=0.1,
                                                            milestones=self.lr_milestones)

            self.resume_epoch = 0

    def generate(self, X_in):
        """ Generate knockoff copies
        :param X_in: data samples
        :return Xk: knockoff copy per each sample in X
        """

        X = torch.from_numpy(X_in).float()
        self.net = self.net.cpu()
        self.net.eval()

        # Run the network in evaluation mode
        Xk = self.net(X, self.noise_std*torch.randn(X.size(0),self.dim_noise))
        Xk = Xk.data.cpu().numpy()

        return Xk
