import numpy as np
import matplotlib.pyplot as pl


class BasicHMC(object):

    def __init__(self, model=None, verbose=True):
        """A basic HMC sampling object.

        :params model:
            An object with the following methods:
            * lnprob(theta)
            * lnprob_grad(theta)
            * (optional) check_constrained

        :params verbose:
            bool, print lots of junk?
        """
        self.verbose = verbose
        self.model = model
        self.has_bounds = hasattr(self.model, "check_constrained")
        self.set_mass_matrix()

    def lnprob(self, theta):
        return self.model.lnprob(theta)

    def lnprob_grad(self, theta):
        return self.model.lnprob_grad(theta)

    def sample(
        self,
        initial,
        iterations=1,
        epsilon=None,
        mass_matrix=None,
        length=10,
        sigma_length=0.0,
        store_trajectories=False,
    ):
        """Sample for `iterations` trajectories (i.e., compute that many
        trajectories, resampling the momenta at the end of each trajectory).

        :params initial:
            The initial position from which to start the sampling.  ndarray of
            shape (`ndim`,)

        :param iterations:
            The number of trajectories to compute.  Integer.

        :param epsilon: (optional, default: None)
            The stepsize for the leapfrog integrator.  Scalar float or ndarray
            of shape (ndim,). If `None`, a scalar value will be crudely
            estimated.

        :param mass_matrix: (optional, default: None)
            "Masses" in each dimension used to rescale the momentum vectors in
            the HMC trajectories.  Ideally this would be the inverse of the
            covariance matrix of the posterior PDF. If `None` all masses will be
            assumed 1.  Otherwise can be ndarray of shape (ndim,) for a
            diagonal covariance matrix or (ndim, ndim), in which case it must
            be positive semi-definite.

        :param length:
            Number of leapfrog steps to take in each trajectory.  Integer.

        :param sigma_length: (optional, default: 0.0)
            The dispersion in the length of each trajectory.  If greater than
            zero, the length of each trajectory will be drawn from a gaussian
            with mean `length` and dispersion `sigma_length`

        :param store_trajectories:
            If `True`, store not just the endpoints of each trajectory but the
            steps along each trajectory in a `trajectories` attribute.
        """
        self.ndim = len(initial)
        self.store_trajectories = store_trajectories

        # set some initial values
        self.set_mass_matrix(mass_matrix)
        if epsilon is None:
            epsilon = self.find_reasonable_stepsize(initial.copy())
            print("using epsilon = {0}".format(epsilon))
        self.mu = np.log(10 * epsilon)

        # set up the output
        self.reset()
        self.chain = np.zeros([iterations, self.ndim])
        self.lnp = np.zeros([iterations])
        self.accepted = np.zeros([iterations])
        if self.store_trajectories:
            self.trajectories = []

        theta = initial.copy()
        self.traj_num = 0
        # loop over trajectories
        lnp, grad = None, None  # initial P and lnP are unknown
        for i in range(int(iterations)):
            ll = int(
                np.clip(np.round(np.random.normal(length, sigma_length)), 2, np.inf)
            )
            if self.verbose:
                print("eps={:3.8f}, L={:5.0f}".format(epsilon, ll))
            info = self.trajectory(theta, epsilon, ll, lnP0=lnp, grad0=grad)
            theta, lnp, grad, accepted = info
            self.lnp[i] = lnp
            self.chain[i, :] = theta
            self.accepted[i] = accepted
            self.traj_num += 1
        return theta, lnp, grad

    def trajectory(self, theta0, epsilon, length, lnP0=None, grad0=None):
        """Compute one trajectory for a given starting location, epsilon, and
        length.  The momenta in each direction are drawn from a gaussian before
        performing 'length' leapfrog steps.  If the trajectories attribute
        exists, store the path of the trajectory.

        :param theta0:
            Starting position, ndarray of shape (ndim,)

        :param epsilon:
            Stepsize(s) to use for this trajectory.  scalar float or ndarray of shape (ndim,)

        :param length:
            The length of this trajectory, integer.

        :param lnP0: optional
            The lnprob value of the initial position (can be used to save a call to lnprob)

        :param grad0: optional
            The gradients of the lnprob function at `theta0`, ndarray of shape (ndim,)

        :returns theta:
            The final position vector, which if the trajectory was not accepted
            will be equal to the initial position.  ndarray of shape (ndim,)

        :returns lnP:
            The ln-probability at the final position, float.

        :returns grad:
            The gradient of the ln-probability at the final position, ndarray of shape (ndim,)

        :returns accepted:
            Whether the trajectory was accepted (1.0) or not (0.0)
        """
        if self.store_trajectories:
            self.trajectories.append(np.zeros([length, self.ndim]))

        #  --- Set up for the run ----
        # save initial position
        theta = theta0.copy()
        # random initial momenta
        p0 = self.draw_momentum()
        # gradient in U at initial position, negative of gradient lnP
        if grad0 is None:
            grad0 = -self.lnprob_grad(theta0)
        if lnP0 is None:
            lnP0 = self.lnprob(theta0)
        # use copies of initial momenta and gradient
        p, grad = p0.copy(), grad0.copy()

        # --- Compute Trajectory ---
        # do 'length' leapfrog steps along the trajectory (and store?)
        for step in range(int(length)):
            theta, p, grad = self.leapfrog(
                theta, p, epsilon, grad, check_oob=self.has_bounds
            )
            if self.store_trajectories:
                self.trajectories[-1][step, :] = theta

        # ---- Accept/Reject ---
        # Odds ratio of the proposed move
        lnP = self.lnprob(theta)
        # change in potential = negative change in lnP
        dU = lnP0 - lnP
        # change in kinetic
        dK = self.kinetic_energy(p) - self.kinetic_energy(p0)
        # acceptance criterion
        alpha = np.exp(-dU - dK)
        if self.verbose:
            print("H={0}, dU={1}, dK={2}".format(alpha, dU, dK))
        # Accept or reject
        if np.random.uniform(0, 1) < alpha:
            accepted = 1.0
            return theta, lnP, grad, accepted
        else:
            accepted = 0.0
            return theta0, lnP0, grad0, accepted

    def leapfrog(self, q, p, epsilon, grad, check_oob=False):
        """Perfrom one leapfrog step, updating the momentum and position
        vectors. This uses one call to the model.lnprob_grad() function, which
        must be defined. It also performs an optional check on the value of the
        new position to make sure it satistfies any parameter constraints, for
        which the check_constrained method of model is called.
        """

        # half step in p
        p -= 0.5 * epsilon * grad
        # full step in theta
        q += epsilon * self.velocity(p)
        # check for constraints on theta
        while check_oob:
            q, sign, check_oob = self.model.check_constrained(q)
            p *= sign  # flip the momentum if necessary
        # compute new gradient in U, which is negative of gradient in lnP
        grad = -self.lnprob_grad(q)
        # another half step in p
        p -= 0.5 * epsilon * grad
        return q, p, grad

    def draw_momentum(self):
        if self.ndim_mass == 0:
            p = np.random.normal(0, 1, self.ndim)
        elif self.ndim_mass == 1:
            p = np.random.normal(0, np.sqrt(self.mass_matrix))
        else:
            p = np.random.multivariate_normal(np.zeros(self.ndim), self.mass_matrix)
        return p

    def velocity(self, p):
        """Get the velocities given a momentum vector."""
        if self.ndim_mass == 0:
            v = p  # Masses all = 1
        elif self.ndim_mass == 1:
            v = self.inverse_mass_matrix * p
            # v =  p
        else:
            # v = np.dot(self.cho_factor, p)
            v = np.dot(self.inverse_mass_matrix, p)
        return v

    def kinetic_energy(self, p):
        """Get the kinetic energy given momenta"""
        if self.ndim_mass == 0:
            K = np.dot(p, p)
        elif self.ndim_mass == 1:
            K = np.dot(p, self.inverse_mass_matrix * p)
        else:
            K = np.dot(p.T, np.dot(self.inverse_mass_matrix, p))
        return 0.5 * K

    def set_mass_matrix(self, mass_matrix=None):
        """Cache the inverse of the mass matrix, and set a flag for the
        dimensionality of the mass matrix. Instead of flags that control
        operation through branch statements, should probably use subclasses for
        different types of mass matrix.
        """
        self.mass_matrix = mass_matrix
        if mass_matrix is None:
            self.inverse_mass_matrix = 1
            self.ndim_mass = 0
        elif mass_matrix.ndim == 1:
            self.inverse_mass_matrix = 1.0 / mass_matrix
            self.ndim_mass = 1
        elif mass_matrix.ndim == 2:
            self.inverse_mass_matrix = np.linalg.inv(mass_matrix)
            self.ndim_mass = 2
        print(mass_matrix, self.ndim_mass)

    def langevin(self):
        """Special case of length = 1 trajectories"""
        raise (NotImplementedError)

    def find_reasonable_stepsize(self, q0, epsilon_guess=1.0):
        """Estimate a reasonable value of the stepsize"""
        epsilon = epsilon_guess
        lnP0, grad0 = self.lnprob(q0.copy()), self.lnprob_grad(q0.copy())
        p0 = self.draw_momentum()

        condition, a, i = True, 0, 0
        while condition:
            p = p0.copy()
            epsilon = 2.0**a * epsilon
            qprime, pprime, gradprime = self.leapfrog(
                q0.copy(), p, epsilon, grad0, check_oob=self.has_bounds
            )
            lnP = self.lnprob(qprime)
            # change in potential
            dU = lnP0 - lnP
            # change in kinetic
            dK = self.kinetic_energy(pprime) - self.kinetic_energy(p0)
            alpha = np.exp(-dU - dK)
            if a == 0:  # this is the first try
                a = (
                    2 * (alpha > 0.5) - 1.0
                )  # direction to change epsilon in the future, + or -
            condition = (alpha**a) > (2 ** (-a))
            i += 1
            print(i, epsilon, alpha)
            if alpha == 0.0:
                raise ValueError("alpha is 0")
        return epsilon

    def reset(self):
        # use this to keep track of the trajectory number within the trajectory
        # (for storage)
        self.traj_num = 0


class TestModel(object):
    """A simple correlated normal distribution to sample."""

    def __init__(self, Sigma=None):
        if Sigma is None:
            Sigma = np.array([[1.0, 1.8], [1.8, 4.0]])
        self.A = np.linalg.inv(Sigma)
        self.has_constraints = False

    def lnprob_grad(self, theta):
        return -np.dot(self.A, theta)

    def lnprob(self, theta):
        return 0.5 * np.dot(theta.T, self.lnprob_grad(theta))


class MixModel(object):
    """A simple line in 2-d space (but constrained) to sample."""

    def __init__(self):
        self.A = np.array([10.0, 20.0])
        # constraints
        self.lower = 0.0
        self.upper = 10.0

    def model(self, theta):
        # super simple model
        return (self.A * theta).sum()

    def lnprob(self, theta):
        # probability of that simple model given observations (which must be defined)
        return -0.5 * ((self.model(theta) - self.obs) ** 2 / self.obs_unc**2).sum()

    def lnprob_grad(self, theta):
        # with simple gradients of the probbility
        grad = -(self.model(theta) - self.obs) / self.obs_unc**2 * self.A
        return grad

    def check_constrained(self, theta):
        """Method that checks the value of theta against constraints.
        If theta is above or below the boundaries, the sign of the momentum
        is flipped and theta is adjusted as if the trajectory had
        bounced off the constraint. Returns the new theta vector, a
        vector of multiplicative signs for the momenta, and a flag for
        if the values are still out of bounds.
        """

        # initially no flips
        sign = np.ones_like(theta)
        oob = True  # pretend we started out-of-bounds to force at least one check
        # print('theta_in ={0}'.format(theta))
        while oob:
            above = theta > self.upper
            theta[above] = 2 * self.upper - theta[above]
            sign[above] *= -1
            below = theta < self.lower
            theta[below] = 2 * self.lower - theta[below]
            sign[below] *= -1
            oob = np.any(below | above)
            # print('theta_out ={0}'.format(theta))
        return theta, sign, oob


def test_mix_hmc(epsilon=0.2, length=10, iterations=100, snr=10):
    """Sample the mixing model using hmc, and plot the results."""
    model = MixModel()
    D = 2

    # generate the mock
    mock_theta = np.random.uniform(1, 5, D)
    # print('mock_theta={0}'.format(mock_theta))
    mock = model.model(mock_theta)
    noised_mock = mock * (1 + np.random.normal(0, 1, 1) / snr)
    noise = mock / snr

    # add the mock to the model
    model.obs = noised_mock
    model.obs_unc = noise
    theta0 = np.random.uniform(0, 10, D)

    # initialize sampler and sample
    sampler = BasicHMC(model, verbose=False)
    pos, prob, eps = sampler.sample(
        theta0,
        iterations=iterations,
        epsilon=epsilon,
        length=length,
        store_trajectories=True,
    )
    print(mock_theta / (np.mean(pos, axis=0)))
    print("mock_theta = {0}".format(mock_theta))

    # plot trajectories
    pl.figure(1)
    pl.clf()
    color = ["red", "blue"]
    pl.plot(
        sampler.chain[::10, 0], sampler.chain[::10, 1], ".", label="Thinned samples"
    )
    for it in np.arange(20) + int(iterations / 3):
        pl.plot(
            sampler.trajectories[it, :, 0],
            sampler.trajectories[it, :, 1],
            color=color[int(sampler.accepted[it])],
        )
    pl.plot(
        mock_theta[0], mock_theta[1], "g.", markersize=20, label="Truth (noiseless)"
    )
    pl.plot(theta0[0], theta0[1], "c.", markersize=15, label="Initial")
    pl.legend(loc="upper right")
    pl.title(
        r"$Z = \theta^T x$, $\theta >0$, $\epsilon = ${0}, Length = {1}, $f_{{accept}} =$ {2}".format(
            epsilon, length, sampler.accepted.sum() / iterations
        )
    )
    pl.xlabel(r"$\theta_1$")
    pl.ylabel(r"$\theta_2$")

    pl.show()
    return sampler


def test_hmc(verbose=False, Sigma=None, **sample_kwargs):
    """sample the correlated normal using hmc"""
    model = TestModel(Sigma=Sigma)
    D = 2
    theta0 = np.random.normal(0, 1, D)

    sampler = BasicHMC(model, verbose=verbose)
    pos, prob, eps = sampler.sample(theta0.copy(), **sample_kwargs)
    print(theta0)
    print(np.std(sampler.chain, axis=0))
    print(sampler.accepted.mean())
    return sampler
