"""mountaincarcont_agents.py: Functions and MRP wrapper class for Gymnasium MountainCarContinuous-v0 environment compatible agents with different approximators.
"""
import copy
import torch
import itertools
import types
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from gymnasium.wrappers import TransformObservation
from algorithms import mlp_basis
from algorithms import multivariate_polynomial_basis as multivarpoly


__author__ = "Hannes Unger"
__version__ = "1.0"
__email__ = "hannes.unger@fh-salzburg.ac.at"


# Constants
MIN_SIGMA = 1e-12


def plot(x,y,z, ax=None, title=None, xlabel=None, ylabel=None, zlabel=None):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    ax.plot_surface(x,y,z, alpha=0.75, rstride=1, cstride=1, color='orangered', edgecolors='k', lw=0.6)
    ax.set_box_aspect(aspect=None, zoom=0.8)


def plot_heatmap(mrp, ax=None, fig=None, title=None, dim=2, unnormalize=True, unnormalized_upper=[0.6, 0.07], unnormalized_lower=[-1.2, -0.07], trajectory=None, trajectory_label=None, actionbar=True, xlim=(-1.2, 0.45), fontsize=14, fontsize_title=20, xticks=None, yticks=None, hideyticks=True, yticksright=False):
    if dim != 2:
        raise Exception('Not implemented.')
    if not ax:
        raise Exception('Need axis object.')

    if unnormalize:
        x = np.linspace(unnormalized_lower[0], unnormalized_upper[0], 100)
        y = np.linspace(unnormalized_lower[1], unnormalized_upper[1], 100)
    else:
        # TrainableMountainCarMRPWrapper sets observation_space bounds to normalized bounds on initialization
        x = np.linspace(mrp.env.observation_space.low[0], mrp.env.observation_space.high[0], 100)
        y = np.linspace(mrp.env.observation_space.low[1], mrp.env.observation_space.high[1], 100)

    X, Y = np.meshgrid(x, y)
    states = np.vstack([X.ravel(), Y.ravel()]).T

    # Predict actions for each state
    if unnormalize:
        actions = np.array([mrp.agent.select_action(mrp.normalize(state, max_value=np.array(unnormalized_upper), min_value=np.array(unnormalized_lower)))[0] for state in states])
    else:
        actions = np.array([mrp.agent.select_action(state)[0] for state in states])

    # Reshape to match the grid for heatmap plotting
    Z = actions.reshape(X.shape)

    levels=np.linspace(-1.0, 1.0, 21)

    # Plot the heatmap
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")  # Use a colormap like 'viridis' or 'plasma'

    if fig and actionbar:
        fig.colorbar(contourf, label="Action Magnitude")

    contour = ax.contour(X, Y, Z, levels=levels, cmap="viridis")  # Adjust levels for granularity

    # Add a specific contour for the 0.0 level with custom styling
    zero_contour = ax.contour(X, Y, Z, levels=[0.0], colors="red", linewidths=2.5, zorder=5)

    if trajectory:
        if unnormalize:
            t = [mrp.unnormalize(state, max_value=np.array(unnormalized_upper), min_value=np.array(unnormalized_lower)) for state in trajectory]
        else:
            t = trajectory
        
        ax.plot([row[0] for row in t], [row[1] for row in t], label=trajectory_label, color='white', zorder=4)
        ax.legend(loc='best', fontsize=fontsize)

    ax.set_xlabel(r"$x$", fontsize=fontsize+5)
    if not hideyticks:       
        ax.set_ylabel(r"$\dot{x}$", fontsize=fontsize+5)
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlim(xlim)
    ax.tick_params(axis='both', labelsize=fontsize)
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
        if yticksright:
            ax.yaxis.tick_right()
    if hideyticks:
        ax.tick_params(axis='y', which='both', labelleft=False)


def average_reward_history(values):
    s_mean = []
    for i in range(len(values[0])):
        episode = [t[i] for t in values]
        s_mean.append(sum(episode)/len(values)) # Calculate mean across runs
    return s_mean


class TrainableContinuousMRPWrapper:
    """
    A wrapper to convert gymnasium env MDP into an MRP by applying a fixed policy offering training algorithms.
    Supports n-dimensional continuous observation spaces with one-dimensional continuous action space.

    Attributes:
        env: The environment to wrap.
        policy (multivarpoly.bivar_power_basis): Function mapping [pos, vel] state to real action value
    """

    def __init__(self, env, degree=2, basis='chebyshev', initialization='constant', initial_sigma=0.2, normalize_observations=True, mu_coeffs=None, sigma_coeffs=None, critic_coeffs=None, mlp_n_input_nodes=2, mlp_n_hidden_nodes=4, mlp_n_output_nodes=1, net_arch=None):
        if normalize_observations:
            new_max = 1.0
            new_min = -1.0

            low_state = np.array(
                [new_min for e in env.observation_space.low], dtype=np.float32
            )
            high_state = np.array(
                [new_max for e in env.observation_space.high], dtype=np.float32
            )
            observation_space = spaces.Box(
                low=low_state, high=high_state, dtype=np.float32
            )

            self.env = TransformObservation(env, lambda obs: self.normalize(obs, max_value=env.observation_space.high, min_value=env.observation_space.low, new_max=new_max, new_min=new_min), observation_space)
        else:
            self.env = env
        if basis == 'mlp':
            self.agent = MLPAgent(n_input_nodes=mlp_n_input_nodes, n_hidden_nodes=mlp_n_hidden_nodes, n_output_nodes=mlp_n_output_nodes, initialization=initialization, initial_sigma=initial_sigma, mu_params=mu_coeffs, sigma_params=sigma_coeffs, critic_params=critic_coeffs, net_arch=net_arch)
        else:
            self.agent = PolyAgent(degree=degree, dim=env.observation_space._shape[0], basis=basis, initialization=initialization, initial_sigma=initial_sigma, mu_coeffs=mu_coeffs, sigma_coeffs=sigma_coeffs, critic_coeffs=critic_coeffs)
        self.normalize_observations = normalize_observations

    def render(self):
        self.env.render()

    def reset(self, options={}):
        """Resets the environment to its initial state."""
        return self.env.reset(options=options)

    def step(self, current_state, sigma=MIN_SIGMA):
        """
        Takes a step in the environment using the action defined by the fixed policy for the current state.

        Returns:
            A tuple of (next_state, reward, done, info), where 'info' contains the probability of the transition.
        """
        action, prob = self.agent.select_action(current_state, sigma)  # Sample action according to current policy
        if torch.is_tensor(action):
            a = action.item() # Gymnasium enviornment does weird stuff when passing tensors
        else:
            a = action
        next_state, reward, terminated, truncated, info = self.env.step([a])

        return next_state, reward, terminated, truncated, info, action, prob

    def train(self, alpha_mu=0.01, alpha_sigma=0.01, alpha_critic=0.01, epochs=10, discount=0.9, mu_optimizer=None, sigma_optimizer=None, critic_optimizer=None, method='reinforce',
              learning_history=None, steps_history=None, coeffs_history=None, loss_history=None, verbose=True):
        if method == 'reinforce':
            return self.agent.reinforce(mrp=self, alpha_mu=alpha_mu, alpha_sigma=alpha_sigma, epochs=epochs, discount=discount,
                                        learning_history=learning_history, steps_history=steps_history, coeffs_history=coeffs_history, loss_history=loss_history, verbose=verbose)
        elif method == 'reinforce_autodiff':
            return self.agent.reinforce(mrp=self, alpha_mu=alpha_mu, alpha_sigma=alpha_sigma, epochs=epochs, discount=discount,
                                        learning_history=learning_history, steps_history=steps_history, coeffs_history=coeffs_history, loss_history=loss_history, autodiff=True, mu_optimizer=mu_optimizer, sigma_optimizer=sigma_optimizer, verbose=verbose)
        else:
            raise Exception("Not implemented.")

    def normalize(self, value, max_value, min_value, new_max=1.0, new_min=-1.0, flip=True):
        """
        Normalize a single value to a specified range [new_min, new_max].

        Args:
        - value: The value to be normalized.
        - min_value: The minimum value of the original range.
        - max_value: The maximum value of the original range.
        - new_min: The minimum value of the new range.
        - new_max: The maximum value of the new range.
        - flip: If True, the axis is flipped. This mainly is there for compatibility to experiments of our mountain car paper.

        Returns:
        - The normalized value.
        """
        if flip:
            return ((value - min_value) / (max_value - min_value)) * (new_min - new_max) + new_max  
        else:
            return ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min 


    def unnormalize(self, value, max_value, min_value, new_max=1.0, new_min=-1.0, flip=True):
        """
        Unnormalize a single value given a specified range [new_min, new_max].

        Args:
        - value: The value to be unnormalized.
        - min_value: The minimum value of the original range.
        - max_value: The maximum value of the original range.
        - new_min: The minimum value of the new range, the value was normalized to.
        - new_max: The maximum value of the new range, the value was normalized to.
        - flip: If True, the axis is flipped back in case it was reverted during
        normalization. This mainly is there for compatibility to experiments
        of our mountain car paper.

        Returns:
        - The unnormalized value.
        """
        if flip:
            return (value-new_max)/(new_min-new_max)*(max_value-min_value)+min_value
        else:
            return (value-new_min)/(new_max-new_min)*(max_value-min_value)+min_value

class GaussianAgent:
    def prob(self, state, action):
        return torch.distributions.Normal(loc=self.evaluate_mu_at(state), scale=self.evaluate_sigma_at(state)).cdf(action)  # cdf gives prob that random variable takes value up to including action

    def log_prob_density(self, state, action):
        return torch.distributions.Normal(loc=self.evaluate_mu_at(state), scale=self.evaluate_sigma_at(state)).log_prob(action)

    def select_action(self, observation, sigma=MIN_SIGMA):
        action = self.sample_action(observation, sigma)
        prob = torch.distributions.Normal(loc=self.mu_approximator.evaluate_point(observation), scale=sigma).cdf(action) #cdf gives prob that random variable takes value up to including action
        return action, prob

    def sample_action(self, x, sigma):
        return torch.distributions.Normal(loc=self.mu_approximator.evaluate_point(x), scale=sigma).sample() # self.actor.evaluate_point(x) gives mu

    def evaluate(self, *data, sigma=0.0):
        # return np.array([[self.sample_action([x,y], sigma) for x in data_x] for y in data_y])

        # Generate all combinations of values from the input data arrays.
        # itertools.product(A, B) returns the same as: ((x,y) for x in A for y in B).
        combinations = itertools.product(*data)  
        # Apply the sample_action function to each combination
        results = [self.sample_action(list(comb), sigma) for comb in combinations]
        # Reshape the results to match the shape of the input data grids
        result_shape = [len(d) for d in data]
        return np.array(results).reshape(result_shape)

    def evaluate_mu_at(self, x):
        return self.mu_approximator.evaluate_point(x)

    def evaluate_mu(self, *data):
        return self.mu_approximator.evaluate(*data)

    def evaluate_sigma_at(self, x):
        return torch.exp(self.sigma_approximator.evaluate_point(x)) # See S&B p.336, sigma should always stay positive

    def evaluate_sigma(self, *data):
        return torch.exp(self.sigma_approximator.evaluate(*data)) # See S&B p.336, sigma should always stay positive

    def evaluate_critic_at(self, x):
        return self.critic_approximator.evaluate_point(x)

    def evaluate_critic(self, *data):
        return self.critic_approximator.evaluate(*data)

    def compute_return(self, rewards, gamma, i):
        '''
        Compute the return at time step i.

        Args:
            - rewards: Array of complete episode rewards
            - gamma: Discount factor
            - i: Current position in array
        '''
        g = 0.0
        t = 0 # t=0 for current time step, future time steps should count less, immediate rewards more
        for j in range(i, len(rewards)): # Move from current time step to end of array (Monte Carlo)
            g = torch.add(g, torch.multiply(rewards[j], gamma**t))
            t += 1
        return g

    def score_mu_approximator(self, state, action):
        '''The score function is the derivative of the logarithm of a parameterized probability'''
        return (1/(self.evaluate_sigma_at(state)**2))*(action-self.evaluate_mu_at(state))

    def score_sigma_approximator(self, state, action):
        '''The score function is the derivative of the logarithm of a parameterized probability'''
        return (((action - self.evaluate_mu_at(state))**2)/(self.evaluate_sigma_at(state)**2))-1

    def get_optimizer(self, name='adam', learning_rate=0.001, coeffs=None):
        if isinstance(coeffs, types.GeneratorType):
            c = coeffs # mlp
        else:
            c = [coeffs] # polynomial

        if name is None:
            opt = torch.optim.Adam(c, lr=learning_rate)
        elif name == "adam":
            opt = torch.optim.Adam(c, lr=learning_rate)
        elif name == "adam-amsgrad":
            opt = torch.optim.Adam(c, lr=learning_rate, amsgrad=True)
        elif name == "adamw":
            opt = torch.optim.AdamW(c, lr=learning_rate)
        elif name == "adamw-amsgrad":
            opt = torch.optim.AdamW(c, lr=learning_rate, amsgrad=True)
        elif name == "adamax":
            opt = torch.optim.Adamax(c, lr=learning_rate)
        elif name == "asgd":
            opt = torch.optim.ASGD(c, lr=learning_rate)
        elif name == "lbfgs":
            opt = torch.optim.LBFGS(c, lr=learning_rate)
        elif name == "nadam":
            opt = torch.optim.NAdam(c, lr=learning_rate)
        elif name == "radam":
            opt = torch.optim.RAdam(c, lr=learning_rate)
        elif name == "rmsprop":
            opt = torch.optim.RMSprop(c, lr=learning_rate)
        elif name == "rprop":
            opt = torch.optim.Rprop(c, lr=learning_rate)
        elif name == "sgd":
            opt = torch.optim.SGD(c, lr=learning_rate)
        elif name == "sgd-momentum":
            opt = torch.optim.SGD(c, lr=learning_rate, momentum=0.9)
        elif name == "sgd-nesterov":
            opt = torch.optim.SGD(c, lr=learning_rate, momentum=0.9, nesterov=True)
        else:
            raise Exception(f'Optimizer {name} not implemented')
        return opt

    def plot_me(self, title=None, critic=False, ax=None, heatmap=False):
        if self.dim != 2:
            raise Exception('Not implemented.')
        create_axes = False
        if not ax:
            create_axes = True
            if critic:
                fig = plt.figure(figsize=(21, 7))
            else:
                fig = plt.figure(figsize=(14, 7))

            ax = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            if critic:
                ax3 = fig.add_subplot(133, projection='3d')

            fig.suptitle(title)

        xss = np.linspace(-1, 1, 20)
        yss = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(xss, yss)

        if heatmap:
            im = ax.imshow(self.evaluate_mu(xss, yss))
            ax.set_title(r'$\mu$')
            ax.set_xlabel('position')
            ax.set_ylabel('velocity')
            fig.colorbar(im, cax=ax, orientation='vertical')
        else:                
            plot(X, Y, self.evaluate_mu(xss, yss),
                title=r'$\mu$', xlabel='position', ylabel='velocity',
                zlabel=r'$\mu(s)$', ax=ax)
            if create_axes:
                if heatmap:
                    ax2.imshow(self.evaluate_sigma(xss, yss))
                    ax2.set_title(r'$\sigma$')
                    ax2.set_xlabel('position')
                    ax2.set_ylabel('velocity')
                else: 
                    plot(X, Y, self.evaluate_sigma(xss, yss),
                        title=r'$\sigma$', xlabel='position', ylabel='velocity',
                        zlabel=r'$\sigma(s)$', ax=ax2)
                if critic:
                    if heatmap:
                        ax3.imshow(self.evaluate_critic(xss, yss))
                        ax3.set_title('critic')
                        ax3.set_xlabel('position')
                        ax3.set_ylabel('velocity')
                    else: 
                        plot(X, Y, self.evaluate_critic(xss, yss),
                        title=r'critic', xlabel='position', ylabel='velocity',
                        zlabel=r'$v(s)$', ax=ax3)


class MLPAgent(GaussianAgent):
    def __init__(self, n_input_nodes=2, n_hidden_nodes=4, n_output_nodes=1, initialization='constant', initial_sigma=0.2, mu_params=None, sigma_params=None, critic_params=None, net_arch=None):
        self.dim = n_input_nodes
        if not net_arch:
            self.mu_approximator = mlp_basis.SingleHiddenLayerMLPApproximator(n_input=n_input_nodes, n_hidden_nodes=n_hidden_nodes, n_output=n_output_nodes, initialization=initialization, params=mu_params)
            self.sigma_approximator = mlp_basis.SingleHiddenLayerMLPApproximator(n_input=n_input_nodes, n_hidden_nodes=n_hidden_nodes, n_output=n_output_nodes, initialization='flat', flat_init_offset=initial_sigma, params=sigma_params)
            self.critic_approximator = mlp_basis.SingleHiddenLayerMLPApproximator(n_input=n_input_nodes, n_hidden_nodes=n_hidden_nodes, n_output=n_output_nodes, initialization=initialization, params=critic_params)           
        else:
            self.mu_approximator = mlp_basis.MLPApproximator(n_input=n_input_nodes, n_output=n_output_nodes, net_arch=net_arch, params=mu_params)
            self.sigma_approximator = mlp_basis.MLPApproximator(n_input=n_input_nodes, n_output=n_output_nodes, net_arch=net_arch, initialization='flat', flat_init_offset=initial_sigma, params=sigma_params)
            self.critic_approximator = mlp_basis.MLPApproximator(n_input=n_input_nodes, n_output=n_output_nodes, net_arch=net_arch, params=critic_params)       

        torch.set_grad_enabled(False) # Disable tracking gradients and only enable it during training for better performance

    def evaluate_sigma_at(self, x):
        return torch.exp(self.sigma_approximator.evaluate_point(x)) # See S&B p.336, sigma should always stay positive

    def evaluate_sigma(self, *data):
        return torch.exp(self.sigma_approximator.evaluate(*data)) # See S&B p.336, sigma should always stay positive

    def reinforce(self, mrp, alpha_mu, alpha_sigma, epochs=10, discount=0.9,
                  learning_history=None, steps_history=None, coeffs_history=None, loss_history=None, autodiff=True, mu_optimizer=None, sigma_optimizer=None, verbose=True):
        '''
        REINFORCE algorithm estimating policy according to S&B p.328.
        :param mrp: Wrapped Markov Reward Process
        :param alpha: Learning Rate
        :param epochs: Number of episodes to train for
        :param discount: gamma
        :return:
        '''

        if not autodiff:
            raise Exception('Not implemented.')

        if steps_history is None:
            steps_history = []
        if learning_history is None:
            learning_history = []
        if coeffs_history is None:
            coeffs_history = []
        if loss_history is None:
            loss_history = []

        mu_optimizer = self.get_optimizer(name=mu_optimizer, learning_rate=alpha_mu, coeffs=self.mu_approximator.model.parameters())
        sigma_optimizer = self.get_optimizer(name=sigma_optimizer, learning_rate=alpha_sigma, coeffs=self.sigma_approximator.model.parameters())

        for episode in range(epochs):
            state = mrp.reset()[0]
            sigma = self.evaluate_sigma_at(state).item()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_probs = []

            s = 0
            # Generate an episode
            while True:
                s += 1
                next_state, reward, terminated, truncated, _, action, prob = mrp.step(state, sigma)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_probs.append(prob)
                state = copy.deepcopy(next_state)
                sigma = self.evaluate_sigma_at(state).item()
                if terminated or truncated:
                    break

            total_t = len(episode_rewards)

            for t, (state, action) in enumerate(zip(episode_states, episode_actions)):                        
                g = self.compute_return(episode_rewards, discount, t)
                def closure(): # second order optimizers like lbfgs require a closure to run loss computation multiple times
                    with torch.enable_grad():
                        mu_optimizer.zero_grad()
                        sigma_optimizer.zero_grad()
                        weight = total_t - t
                        loss = -torch.multiply(torch.multiply(self.log_prob_density(state, action), g), weight) # weight more recent actions (=end of episode) higher
                        loss.backward()
                        return loss
                loss = mu_optimizer.step(closure=closure)
                sigma_optimizer.step(closure=closure)
                loss_history.append(loss.item())    
            cumulated_reward = np.sum(episode_rewards)
            learning_history.append(cumulated_reward)
            steps_history.append(s)
            #coeffs_history.append(copy.deepcopy((self.mu_approximator.coeffs.numpy())))

            if verbose:
                print(f"Episode {episode + 1}: Cumulated Reward: {cumulated_reward}\r", end="")

        return learning_history, loss_history, steps_history#, coeffs_history


class PolyAgent(GaussianAgent):
    def __init__(self, dim=2, degree=2, basis='chebyshev', initialization='constant', initial_sigma=0.2, mu_coeffs=None, sigma_coeffs=None, critic_coeffs=None):
        self.dim = dim
        self.degree = degree
        self.mu_approximator = multivarpoly.MultiVarPoly(dim=dim, degree=degree, basis=basis, initialization=initialization, coeffs=mu_coeffs)
        self.sigma_approximator = multivarpoly.MultiVarPoly(dim=dim, degree=degree, basis=basis, initialization='flat', flat_init_offset=initial_sigma, coeffs=sigma_coeffs)
        self.critic_approximator = multivarpoly.MultiVarPoly(dim=dim, degree=degree, basis=basis, initialization='flat', flat_init_offset=0.0, coeffs=critic_coeffs)            

        torch.set_grad_enabled(False) # Disable tracking gradients and only enable it during training for better performance

    def grad_score_mu_approximator(self, state, action):
        '''The gradient at the current state multiplied by the score at the current state'''
        grads = torch.tensor(self.mu_approximator.evaluate_basis_vectors_at(state))
        #grads = tf.clip_by_value(g, -1.0, 1.0)
        return torch.multiply(self.score_mu_approximator(state, action), grads) # S&B p.336

    def grad_score_sigma_approximator(self, state, action):
        '''The gradient at the current state multiplied by the score at the current state'''
        grads = torch.tensor(self.sigma_approximator.evaluate_basis_vectors_at(state))
        #grads = tf.clip_by_value(g, -1.0, 1.0)
        return torch.multiply(self.score_sigma_approximator(state, action), grads) # S&B p.336

    def reinforce_update_weights(self, alpha_mu, alpha_sigma, discount, g, state, action, weights_mu, weights_sigma, t, total_t):
        weight = total_t - t # weight more recent actions (=end of episode) higher
        weights_mu += alpha_mu*discount**weight*g*self.grad_score_mu_approximator(state, action)
        weights_sigma += alpha_sigma*discount**weight*g*self.grad_score_sigma_approximator(state, action)

    def reinforce(self, mrp, alpha_mu, alpha_sigma, epochs=10, discount=0.9,
                  learning_history=None, steps_history=None, coeffs_history=None, loss_history=None, autodiff=False, mu_optimizer=None, sigma_optimizer=None, verbose=True):
        '''
        REINFORCE algorithm estimating policy according to S&B p.328.
        :param mrp: Wrapped Markov Reward Process
        :param alpha: Learning Rate
        :param epochs: Number of episodes to train for
        :param discount: gamma
        :return:
        '''

        if steps_history is None:
            steps_history = []
        if learning_history is None:
            learning_history = []
        if coeffs_history is None:
            coeffs_history = []
        if loss_history is None:
            loss_history = []

        mu_optimizer = self.get_optimizer(name=mu_optimizer, learning_rate=alpha_mu, coeffs=self.mu_approximator.coeffs)
        sigma_optimizer = self.get_optimizer(name=sigma_optimizer, learning_rate=alpha_sigma, coeffs=self.sigma_approximator.coeffs)

        for episode in range(epochs):
            state = mrp.reset()[0]
            sigma = self.evaluate_sigma_at(state)
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_probs = []

            s = 0
            # Generate an episode
            while True:
                s += 1
                next_state, reward, terminated, truncated, _, action, prob = mrp.step(state, sigma)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_probs.append(prob)
                state = next_state
                sigma = self.evaluate_sigma_at(state)
                if terminated or truncated:
                    break

            total_t = len(episode_rewards)

            if autodiff:               
                for t, (state, action) in enumerate(zip(episode_states, episode_actions)):                        
                    g = self.compute_return(episode_rewards, discount, t)
                    def closure(): # second order optimizers like lbfgs require a closure to run loss computation multiple times
                        with torch.enable_grad():
                            mu_optimizer.zero_grad()
                            sigma_optimizer.zero_grad()
                            weight = total_t - t
                            loss = -torch.multiply(torch.multiply(self.log_prob_density(state, action), g), weight) # weight more recent actions (=end of episode) higher
                            loss.backward()
                            return loss
                    loss = mu_optimizer.step(closure=closure)
                    sigma_optimizer.step(closure=closure)
                    loss_history.append(loss.item())               
            else:
                # Update copies of the weights so that we can evaluate to the same values as during the actual episode
                weights_mu = copy.deepcopy(self.mu_approximator.coeffs)
                weights_sigma = copy.deepcopy(self.sigma_approximator.coeffs)

                for t, (state, action) in enumerate(zip(episode_states, episode_actions)):
                    g = self.compute_return(episode_rewards, discount, t)
                    self.reinforce_update_weights(alpha_mu, alpha_sigma, discount, g, state, action, weights_mu, weights_sigma, t, total_t)

                # Update model weights with the updated copies
                self.mu_approximator.coeffs = weights_mu
                self.sigma_approximator.coeffs = weights_sigma

            cumulated_reward = np.sum(episode_rewards)
            learning_history.append(cumulated_reward)
            steps_history.append(s)
            coeffs_history.append(copy.deepcopy((self.mu_approximator.coeffs.numpy())))

            if verbose:
                print(f"Episode {episode + 1}: Cumulated Reward: {cumulated_reward}\r", end="")

        return learning_history, loss_history, steps_history, coeffs_history
        