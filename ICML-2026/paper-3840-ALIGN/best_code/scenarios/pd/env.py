class PDEnv:
    def __init__(self, cfg):
        self.config = cfg
        self.payoff_matrix = {
            ('C', 'C'): (self.config.experiment.env.benefit - self.config
.experiment.env.cost, self.config.experiment.env.benefit - self.config.experiment.env.cost),
            ('C', 'D'): (-self.config.experiment.env.cost, self.config.experiment.env.benefit),
            ('D', 'C'): (self.config.experiment.env.benefit, -self.config.experiment.env.cost),
            ('D', 'D'): (0, 0)}

    def reset(self, agents):
        for agent in agents:
            agent.actions = []
            agent.rewards = [] # stores the reward obtained at every time step
            agent.stm = [] # The interaction log for each episode

    def step(self, actions):
        action1, action2 = actions
        reward1, reward2 = self.payoff_matrix[(action1, action2)]
        return (reward1, reward2)