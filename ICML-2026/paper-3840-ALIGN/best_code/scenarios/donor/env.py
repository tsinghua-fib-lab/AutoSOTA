class DonorGameEnv:
    def __init__(self, cfg):
        self.config = cfg

    def reset(self, agents):
        for agent in agents:
            agent.resources = self.config.experiment.env.initial_resources
            agent.donations = []
            agent.donation_ratios = []
            agent.benefits = [] # stores the benefit received by the agent everytime he takes on recipient role
            agent.rewards = [] # stores the reward obtained at every time step (so includes both donor and recipient roles)
            agent.stm = [] # The interaction log for each episode

    def step(self, donor, recipient, donation, benefit):
        donor.resources -= donation
        recipient.resources += benefit