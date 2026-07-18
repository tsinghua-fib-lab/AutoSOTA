class TrustGameEnv:
    def __init__(self, cfg):
        self.config = cfg
        self.investment_multiplier = cfg.experiment.env.investment_multiplier

    # Additional methods to implement the environment dynamics would go here
    def reset(self, agents):
        for agent in agents:
            agent.resources = self.config.experiment.env.initial_resources
            agent.investments = [] # stores the investment made by the agent every time they take on the investor role
            agent.investment_ratios = [] # stores the investment made by the agent every time they take on the investor role
            agent.benefits = []  # stores the benefit received by the agent every time they take on the responder role
            agent.returned_amounts = [] # stores the amount returned by the agent every time they take on the responder role
            agent.returned_ratios = [] # stores the return ratio (returned amount / (investment * investment_ratio)) by the agent every time they take on the responder role
            agent.rewards = []  # stores the reward obtained at every time step (so includes both investor and responder roles)
            agent.stm = []  # The interaction log for each episode
    
    def step(self, investor, responder, investment, investment_ratio, returned_amount, returned_ratio):
        investor.resources = investor.resources - investment + returned_amount
        responder.resources = responder.resources + investment * self.config.experiment.env.investment_multiplier - returned_amount
        # Update reward signals and investments made
        investor.investments.append(investment)
        investor.investment_ratios.append(investment_ratio)
        responder.returned_amounts.append(returned_amount)
        responder.returned_ratios.append(returned_ratio)
        investor.rewards.append(-investment + returned_amount)
        responder.rewards.append(investment * self.investment_multiplier - returned_amount)