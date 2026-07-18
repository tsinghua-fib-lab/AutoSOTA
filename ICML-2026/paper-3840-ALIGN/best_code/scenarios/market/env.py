class ProductChoiceMarketEnv:
    """
    Multi-seller / multi-buyer product-choice environment.
    """

    def __init__(self, cfg):
        self.config = cfg
        env_cfg = cfg.experiment.env

        # Prices
        self.P_c = float(getattr(env_cfg, "P_c", 3.0))
        self.P_s = float(getattr(env_cfg, "P_s", 1.0))

        # Costs
        self.C_H = float(getattr(env_cfg, "C_H", 1.0))
        self.C_L = float(getattr(env_cfg, "C_L", 0.0))

        # Customer values V_{q,t}
        self.V_Hc = float(getattr(env_cfg, "V_Hc", 6.0))
        self.V_Hs = float(getattr(env_cfg, "V_Hs", 3.0))
        self.V_Lc = float(getattr(env_cfg, "V_Lc", 3.0))
        self.V_Ls = float(getattr(env_cfg, "V_Ls", 2.0))

        self.payoff_matrix = {
            ('H', 'c'): (self.P_c - self.C_H, self.V_Hc - self.P_c),
            ('H', 's'): (self.P_s - self.C_H, self.V_Hs - self.P_s),
            ('L', 'c'): (self.P_c - self.C_L, self.V_Lc - self.P_c),
            ('L', 's'): (self.P_s - self.C_L, self.V_Ls - self.P_s),
            ('none', 'none'): (0.0, 0.0)
        }

    def reset(self, sellers, buyers):
        for agent in list(sellers) + list(buyers):
            agent.actions = []
            agent.rewards = [] # stores the reward obtained at every time step
            agent.stm = [] # The interaction log for each episode    

    def step(self, seller_action, buyer_action):
        # Refuse to buy: no sale, no production, no cost.
        if buyer_action == "none":
            return self.payoff_matrix[('none', 'none')]

        if seller_action not in {"H", "L"}:
            raise ValueError(f"seller_action must be 'H' or 'L', got {seller_action}")
        if buyer_action not in {"c", "s"}:
            raise ValueError(f"buyer_action must be 'c', 's', or 'none', got {buyer_action}")
        
        seller_reward, buyer_reward = self.payoff_matrix[(seller_action, buyer_action)]
        
        return seller_reward, buyer_reward