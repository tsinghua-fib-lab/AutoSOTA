class Welford:
    def __init__(self, var_floor: float = 0.01, initial_var: float = float('inf'), ema_alpha: float = 0.0):
        if var_floor < 0:
            raise ValueError("var_floor must be non-negative.")
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.var_floor = float(var_floor)
        self.initial_var = float(initial_var)
        self.ema_alpha = float(ema_alpha)
        if not (0.0 <= self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in [0, 1].")
        self.exp_var = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.ema_alpha > 0.0:
            sq_dev = delta2 * delta2
            self.exp_var = sq_dev if self.n == 1 else (1.0 - self.ema_alpha) * self.exp_var + self.ema_alpha * sq_dev

    @property
    def var(self) -> float:
        if self.n < 2:
            return self.initial_var
        adaptive_floor = max(0.001, self.var_floor / max(1.0, self.n ** 0.5))
        std_var = self.M2 / (self.n - 1)
        if self.ema_alpha > 0.0 and self.n >= 3:
            # Blend standard Welford variance with EMA (recency-weighted) variance
            # Higher ema_alpha gives more weight to recent observations
            blend_weight = 0.5  # equal blend
            blended = (1.0 - blend_weight) * std_var + blend_weight * self.exp_var
            return max(blended, adaptive_floor)
        return max(std_var, adaptive_floor)
