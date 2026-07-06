import numpy as np

class FundamentalsLayer:
    def __init__(self, payout_ratio=0.3, volatility=None):
        self.payout_ratio = np.array(payout_ratio) if not isinstance(payout_ratio, (float,int)) else float(payout_ratio)
        self.volatility = volatility
        self.interest_rate = 0.03
        self.gdp_growth = 0.02
        self.inflation = 0.02


        self._denom_tol = 1e-3
        self._eps_min = 1e-6
        self._intrinsic_cap = 1e10

    def update_macro(self, t):
        self.interest_rate = 0.02 + 0.01 * np.sin(2*np.pi*t/50) + np.random.normal(0, 0.002)
        self.gdp_growth   = 0.02 + 0.01 * np.sin(2*np.pi*t/100) + np.random.normal(0, 0.002)
        self.inflation    = 0.02 + np.random.normal(0, 0.001)

    def compute_eps(self, current_prices):
        """Estimate EPS from current prices."""
        current_prices = np.asarray(current_prices, dtype=float)
        eps = current_prices * 0.05

        if self.volatility is not None:

            noise = np.random.normal(0, self.volatility, size=eps.shape)
            noise = np.clip(noise, -0.5, 0.5)
            eps = eps * (1 + noise)

        drift = self.gdp_growth - self.inflation
        eps = eps * (1 + drift)


        eps = np.maximum(eps, self._eps_min)
        return eps

    def compute_dividends(self, eps):
        if isinstance(self.payout_ratio, (float, int)):
            return eps * float(self.payout_ratio)
        else:
            payout = np.array(self.payout_ratio, dtype=float)
            return eps * payout

    def _estimate_growth(self, history_i):
        """Docstring."""
        h = np.asarray(history_i, dtype=float)
        if len(h) < 2 or h[0] <= 0:
            return 0.0

        lr = np.diff(np.log(h))

        g_continuous = np.mean(lr)

        g = np.exp(g_continuous) - 1
        return float(g)

    def intrinsic_value(self, eps, history=None, method="advanced"):
        values = []
        for i in range(len(eps)):
            r = max(0.01, float(self.interest_rate + self.inflation))

            if method == "simple":
                g = float(self.gdp_growth * 0.5)
            elif method == "advanced":
                if history is None:
                    raise ValueError("history required for advanced method")
                g = self._estimate_growth(history[i])
            else:
                raise ValueError("invalid method")

            denom = r - g

            if abs(denom) < self._denom_tol:
                denom = np.sign(denom) * self._denom_tol if denom != 0 else self._denom_tol

            val = eps[i] / denom


            if not np.isfinite(val) or abs(val) > self._intrinsic_cap or val < 0:

                pe = 15.0
                val = max(eps[i] * pe, 0.0)

            values.append(float(val))

        return np.array(values)

    def step(self, history, t, method="advanced"):
        """Docstring."""
        self.update_macro(t)

        current_prices = np.array([h[-1] for h in history], dtype=float)
        eps = self.compute_eps(current_prices)
        dividends = self.compute_dividends(eps)
        intrinsic_values = self.intrinsic_value(eps, history, method=method)


        for i, v in enumerate(intrinsic_values):
            if v > 1e8:
                print(f"WARNING: intrinsic_value[{i}]={v} (eps={eps[i]}, r={self.interest_rate+self.inflation}, history0={history[i][0]}, history_last={history[i][-1]})")

        return intrinsic_values

def update_price(P_curr, P_deal, Q, F, Q_total, V=None,
                 mu=0.0, sigma=0.01, dt=1.0,
                 k=0.5, lam=0.05, L=100.0, gamma=0.5, eta=0.2):
        """Docstring."""

        # -----------------


        # -----------------
        q_ratio = Q / (Q_total + 1e-6)
        impact = eta * np.sign(P_deal - P_curr) * (abs(Q)**gamma) / (L + abs(Q))

        # -----------------

        # -----------------
        if V is not None:
            revert = lam * (V - P_curr)
        else:
            revert = 0.0

        # -----------------

        # -----------------
        mu_prime = mu + k * q_ratio + revert + impact

        # -----------------

        # -----------------
        Z = np.random.normal()
        P_next = P_curr * np.exp((mu_prime - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        if P_next > 0:
            return P_next
        else:
            return self.update_price_gbm(P_curr, P_deal, Q, F, Q_total)

if __name__ == "__main__":

    price_history = [
        np.array([465.75, 469.40, 472.85, 471.10, 474.25, 478.50, 482.75, 480.20, 483.10, 486.55]),
        np.array([508.45, 512.30, 517.75, 520.60, 516.20, 523.15, 529.40, 532.75, 528.90, 535.20]),
        np.array([388.40, 384.15, 380.25, 377.90, 373.50, 369.10, 365.45, 361.80, 358.20, 355.75])
    ]

    fundamentals = FundamentalsLayer(payout_ratio=[0.2, 0.3, 0.5], volatility=0.1)
    for i in range(10):
        intrinsic_values = fundamentals.step(price_history, t=i, method="advanced")
        print(f"Intrinsic={intrinsic_values}")

    price_history = [
        np.array([474.25, 478.50, 482.75, 480.20, 483.10, 486.55]),
        np.array([516.20, 523.15, 529.40, 532.75, 528.90, 535.20]),
        np.array([373.50, 369.10, 365.45, 361.80, 358.20, 355.75])
    ]

    intrinsic_values = fundamentals.step(price_history, t=10, method="advanced")
    print(f"Intrinsic={intrinsic_values}")
    price = update_price(486.55,480,20,1000,intrinsic_values[0])
    print(price)
