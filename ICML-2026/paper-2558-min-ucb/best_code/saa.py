import numpy as np
import gurobipy as gp
from gurobipy import GRB

class SAA:
    def __init__(self, model, c, n_items, n_machines):
        self.model = model
        self.c = c
        self.n_items = n_items
        self.n_machines = n_machines

    def solve_nf(self, params_list, max_iter=30, tol=1e-4):
        """
        Two-stage stochastic program solved by classical L-shaped algorithm.
        """
        try:
            N = len(params_list['h'])   
            p = np.ones(N)              

            model = gp.Model("TwoStage_LShaped")
            x = model.addVars(self.n_items, lb=0, ub=5000, name="x")
            theta = model.addVar(lb=-3000, name="theta")

            model.setObjective(gp.quicksum(self.c[i] * x[i] for i in range(self.n_items)) + theta,
                            GRB.MINIMIZE)
            model.setParam("OutputFlag", 0)

            it, LB, UB = 0, -np.inf, np.inf

            while it < max_iter and UB - LB > tol:
                it += 1
                model.optimize()
                if model.status != GRB.OPTIMAL:
                    raise RuntimeError("Master problem not optimal")

                x_val = np.array([x[i].X for i in range(self.n_items)])
                LB = model.ObjVal

                Es, es = np.zeros(self.n_items), 0.0
                q_vals = []

                for n in range(N):
                    sub = gp.Model(f"Sub_{n}")
                    sub.setParam("OutputFlag", 0)
                    y = sub.addVars(3*self.n_machines, lb=0, name="y")

                    rhs_vec = - params_list['T'] @ x_val
                    for i in range(self.n_machines):
                        sub.addConstr(
                            gp.quicksum(params_list['W'][n][i, j] * y[j] for j in range(3*self.n_machines)) == rhs_vec[i]
                        )

                    sub.addConstr(
                        gp.quicksum(y[j] for j in range(self.n_machines, 2*self.n_machines)) == params_list['h'][n][-1]
                    )

                    sub.setObjective(
                        gp.quicksum(params_list['q'][n][j] * y[j] for j in range(3*self.n_machines)),
                        GRB.MINIMIZE
                    )
                    sub.optimize()

                    if sub.status != GRB.OPTIMAL:
                        raise RuntimeError(f"Subproblem infeasible at scenario {n}")

                    q_vals.append(sub.ObjVal)

                    pi = np.array([c.Pi for c in sub.getConstrs()])
                    Es += p[n] * (pi @ params_list['T'])
                    es += p[n] * (pi @ params_list['h'][n])

                recourse_est = np.mean(q_vals)
                UB = min(UB, float(np.dot(self.c, x_val) + recourse_est))

                model.addConstr(theta >= es - gp.quicksum(Es[i] * x[i] for i in range(self.n_items)))

            return x_val, UB, it

        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
        except Exception as e:
            print(f"Other error: {e}")



    def saa_oos(self, x: np.ndarray, test_params):
        N, J, I = len(test_params['h']), self.n_machines, self.n_items
        q, W, h, T = test_params['q'], test_params["W"], test_params["h"], test_params["T"]

        x_use = np.asarray(x[:I], dtype=float)
        c_term = float(np.dot(self.c[:I], x_use))

        # For each scenario, solve the second-stage recourse LP with fixed x
        q_vals = []
        for n in range(N):
            m = gp.Model(f"Q_n_{n}")
            m.setParam('OutputFlag', 0)
            y = m.addVars(3*J, lb=0.0, name="y")

            m.addConstr(gp.quicksum(y[j] for j in range(J,2*J)) == float(h[n][-1]))
            for i in range(self.n_machines):
                m.addConstr(
                    gp.quicksum(W[n][i,j] * y[j] for j in range(3*J)) == -gp.quicksum(
                        T[i,j] * x_use[j] for j in range(I)), name=f"Sub_Constr_{i}")

            m.setObjective(gp.quicksum(float(q[n, j]) * y[j] for j in range(J)), GRB.MINIMIZE)
            m.optimize()
            if m.status != GRB.OPTIMAL:
                raise RuntimeError(f"OOS second-stage failed at scenario {n} with status {m.status}")
            q_vals.append(float(m.objVal))

        recourse_mean = float(np.mean(q_vals))
        return c_term + recourse_mean
    