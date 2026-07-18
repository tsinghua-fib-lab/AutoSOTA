from logdiff.score.sampling_compositional import Expression
import torch

class ModelCompExpression(Expression):
    def log_probability(self, classifier, x, t):
        return 0.0
    
    @staticmethod
    def _append_parent(score_cache, child_expr, parent_expr):
        key = str(child_expr)
        if key not in score_cache:
            return
        entry = score_cache[key]
        parents = entry.setdefault("parents", [])
        parent_name = str(parent_expr)
        if parent_name not in parents:
            parents.append(parent_name)
        
#### AND ###
# AND: constant mix of models 0.5 s(a) + 0.5 * s(b)
class AndModelsAB(ModelCompExpression):
    method = "ModelComp"

    # 0.5 s(a) + 0.5 * s(b)
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding=False, method = "ModelComp"):
        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)

        def fn():      
            left = left_fn()
            right = right_fn()
            self._append_parent(score_cache, self.left, self)
            self._append_parent(score_cache, self.right, self)
            return guidance_dict["constant"]["and"] * (0.5 * left + 0.5 * right)
        return fn
    


class AndDombi(ModelCompExpression):
    method = "Dombi"
    gamma = 1.0

    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding=False, method = "ModelComp"):
        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)

        def fn(): 
            score1 = left_fn()
            score2 = right_fn()
            self._append_parent(score_cache, self.left, self)
            self._append_parent(score_cache, self.right, self)
            w1 = (1 + self.gamma) / (1 + self.gamma * score1)**2
            w2 = (1 + self.gamma) / (1 + self.gamma * score2)**2     
            return w1 * score1 + w2 * score2
        return fn
    
# AND: 
class AndGaripov(ModelCompExpression):
    method = "Garipov"
    # s(a) + CG(b)
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=False, method="Garipov"):
        right_fn = self.right.get_classifier_guidance(classifier, xt, t)
        left_fn = self.left.get_classifier_guidance(classifier, xt, t)

        def fn():      
            return 3 * (left_fn + right_fn)
        return fn
    
# AND: 
class AndSkreta(ModelCompExpression):
    method = "Skreta"

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

    def solve_superposition_system(self, div_l, div_r, eps_l, eps_r, xt, t, scheduler):
        # velocity in DDIM is negative noise prediction
        v_l = -eps_l
        v_r = -eps_r

        diff_neg_div = div_l - div_r
        diff_vel = v_l - v_r

        alpha_bar_t = scheduler.alphas_cumprod[t.long()].to(device=xt.device, dtype=xt.dtype)
        sigma_t = torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-12))

        # Enforce equal instantaneous log-density updates:
        # -div(v_l) + <v* - v_l, s_l> = -div(v_r) + <v* - v_r, s_r>
        # with s_i = v_i / sigma_t and v* = v_r + kappa * (v_l - v_r).
        denom = (diff_vel ** 2).sum((1, 2, 3)).clamp_min(1e-12)
        kappa = 1 + ((diff_vel * v_r).sum((1, 2, 3)) - sigma_t * diff_neg_div) / denom
        kappa = torch.nan_to_num(kappa, nan=0.5, posinf=1.0, neginf=0.0).clamp(-10, 10)
        combined_vel = v_r + kappa[:, None, None, None] * diff_vel
        
        # Return as epsilon (negative velocity)
        return -combined_vel

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, 
              ll_state, scheduler, score_cache, neg_guiding=False, method="Skreta"):
        
        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, 
                                  guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, 
                                   guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)

        def fn():
            eps_l = left_fn()
            eps_r = right_fn()
            self._append_parent(score_cache, self.left, self)
            self._append_parent(score_cache, self.right, self)

            # Density Update for AND (Composition)
            # The joint density is the sum of the densities (Eq. 14/15)
            div_l = score_cache[str(self.left)]["div"]
            div_r = score_cache[str(self.right)]["div"]
            
            # Solve the Skreta linear system
            eps_and = self.solve_superposition_system(div_l, div_r, eps_l, eps_r, xt, t, scheduler)
            
            score_cache[str(self)] = {
                "v": -eps_and,
            }

            return eps_and
            
        return fn
    

### OR ###

class OrModels(ModelCompExpression):
    method = "ModelComp"

    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ∨ {self.right})"

    def log_probability(self, classifier, x, t):
        # log(P_left + P_right) = log(exp(log P_left) + exp(log P_right))
        log_p_left = self.left.log_probability(classifier, x, t)
        log_p_right = self.right.log_probability(classifier, x, t)
        
        log_p_or = torch.logaddexp(log_p_left, log_p_right)
        
        return torch.clamp_max(log_p_or, 0.0)

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding=False, method="ModelComp"):
        log_p_L = self.left.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t)
        log_p_R = self.right.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t)
        log_p_OR = self.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t)

        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding=neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding=neg_guiding, method=self.method)
        
        # Calculate P_L / P_OR and P_R / P_OR in log-space
        log_weight_L = log_p_L - log_p_OR
        log_weight_R = log_p_R - log_p_OR
        
        # Convert to probability weights
        weight_L = torch.exp(log_weight_L)
        weight_R = torch.exp(log_weight_R)

        def fn():
            w_L = self.match_dims(weight_L, xt)
            w_R = self.match_dims(weight_R, xt)
            self._append_parent(score_cache, self.left, self)
            self._append_parent(score_cache, self.right, self)

            w_L = torch.clamp_max(w_L, torch.tensor(self.max_guidance_scale).to(device=xt.device))
            w_R = torch.clamp_max(w_R, torch.tensor(self.max_guidance_scale).to(device=xt.device))

            or_me_score = 1/(w_L + w_R) * (w_L * left_fn() + w_R * right_fn())
            return guidance_dict["logdiff"]["or_me"] * or_me_score
        return fn 
    

class OrDombi(ModelCompExpression):
    method = "Dombi"
    gamma = 1.0

    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding=False, method = "ModelComp"):
        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)

        def fn(): 
            score_1 = left_fn()
            score_2 = right_fn()
            self._append_parent(score_cache, self.left, self)
            self._append_parent(score_cache, self.right, self)
            w1 = (1 + self.gamma) / (1 + self.gamma * (1 - score_1))**2
            w2 = (1 + self.gamma) / (1 + self.gamma * (1 - score_2))**2  
            return w1 * score_1 + w2 * score_2
        return fn
    
# https://arxiv.org/abs/2412.17762
class OrSkreta(ModelCompExpression):
    method = "Skreta"

    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ∨ {self.right})"


    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, 
              ll_state, scheduler, score_cache, neg_guiding=False, method="Skreta"):
        
        # Pass the score_cache down to children
        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, 
                                  guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, 
                                   guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)

        def fn():
            eps_l = left_fn()
            eps_r = right_fn()
            self._append_parent(score_cache, self.left, self)
            self._append_parent(score_cache, self.right, self)
            
            # 1. Retrieve the densities calculated in the PREVIOUS step
            l_l = ll_state[str(self.left)]
            l_r = ll_state[str(self.right)]
            
            # 2. Compute mixing weights via Softmax (Eq. 12)
            weights = torch.softmax(torch.stack([l_l, l_r]), dim=0)
            w_l, w_r = weights[0].view(-1,1,1,1), weights[1].view(-1,1,1,1)
            
            # 3. Combine the noise
            eps_or = w_l * eps_l + w_r * eps_r

            # Keep parent velocity in cache so children can reference it.
            score_cache[str(self)] = {
                "v": -eps_or,
            }
            
            return eps_or
            
        return fn
    
class NotModels(ModelCompExpression):  
    method = "ModelComp"
    # s(∅)− alpha * s(a)
    def __init__(self, expression: Expression):
        super().__init__()
        self.expression: Expression = expression
        self.alpha = 0.07

    def __str__(self):
        return f"¬{self.expression}"

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding=False, method="ModelComp"):
        atom_fn = self.expression.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state, scheduler, score_cache, neg_guiding, method=self.method)

        def fn():
            atom = atom_fn()
            self._append_parent(score_cache, self.expression, self)
            return x_0_unconditional + guidance_dict["constant"]["not"] * -self.alpha * atom
        return fn
