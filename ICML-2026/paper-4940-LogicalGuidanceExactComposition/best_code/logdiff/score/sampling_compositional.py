import torch
from abc import ABC, abstractmethod

class Expression(ABC):
    def __init__(self):
        super().__init__()
        self.max_guidance_scale = 3
        self.num_divergence_probes = 2

    @abstractmethod
    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=False):
        raise NotImplementedError
    
    @abstractmethod
    def log_probability(self, classifier, x, t):
        """Return log-probability of the expression being true."""
        raise NotImplementedError
    
    @staticmethod
    def match_dims(x, ref):
        return x.view(*x.shape, *([1] * (ref.ndim - x.ndim)))
    
    def get_classifier_x(self, xt, x_0_noise):
        return xt 
    
class LGCDiffExpression(Expression):
    method = "LGCDiff"
    

class Atom(Expression):
    def __init__(self, condition):
        super().__init__()
        self.condition = condition

    @abstractmethod
    def log_probability(self, classifier, x, t):
        raise NotImplementedError("Implement probability for atom type.")
    
    def get_classifier_guidance(self, classifier, xt, t, neg_guiding=False):
        raise NotImplementedError("Implement negative guiding for atom type.")
    
    @abstractmethod
    def get_neg_guiding_cond_prob(self, classifier, x, t):
        raise NotImplementedError("Implement negative guiding for atom type.")

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, ll_state=None, scheduler=None, score_cache=None, neg_guiding=False, method="LGCDiff"):
    
        def cfg_fn():
            if neg_guiding:
                most_propable_cond, log_p = self.get_neg_guiding_cond_prob(classifier, self.get_classifier_x(xt, x_0_unconditional), t)
                neg_cond = torch.tensor(most_propable_cond, dtype=torch.long).to(device=xt.device)
                x_cond_neg = model(xt, t, neg_cond)
                not_weight = Not.get_not_weights(log_p, self.max_guidance_scale, xt)
            x_cond = model(xt, t, self.condition.unsqueeze(0).repeat(xt.shape[0], 1).to(device=xt.device))
            if neg_guiding:
                atom = guidance_dict["atom"] * (x_cond - x_0_unconditional) + guidance_dict["not"] * not_weight * (x_cond_neg - x_0_unconditional)
            else:
                atom = guidance_dict["atom"] * (x_cond - x_0_unconditional)
            return atom
        
        def ito_fn():
            cond = self.condition.view(1, -1).repeat(xt.shape[0], 1).to(device=xt.device)

            def vel_fn(x, t, labels):
                noise_pred = model(x, t, labels)
                return -noise_pred 

            shared_probes = score_cache.get("_shared_probes")
            if shared_probes is None:
                shared_probes = [
                    torch.randint_like(xt, 2, dtype=xt.dtype, device=xt.device) * 2 - 1
                    for _ in range(self.num_divergence_probes)
                ]
                score_cache["_shared_probes"] = shared_probes

            vel = None
            vel_div_acc = None
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                for probe_idx, eps in enumerate(shared_probes):
                    vel_curr, vel_div_curr = torch.func.jvp(
                        lambda x: vel_fn(x, t, cond),  # Only x is the input to the differentiated function
                        (xt,),                         # Primal (must be a tuple)
                        (eps,)                         # Tangent (must be a tuple)
                    )
                    if vel is None:
                        vel = vel_curr
                        vel_div_acc = -(eps * vel_div_curr).sum((1, 2, 3)).to(xt.dtype)
                    else:
                        vel_div_acc = vel_div_acc + (-(eps * vel_div_curr).sum((1, 2, 3)).to(xt.dtype))

            vel = vel.to(xt.dtype)
            vel_div = vel_div_acc / len(shared_probes)
            
            cache_entry = {
                "div": vel_div,
                "v": vel,
                "parents": [],
            }
            
            score_cache[str(self)] = cache_entry

            # vel = -eps, but code works with eps
            return -vel
        
        if method in ["LGCDiff", "Constant"]:
            return cfg_fn
        elif method in ["ModelComp", "Skreta", "Dombi"]:
            return ito_fn
        else:
            raise NotImplementedError(f"Method {method} not implemented for Atom.")
    
    
class Not(LGCDiffExpression):  
    def __init__(self, expression: Expression):
        super().__init__()
        self.expression: Expression = expression

    def __str__(self):
        return f"¬{self.expression}"

    def log_probability(self, classifier, x, t):
        # p_not = 1 - p
        log_p = self.expression.log_probability(classifier, x, t)
        
        clamped_log_p = torch.clamp_max(log_p, 0.0) 

        # log(1 - P) = log(1 - exp(log P))
        log_p_not = torch.log(-torch.expm1(clamped_log_p))
        return log_p_not
    
    @classmethod
    def get_not_weights(cls, log_p, max_guidance_scale, xt):
        # log(1 - P)
        clamped_log_p = torch.clamp_max(log_p, 0.0)
        log_p_not = torch.log(-torch.expm1(clamped_log_p))
        
        # log(P / (1 - P)) = log_p - log_p_not
        log_weight_magnitude = clamped_log_p - log_p_not
        
        weight_magnitude = torch.exp(log_weight_magnitude)
        weight = -torch.clamp_max(weight_magnitude, torch.tensor(max_guidance_scale).to(device=xt.device)) # Re-introduce negative sign

        weight = cls.match_dims(weight, xt)
        return weight

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=False, method="LGCDiff"):
        log_p = self.expression.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t)
        expression_fn = self.expression.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=neg_guiding, method=self.method)

        def fn():
            not_weight = self.get_not_weights(log_p, self.max_guidance_scale, xt)
            return guidance_dict["logdiff"]["not"] * not_weight * expression_fn()
        return fn


class And(LGCDiffExpression):
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

    def log_probability(self, classifier, x, t):
        log_p = self.left.log_probability(classifier, x, t) + self.right.log_probability(classifier, x, t)
        return torch.clamp_max(log_p, 0.0)

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=False, method="LGCDiff"):
        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=neg_guiding, method=self.method)

        def fn():
            return guidance_dict["logdiff"]["and"] * (left_fn() + right_fn())
        return fn
    
class Or_CI(LGCDiffExpression):
    # mutually independent events
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ∨ {self.right})"

    def log_probability(self, classifier, x, t):
        log_p_left = self.left.log_probability(classifier, x, t)
        log_p_right = self.right.log_probability(classifier, x, t)
        
        clamped_log_p_left = torch.clamp_max(log_p_left, 0.0)
        clamped_log_p_right = torch.clamp_max(log_p_right, 0.0)
        
        # log(1 - P)
        log_p_not_left = torch.log(-torch.expm1(clamped_log_p_left))
        log_p_not_right = torch.log(-torch.expm1(clamped_log_p_right))
        
        log_p_not_and = log_p_not_left + log_p_not_right
        
        # log(P_or) = log(1 - P_not_and)
        log_p_or = torch.log(-torch.expm1(log_p_not_and))
        
        return torch.clamp_max(log_p_or, 0.0)


    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=False, method="LGCDiff"):
        log_p_L = self.left.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t)
        log_p_R = self.right.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t)
        
        clamped_log_p_L = torch.clamp_max(log_p_L, 0.0)
        clamped_log_p_R = torch.clamp_max(log_p_R, 0.0)
        log_p_OR = self.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t) 
        
        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=neg_guiding, method=self.method)
        
        # Calculate P(L AND NOT R) and P(R AND NOT L) in log-space
        
        # log P(NOT R)
        log_p_not_R = torch.log(-torch.expm1(clamped_log_p_R))
        log_p_L_and_not_R = clamped_log_p_L + log_p_not_R
        
        # log P(NOT L)
        log_p_not_L = torch.log(-torch.expm1(clamped_log_p_L))
        log_p_R_and_not_L = clamped_log_p_R + log_p_not_L

        def fn():
            weight_L = torch.exp(log_p_L_and_not_R - log_p_OR)
            weight_R = torch.exp(log_p_R_and_not_L - log_p_OR)
            
            w_L = self.match_dims(weight_L, xt)
            w_R = self.match_dims(weight_R, xt)

            w_L = torch.clamp_max(w_L, torch.tensor(self.max_guidance_scale).to(device=xt.device))
            w_R = torch.clamp_max(w_R, torch.tensor(self.max_guidance_scale).to(device=xt.device))

            return guidance_dict["logdiff"]["or_ci"] * (w_L * left_fn() + w_R * right_fn())
        return fn 
    
class Or_ME(LGCDiffExpression):
    # mutually exclusive events (P_or = P_left + P_right)
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ⊻ {self.right})"

    def log_probability(self, classifier, x, t):
        # log(P_left + P_right) = log(exp(log P_left) + exp(log P_right))
        log_p_left = self.left.log_probability(classifier, x, t)
        log_p_right = self.right.log_probability(classifier, x, t)
        
        log_p_or = torch.logaddexp(log_p_left, log_p_right)
        
        return torch.clamp_max(log_p_or, 0.0)

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=False, method="LGCDiff"):
        log_p_L = self.left.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t)
        log_p_R = self.right.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t)
        log_p_OR = self.log_probability(classifier, self.get_classifier_x(xt, x_0_unconditional), t)

        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=neg_guiding, method=self.method)
        
        # Calculate P_L / P_OR and P_R / P_OR in log-space
        log_weight_L = log_p_L - log_p_OR
        log_weight_R = log_p_R - log_p_OR
        
        # Convert to probability weights
        weight_L = torch.exp(log_weight_L)
        weight_R = torch.exp(log_weight_R)

        def fn():
            w_L = self.match_dims(weight_L, xt)
            w_R = self.match_dims(weight_R, xt)

            w_L = torch.clamp_max(w_L, torch.tensor(self.max_guidance_scale).to(device=xt.device))
            w_R = torch.clamp_max(w_R, torch.tensor(self.max_guidance_scale).to(device=xt.device))

            or_me_score = w_L * left_fn() + w_R * right_fn()
            return guidance_dict["logdiff"]["or_me"] * or_me_score
        return fn      


class LogicModelWrapper(torch.nn.Module):
    def __init__(self, model, classifier, neg_guiding=False):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.neg_guiding = neg_guiding

    @property
    def config(self):
        return self.model.config

    @torch.no_grad()
    def forward(self, xt, t, query, guidance_dict, null_token,  ll_state, scheduler, **kwargs):
        xt = xt.to(device=null_token.device)
        t = t.to(device=null_token.device)
        x0_uncond_noise = self.model(xt, t, null_token)
        if query is None:  # unconditional
            return x0_uncond_noise, None
        if query.method in ["LGCDiff", "Constant", "Garipov"]: 
            guidance_fn = query.to_fn(self.model, self.classifier, x0_uncond_noise, xt, t, guidance_dict, neg_guiding=self.neg_guiding, method=query.method)
            guidance = guidance_fn()
            return x0_uncond_noise + guidance, None
        elif query.method in ["Skreta", "ModelComp", "Dombi"]: 
            score_cache = {}
            xt_composed_fn = query.to_fn(self.model, self.classifier, x0_uncond_noise, xt, t, guidance_dict, ll_state=ll_state, 
                scheduler=scheduler, score_cache=score_cache, neg_guiding=self.neg_guiding, method=query.method)
            eps_total = xt_composed_fn()
            return eps_total, score_cache
        else:
            raise NotImplementedError(f"Method {query.method} not implemented in LogicModelWrapper.")
        
