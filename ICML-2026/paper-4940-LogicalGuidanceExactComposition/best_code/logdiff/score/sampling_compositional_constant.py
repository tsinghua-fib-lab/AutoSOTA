from logdiff.score.sampling_compositional import Expression, Or_ME
import torch

class ConstantExpression(Expression):
    method = "Constant"

    def log_probability(self, classifier, x, t):
        return torch.Tensor([0.0]).to(x.device) 
    
class NotConstant(ConstantExpression):  
    # s(∅)−s(a)
    def __init__(self, expression: Expression):
        super().__init__()
        self.expression: Expression = expression
        self.alpha = 1.0

    def __str__(self):
        return f"¬{self.expression}"

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=False, method="Constant"):
        atom_fn = self.expression.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding, method=self.method)

        def fn():
            return guidance_dict["constant"]["not"] * -self.alpha * atom_fn()
        return fn

    
class AndConstant(ConstantExpression):
    # s(a)+s(b)−s(∅)
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=False, method="Constant"):
        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding)

        def fn():      
            return guidance_dict["constant"]["and"] * (left_fn() + right_fn())
        return fn
    
# class OrConstant(Or_ME, ConstantExpression):
#     # s(a)+s(b)−s(a)∗s(b)
#     def __init__(self, left: Expression, right: Expression):
#         super().__init__(left, right)

#     def log_probability(self, classifier, x, t):
#         return Or_ME.log_probability(self, classifier, x, t)

class OrConstantWeights(ConstantExpression):
    # 0.5s(a)+0.5s(b)
    def __init__(self, left: Expression, right: Expression):
        super().__init__()
        self.left: Expression = left
        self.right: Expression = right
        self.alpha = 0.5  # weighting factor for balancing contributions

    def __str__(self):
        return f"({self.left} ∨ {self.right})"

    def to_fn(self, model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding=False, method="Constant"):   
        left_fn = self.left.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding, method=self.method)
        right_fn = self.right.to_fn(model, classifier, x_0_unconditional, xt, t, guidance_dict, neg_guiding, method=self.method)

        def fn():
            return guidance_dict["constant"]["or"] * (self.alpha * left_fn() + self.alpha * right_fn())
        return fn 
    

