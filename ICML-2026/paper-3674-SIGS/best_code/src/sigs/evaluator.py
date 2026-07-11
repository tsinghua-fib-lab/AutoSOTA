
import re
import concurrent.futures
import numpy as np
import sympy as sp
import symengine as sm
from timeout_decorator import timeout, TimeoutError


class DerivativeEvaluator:
    """
    Symbolic PDE residual evaluator used in Stage I candidate scoring (paper §3.1).

    Given a candidate ansatz u(x,y,t) and a PDE operator string, computes the
    residual  L[u] - F  and evaluates boundary/initial conditions on a mesh.
    Scoring combines PDE RMSE + IC/BC RMSE to rank candidates.
    """

    def __init__(self, operator, decoded_expression, F, initial_conditions=None, param_map=None, symbols=None):
        self.operator_expression = operator
        try:
            self.main_expression = sp.sympify(decoded_expression)
        except RuntimeError as e:
            print(f"Error sympifying expression: {decoded_expression}: {e}")

        if 'u' in str(F):
            F = F.replace('u', f"({decoded_expression})")
        self.F = sp.sympify(F)
        self.initial_conditions = initial_conditions or []
        self.param_map = param_map or {}
        self.symbols = symbols

        self.operator_map = {
            'd/dx':      lambda expr: sm.diff(expr, self.symbols['x']),
            '(d/dx)^2':  lambda expr: sm.diff(expr, self.symbols['x']) ** 2,
            'd2/dx2':    lambda expr: sm.diff(expr, self.symbols['x'], 2),
            'd3/dx3':    lambda expr: sm.diff(expr, self.symbols['x'], 3),
            'd/dy':      lambda expr: sm.diff(expr, self.symbols['y']),
            'd2/dy2':    lambda expr: sm.diff(expr, self.symbols['y'], 2),
            'd/dt':      lambda expr: sm.diff(expr, self.symbols['t']),
            'd2/dt2':    lambda expr: sm.diff(expr, self.symbols['t'], 2),
        }

        self.F = self.F.subs(self.param_map)
        self.simplified_expression = self.main_expression.subs(self.param_map)

    def distribute_and_evaluate_derivative(self, symbols, meshes):
        """Compute PDE residual string and IC/BC errors on the provided mesh."""
        terms = re.findall(
            r'(?:\(\s*d/dx\s*\)\s*\^\s*\d+)|(?:\d+|\([^()]+\)|[+-]|(?:d/dx|d2/dx2|d3/dx3|d/dt|d/dy|d2/dy2|d2/dt2|))(?:\s*\*\s*(?:\d+|\([^()]+\)|(?:d/dx|d2/dx2|d3/dx3|d/dt|d/dy|d2/dy2|d2/dt2)))*',
            self.operator_expression
        )
        terms = [t.strip() for t in terms if t.strip()]
        self.result_expr = sm.Integer(0)
        current_sign = 1

        for term in terms:
            if term in ('+', '-'):
                current_sign = 1 if term == '+' else -1
                continue
            try:
                parts = re.split(r'\s*\*\s*', term)
                coeff = sm.Integer(1)
                operators = []
                for part in parts:
                    part = part.strip()
                    if part in self.operator_map:
                        operators.append(part)
                    else:
                        part = part.replace('u', f"({self.main_expression})")
                        coeff *= sp.sympify(part)
                current_expr = self.simplified_expression
                for op in operators:
                    current_expr = self.operator_map[op](current_expr)
                self.result_expr += current_sign * coeff * current_expr
            except Exception as e:
                print(f"Error processing term {term}: {e}")
                raise

        self.result_expr -= self.F
        evaluated_conditions = self.apply_initial_conditions(self.main_expression, symbols, meshes)
        return str(self.result_expr), evaluated_conditions, self.F

    def parse_initial_conditions(self, condition):
        """Parse 'u(x=0,t=free) = 0' style condition string into components."""
        func_and_vars, result = map(str.strip, condition.rsplit('=', 1))
        result = sm.sympify(result)
        func_part, vars_part = func_and_vars.strip('()').split(',', 1)
        if ' ' in func_part:
            operator, func = func_part.split(' ', 1)
        else:
            operator, func = None, func_part
        vars_values = {}
        for var_value in vars_part.split(','):
            var, val = map(str.strip, var_value.split('='))
            vars_values[var] = 'free' if val == var else sm.sympify(val)
        return func, vars_values, result, operator

    def is_number(self, value=0):
        """Return True if value is a numeric constant (Python or symbolic)."""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, sm.Basic) and not value.free_symbols:
            try:
                numeric_value = value.evalf()
                return numeric_value.is_real or value == sm.Integer(0)
            except (TypeError, ValueError):
                pass
        return False

    @timeout(5)
    def apply_initial_conditions(self, expr, symbols, meshes):
        """Evaluate each IC/BC and return a list of per-condition RMSE values."""
        evaluated_conditions = []
        var_names = list(symbols.keys())

        for condition in self.initial_conditions:
            try:
                func, vars_values, expected, operator = self.parse_initial_conditions(condition)
                expr_to_evaluate = self.operator_map[operator](expr) if (operator and operator in self.operator_map) else expr

                eval_meshes = [
                    np.full_like(meshes[i], float(vars_values[var]))
                    if var in vars_values and vars_values[var] != 'free'
                    else meshes[i]
                    for i, var in enumerate(var_names)
                ]

                predicted = ExpressionEvaluator.evaluate(expr_to_evaluate, eval_meshes, self.param_map, symbols)
                predicted = np.nan_to_num(predicted, nan=0.0, posinf=1e3, neginf=-1e3)

                if self.is_number(expected):
                    actual = np.full_like(predicted, float(expected))
                else:
                    free_vars = [v for v in var_names if v in vars_values and vars_values[v] == 'free']
                    free_symbols = {v: symbols[v] for v in free_vars}
                    free_meshes = [meshes[var_names.index(v)] for v in free_vars]
                    actual = np.nan_to_num(np.asarray(
                        ExpressionEvaluator.evaluate(expected, free_meshes, self.param_map, free_symbols)
                    ), nan=0.0, posinf=1e3, neginf=-1e3)

                if predicted.shape == actual.shape:
                    evaluated_conditions.append(float(np.sqrt(np.mean((predicted - actual) ** 2))))
                else:
                    evaluated_conditions.append(1e3)
            except Exception as e:
                print(f"Error applying condition '{condition}': {e}")
                evaluated_conditions.append(1e3)

        return evaluated_conditions


class ExpressionEvaluator:
    """Lambdify-based numerical evaluator for sympy expressions on NumPy meshes."""

    @staticmethod
    def prepare_expression(expr):
        if isinstance(expr, sp.Basic):
            return expr
        return sp.sympify(str(expr))

    @staticmethod
    def evaluate(expr, meshes, param_map, symbols):
        """Substitute parameters, lambdify, and evaluate on meshes. Returns 1e4 on error."""
        try:
            expr = ExpressionEvaluator.prepare_expression(expr)
            expr = expr.subs({sp.Symbol(k): v for k, v in param_map.items()}).evalf(4)
            func = sp.lambdify(list(symbols.values()), expr, modules=['numpy'])
            return func(*[np.asarray(m) for m in meshes])
        except Exception:
            return np.full_like(meshes[0], 1e4)
