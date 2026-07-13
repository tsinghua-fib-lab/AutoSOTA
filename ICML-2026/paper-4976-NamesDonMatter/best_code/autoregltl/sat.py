# The contents of this file is modified from the files in the examples/random_formulas directory of py-aiger:
# https://github.com/MarkusRabe/py-aiger
# which is licensed under the following:
#
# MIT License
#
# Copyright (c) 2018 Marcell Vazquez-Chanlat
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import time

import spot
import aiger
import aiger_sat

from autoregltl.ltl.parser import ParseError


class Generator():
    def __init__(self, num_variables=10, bool_constants=['True', 'False'], unary_operators=['neg'], binary_operators=['or', 'and', 'xor', 'eq']):
        self.num_variables = num_variables
        self.variables = ['var%02d' % v for v in range(self.num_variables)]
        self.bool_constants = bool_constants.copy()
        self.unary_operators = unary_operators.copy()
        self.binary_operators = binary_operators.copy()

    def to_expression(self, token_sequence): # pre-ordered list
        token_sequence = iter(token_sequence)
        elem = next(token_sequence, None)
        if elem is None:
            raise ValueError('Sequence ends before expression is complete.')
        if elem in self.variables:
            return aiger.atom(elem)
        if elem in self.bool_constants:
            return aiger.atom(elem == 'True')
        if elem in self.unary_operators:
            assert elem == 'neg'
            return ~ self.to_expression(token_sequence)
        assert elem in self.binary_operators, 'Unknown op: %s' % elem
        left = self.to_expression(token_sequence)
        right = self.to_expression(token_sequence)
        if elem == 'or':
            return left | right
        if elem == 'and':
            return left & right
        if elem == 'xor':
            return left ^ right
        if elem == 'eq':
            return left == right
        raise ValueError('Should not reach this point')


def minimize_model(formula, model):
    if model is None:
        return None
    solver = aiger_sat.SolverWrapper()
    solver.add_expr(~formula)
    if solver.is_sat(assumptions=model):
        raise ValueError('UNSAT core generation failed.')
    minimized = solver.get_unsat_core()
    if minimized is None:
        minimized = {}
    return minimized

def generate_model(formula_pre, generator=None):
    start_time = time.time()
    if generator is None:
        generator = Generator()
    expr = generator.to_expression(formula_pre)
    solver = aiger_sat.SolverWrapper()
    try:
        solver.add_expr(expr)
    except ValueError as e:
        print('ValueError', str(e))
        return None
    model = solver.get_model()
    model_str = None
    if model:
        model_word_pairs = ['%s %s' % x for x in sorted(model.items(), key=lambda x: x[0])]
        model_str = ' '.join(model_word_pairs)

    minimized = minimize_model(expr, model)
    minimized_str = None
    if minimized:
        minimized_word_pairs = ['%s %s' % x for x in sorted(minimized.items(), key=lambda x: x[0])]
        minimized_str = ' '.join(minimized_word_pairs)
    return minimized_str, time.time() - start_time

def is_model(polish_formula, model, generator=None):
    if generator is None:
        generator = Generator()
    formula = generator.to_expression(polish_formula)
    solver = aiger_sat.SolverWrapper()
    solver.add_expr(~formula)
    return not solver.is_sat(assumptions=model)


## pyaiger <-> spot conversion things
spot_to_pyaiger_dict = {'1':'True', '0':'False', '!':'neg', '<->':'eq', 'xor':'xor', '&':'and', '|':'or'}
pyaiger_to_spot_dict = {val : key for key, val in spot_to_pyaiger_dict.items()}
spot_to_pyaiger_dict |= {'=':'eq', '^':'xor'}

def spot_to_pyaiger(token_list):
    if isinstance(token_list, str):
        token_list = token_list.replace("<->", "=").replace("xor", "^")
    res = []
    for token in token_list:
        if token in spot_to_pyaiger_dict:
            res.append(spot_to_pyaiger_dict[token])
        else:
            n = ord(token) - 97
            if n >= 26:
                raise ValueError()
            res.append(f'var{n:02}')
    return res

def pyaiger_to_spot(token_list):
    res = []
    for token in token_list:
        if token in pyaiger_to_spot_dict:
            res.append(pyaiger_to_spot_dict[token])
        else:
            if not token.startswith('var') or len(token) != 5:
                raise ValueError('Expected varXX')
            n = int(token[3:])
            if n >= 26:
                raise ValueError()
            res.append(chr(n + 97))
    return res

def get_assignments(lst):
    if len(lst) % 2 != 0:
        raise ParseError('length of assignments not even')
    iterator = iter(lst)
    assignments = {}
    for var in iterator:
        val = next(iterator)
        if val == 'True' or val == '1':
            assignments[var] = True
        elif val == 'False' or val == '0':
            assignments[var] = False
        else:
            raise ParseError('assignment var not True or False')
    return assignments