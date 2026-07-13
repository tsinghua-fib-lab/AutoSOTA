"""
Parsers for LTL formulas and traces, for both spot and the neural network.
"""

from enum import Enum
import re
from functools import reduce
import operator

# Regular expression for allowed APs. Can disallow 't' and 'f', but all lowercase characters should be fine
ap_re = '[a-z]'
# ap_re = '[a-eg-su-z]'

network_step_re = '([&|!|10]|' + ap_re + ')+'
network_output_re = '(' + network_step_re + ';)*' + '[{]' + '(' + network_step_re + ';)*' + network_step_re + '[}]'


def ltl_formula(formula_string: str, format: str) -> 'LTLFormula':
    """Parse a LTL formula in the specified format (spot, lbt, network-infix, network-polish)"""
    token_list = tokenize_formula(formula_string, format.split('-')[0])
    if format == 'spot' or format == 'network-infix':
        tree, remainder = parse_infix_formula(token_list)
    elif format == 'lbt' or format == 'network-polish':
        tree, remainder = parse_polish_formula(token_list)
    else:
        raise ValueError("'format' must be one of: spot, lbt, network-infix, network-polish")
    if remainder:
        raise ParseError("Could not fully parse formula, remainder: '" + str(remainder) + "'")
    return tree


class LTLFormula():
    """Represents a parsed LTL formula, use to_str() to get a representation in the desired format (spot, lbt, network-infix, network-polish)"""

    def __str__(self):
        return self.to_str(format='spot')

    def to_str(self, format, spacing=None, full_parens=False) -> str:  # spacing: 'none' (a&X!b), 'binary ops' (a & X!b), 'all ops' (a & X ! a)
        if format == 'spot':
            return self._to_str('infix', 'spot', spacing=spacing if spacing is not None else 'binary ops', full_parens=full_parens)
        elif format == 'lbt':
            return self._to_str('polish', 'lbt', spacing=spacing if spacing is not None else 'all ops', full_parens=full_parens)
        elif format == 'network-infix':
            if spacing is None:
                spacing = 'none'
            return self._to_str('infix', 'network', spacing=spacing, full_parens=full_parens)
        elif format == 'network-polish':
            if spacing is None:
                spacing = 'none'
            return self._to_str('polish', 'network', spacing=spacing, full_parens=full_parens)
        else:
            raise ValueError("Unrecognized format")

    def binary_position_list(self, format, add_first):
        """Returns a list of binary lists where each list represents the position of a node in the abstract syntax tree as a sequence of steps along tree branches starting in the root node with each step going left ([1,0]) or going right([0,1]). The ordering of the lists is given by the format. add_first specifies whether branching choices are added at the first position or at the last position of a list.
        Example: Given the formula aUb&Xc depeding on the parameters the following lists are returned.
            format=spot, add_first=True: [[1,0,1,0],[1,0],[0,1,1,0],[],[0,1],[1,0,0,1]]
            format=spot, add_first=False: [[1,0,1,0],[1,0],[1,0,0,1],[],[0,1],[0,1,1,0]]
            format=lbt, add_first=True: [[], [1,0], [1,0,1,0], [0,1,1,0], [0,1], [1,0,0,1]]
            format=lbt, add_first=False: [[],[1,0],[1,0,1,0],[1,0,0,1],[0,1],[0,1,1,0]]
        """
        raise NotImplementedError()

    def _to_str(self, notation, format_, spacing, full_parens):
        raise NotImplementedError()

    def equal_to(self, other: 'LTLFormula', extended_eq=False):
        raise NotImplementedError()

    def size(self):
        raise NotImplementedError()

    def contained_aps(self):
        raise NotImplementedError()

    def rewrite(self, token):
        raise NotImplementedError()

    def __add__(self, other):
        if other is None:
            return self
        if not isinstance(other, LTLFormula):
            raise ValueError('and operand is no formula')
        return LTLFormulaBinaryOp(Token.AND, self, other)

    def __radd__(self, other):
        return self.__add__(other)


def ltl_trace(trace_string, format):
    """Parse a LTL trace in the specified format (spot, lbt, network-infix, network-polish)"""
    if format.startswith('network'):
        if not re.fullmatch(network_output_re, trace_string):
            raise ParseError("Network output did not match regex")

    m = re.fullmatch('(.*)' + ('cycle' if not format.startswith('network') else '') + '[{](.+)[}]', trace_string)
    if m is None:
        raise ParseError("Could not match cycle part of trace")
    prefix_steps = m.groups()[0].split(';')
    cycle_steps = m.groups()[1].split(';')
    prefix_steps = prefix_steps[:-1]  # remove element past last ;
    prefix_formulas = [ltl_formula(step_string, format=format) for step_string in prefix_steps]
    cycle_formulas = [ltl_formula(step_string, format=format) for step_string in cycle_steps]
    return LTLTrace(prefix_formulas, cycle_formulas)


class LTLTrace():
    """Represents a parsed LTL trace, use to_str() to get a representation in the desired format (spot, network-infix, network-polish)"""

    def __init__(self, prefix_formulas, cycle_formulas):
        self.prefix_formulas = prefix_formulas
        self.cycle_formulas = cycle_formulas

    def to_str(self, format):
        space = '' if format.startswith('network') else ' '
        cycle_start = '{' if format.startswith('network') else 'cycle{'
        prefix_strings = [step.to_str(format) for step in self.prefix_formulas]
        cycle_strings = [step.to_str(format) for step in self.cycle_formulas]
        if len(prefix_strings) > 0:
            prefix_strings.append('')  # for trailing ;
        return (';' + space).join(prefix_strings) + cycle_start + (';' + space).join(cycle_strings) + '}'

    def equal_to(self, other: 'LTLTrace', extended_eq=False):
        return len(self.prefix_formulas) == len(other.prefix_formulas) \
            and len(self.cycle_formulas) == len(other.cycle_formulas) \
            and all([my_step.equal_to(other_step, extended_eq=extended_eq) for my_step, other_step in zip(self.prefix_formulas, other.prefix_formulas)]) \
            and all([my_step.equal_to(other_step, extended_eq=extended_eq) for my_step, other_step in zip(self.cycle_formulas, other.cycle_formulas)])

    def contained_aps(self):
        return (reduce(operator.or_, [q.contained_aps() for q in self.prefix_formulas]) if self.prefix_formulas else set()) | reduce(operator.or_, [q.contained_aps() for q in self.cycle_formulas])


class ParseError(Exception):
    pass


Token = Enum('Node', 'NOT AND OR IMPLIES EQUIV XOR NEXT UNTIL RELEASE GLOBALLY FINALLY TRUE FALSE AP LPAR RPAR STEP CYCLE')

token_dict_spot = {'!':(1, Token.NOT), '&':(2, Token.AND), '|':(2, Token.OR), '->':(2, Token.IMPLIES), '<->':(2, Token.EQUIV), 'xor':(2, Token.XOR), 'X':(1, Token.NEXT), 'U':(2, Token.UNTIL),
        'R':(2, Token.RELEASE), 'G':(1, Token.GLOBALLY), 'F':(1, Token.FINALLY), '1':(0, Token.TRUE), '0':(0, Token.FALSE), '(':(-1, Token.LPAR), ')':(-1, Token.RPAR)}
token_reverse_dict_spot = {token: ch for ch, (nnum_children, token) in token_dict_spot.items()}
token_dict_network = token_dict_spot
token_reverse_dict_network = token_reverse_dict_spot
token_dict_lbt = {'!':(1, Token.NOT), '&':(2, Token.AND), '|':(2, Token.OR), 'i':(2, Token.IMPLIES), 'e':(2, Token.EQUIV), '^':(2, Token.XOR), 'X':(1, Token.NEXT), 'U':(2, Token.UNTIL),
        'R':(2, Token.RELEASE), 'G':(1, Token.GLOBALLY), 'F':(1, Token.FINALLY), 't':(0, Token.TRUE), 'f':(0, Token.FALSE)}
token_reverse_dict_lbt = {token: ch for ch, (nnum_children, token) in token_dict_lbt.items()}

precedence = {Token.NOT : 1, Token.AND : 3, Token.OR : 4, Token.IMPLIES : 5, Token.EQUIV : 5, Token.XOR : 5, Token.NEXT : 1, Token.UNTIL : 2, Token.RELEASE : 2, Token.GLOBALLY : 1, Token.FINALLY : 1, Token.TRUE : 0, Token.FALSE : 0, Token.AP : 0} # higher number = weaker
left_associative = {Token.AND : True, Token.OR: True, Token.IMPLIES : False, Token.EQUIV : None, Token.XOR : None, Token.UNTIL : False, Token.RELEASE : False}

class LTLFormulaBinaryOp(LTLFormula):
    def __init__(self, type_, lchild, rchild):
        self.type_ = type_
        self.lchild = lchild
        self.rchild = rchild
        self.precedence = precedence[type_]
        self.left_associative = left_associative[type_]

    def _to_str(self, notation, format_, spacing, full_parens):
        space = '' if spacing == 'none' else ' '
        if notation == 'polish':
            return globals()['token_reverse_dict_' + format_][self.type_] + space + self.lchild._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens) + space + self.rchild._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens)
        elif notation == 'infix':
            if full_parens or self.lchild.precedence > self.precedence:
                par_left = True
            elif self.lchild.precedence == self.precedence:
                par_left = self.left_associative is None or not self.left_associative
            else:
                par_left = False
            if full_parens or self.rchild.precedence > self.precedence:
                par_right = True
            elif self.rchild.precedence == self.precedence:
                par_right = self.left_associative is None or self.left_associative
            else:
                par_right = False
            return ('(' if par_left else '') + self.lchild._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens) + (')' if par_left else '') + space + globals()['token_reverse_dict_' + format_][self.type_] + space + ('(' if par_right else '') + self.rchild._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens) + (')' if par_right else '')
        else:
            raise ValueError("Unrecognized notation")

    def equal_to(self, other: LTLFormula, extended_eq=False):
        if not isinstance(other, LTLFormulaBinaryOp) or not self.type_ == other.type_:
            return False
        children_equal = self.lchild.equal_to(other.lchild, extended_eq=extended_eq) and self.rchild.equal_to(other.rchild, extended_eq=extended_eq)
        if extended_eq and self.type_ in [Token.AND, Token.OR]:
            children_equal = children_equal or (self.lchild.equal_to(other.rchild, extended_eq=extended_eq) and self.rchild.equal_to(other.lchild, extended_eq=extended_eq))
        return children_equal

    def size(self):
        return 1 + self.lchild.size() + self.rchild.size()

    def contained_aps(self):
        return self.lchild.contained_aps() | self.rchild.contained_aps()

    def rewrite(self, token):
        lchild_r = self.lchild.rewrite(token)
        rchild_r = self.rchild.rewrite(token)
        if self.type_ == token:
            if token == Token.OR:
                return F_NOT(F_AND(F_NOT(lchild_r), F_NOT(rchild_r)))
            else:
                raise ValueError("Don't know how to rewrite " + str(token))
        else:
            return LTLFormulaBinaryOp(self.type_, lchild_r, rchild_r)

    def binary_position_list(self, format, add_first):
        lchild_pos_list = self.lchild.binary_position_list(format, add_first)
        rchild_pos_list = self.rchild.binary_position_list(format, add_first)
        if add_first:
            # due to the recursion the branching choice is added at the end of the list if add_first is true
            lchild_pos_list = [l + [1, 0] for l in lchild_pos_list]
            rchild_pos_list = [l + [0, 1] for l in rchild_pos_list]
        else:
            lchild_pos_list = [[1, 0] + l for l in lchild_pos_list]
            rchild_pos_list = [[0, 1] + l for l in rchild_pos_list]
        if format == 'lbt':
            return [[]] + lchild_pos_list + rchild_pos_list
        elif format == 'spot':
            return lchild_pos_list + [[]] + rchild_pos_list
        else:
            raise ValueError("Unrecognized format")


class LTLFormulaUnaryOp(LTLFormula):
    def __init__(self, type_, child):
        self.type_ = type_
        self.child = child
        self.precedence = precedence[type_]

    def _to_str(self, notation, format_, spacing, full_parens):
        space = '' if spacing in ['none', 'binary ops'] else ' '
        if notation == 'polish':
            return globals()['token_reverse_dict_' + format_][self.type_] + space + self.child._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens)
        elif notation == 'infix':
            par = (self.child.precedence > self.precedence) or full_parens
            return globals()['token_reverse_dict_' + format_][self.type_] + space + ('(' if par else '') + self.child._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens) + (')' if par else '')
        else:
            raise ValueError("Unrecognized notation")

    def equal_to(self, other: LTLFormula, extended_eq=False):
        return isinstance(other, LTLFormulaUnaryOp) and self.type_ == other.type_ and self.child.equal_to(other.child, extended_eq=extended_eq)

    def size(self):
        return 1 + self.child.size()

    def contained_aps(self):
        return self.child.contained_aps()

    def rewrite(self, token):
        child_r = self.child.rewrite(token)
        if self.type_ == token:
            if token == Token.GLOBALLY:
                raise NotImplementedError()
            else:
                raise ValueError("Don't know how to rewrite " + str(token))
        else:
            return LTLFormulaUnaryOp(self.type_, child_r)

    def binary_position_list(self, format, add_first):
        child_pos_list = self.child.binary_position_list(format, add_first)
        if add_first:
            child_pos_list = [l + [1, 0] for l in child_pos_list]
        else:
            child_pos_list = [[1, 0] + l for l in child_pos_list]
        if format == 'lbt' or format == 'spot':
            return [[]] + child_pos_list
        else:
            raise ValueError("Unrecognized format")


class LTLFormulaLeaf(LTLFormula):
    def __init__(self, type_, ap=None):
        self.type_ = type_
        self.ap = ap
        self.precedence = precedence[type_]

    def _to_str(self, notation, format_, spacing, full_parens):
        if not self.type_ == Token.AP:
            return globals()['token_reverse_dict_' + format_][self.type_]
        if format_ == 'lbt':
            return '"' + self.ap + '"'
        elif format_ == 'spot' or format_ == 'network':
            return self.ap
        else:
            raise ValueError("'format' must be either spot or lbt")

    def equal_to(self, other: LTLFormula, extended_eq=False):
        return isinstance(other, LTLFormulaLeaf) and self.type_ == other.type_ and self.ap == other.ap

    def size(self):
        return 1

    def contained_aps(self):
        if self.type_ == Token.AP:
            return {self.ap}
        else:
            return set()

    def rewrite(self, token):
        if self.type_ == token:
            raise ValueError("Cannot rewrite a " + str(token))
        else:
            return LTLFormulaLeaf(self.type_, ap=self.ap)

    def binary_position_list(self, format, add_first):
        return [[]]


def tokenize_formula(formula_string, format_):
    token_list = []
    while formula_string:
        if len(formula_string) >= 3:
            token = globals()['token_dict_' + format_].get(formula_string[:3]) # 3 character match (ugly, damn)
            if token:
                token_list.append(token)
                formula_string = formula_string[3:]
                continue
        if len(formula_string) >= 2:
            token = globals()['token_dict_' + format_].get(formula_string[:2]) # 2 character match (ugly, damn)
            if token:
                token_list.append(token)
                formula_string = formula_string[2:]
                continue
        c = formula_string[:1]
        formula_string = formula_string[1:]
        if c.isspace() and format_ != 'network':
            continue
        token = globals()['token_dict_' + format_].get(c)
        if token:
            token_list.append(token)
        elif (format_ == 'spot' or format_ == 'network') and re.match(ap_re, c):  # check for AP a
            token_list.append((0, Token.AP, c))
        elif format_ == 'lbt' and len(formula_string) >= 2 and re.match('"' + ap_re + '"', c + formula_string[0] + formula_string[1]):  # check for AP "a"
            token_list.append((0, Token.AP, formula_string[0]))
            formula_string = formula_string[2:]
        else:
            raise ParseError("Cannot tokenize '" + c + "', remainder '" + formula_string + "'")
    return token_list


def parse_polish_formula(token_list):
    if len(token_list) == 0:
        raise ParseError('Attempt to parse from empty token list')
    num_children, type_, *name = token_list.pop(0)
    if num_children == 2:
        lchild, token_list = parse_polish_formula(token_list)
        rchild, token_list = parse_polish_formula(token_list)
        return LTLFormulaBinaryOp(type_, lchild, rchild), token_list
    elif num_children == 1:
        child, token_list = parse_polish_formula(token_list)
        return LTLFormulaUnaryOp(type_, child), token_list
    elif num_children == 0:
        if type_ == Token.AP:
            return LTLFormulaLeaf(type_, ap=name[0]), token_list
        else:
            return LTLFormulaLeaf(type_, ap=None), token_list
    else:
        raise ParseError("Illegal token '" + str(type_) + "'")


def parse_infix_formula(token_list, expect_rpar=False):
    # first part, until possible first binary op
    node, token_list = infix_parse_single(token_list)
    if len(token_list) == 0:
        if expect_rpar:
            raise ParseError("Parsing error: End of string but expected RPAR")
        else:
            return node, []
    num_children, type_, *_ = token_list.pop(0)
    if expect_rpar and type_ == Token.RPAR:
        return node, token_list
    if num_children != 2:
        raise ParseError("Parsing error: Binary operator expected, got " + str(type_) + ", remainder: " + str(token_list))

    # main part, at least one binary op
    stack = [(node, type_)]
    while True:
        current_node, token_list = infix_parse_single(token_list)
        l = len(token_list)
        if l == 0 and expect_rpar:
            raise ParseError("Parsing error: End of string but expected RPAR")
        if l > 0:
            num_children, right_op, *_ = token_list.pop(0)
        if l == 0 or (expect_rpar and right_op == Token.RPAR):  # finished
            assert len(stack) > 0
            while len(stack) > 0:
                left_node, left_op = stack.pop()
                current_node = LTLFormulaBinaryOp(left_op, left_node, current_node)
            return current_node, token_list

        # not yet finished, binary op present
        if num_children != 2:
            raise ParseError("Parsing error: Binary operator expected, got " + str(right_op) + ", remainder: " + str(token_list))
        left_node, left_op = stack[-1]
        while precedence[left_op] < precedence[right_op] or (precedence[left_op] == precedence[right_op] and left_associative[left_op]):  # left is stronger, apply left
            current_node = LTLFormulaBinaryOp(left_op, left_node, current_node)
            stack.pop()
            if len(stack) == 0:
                break
            left_node, left_op = stack[-1]
        stack.append((current_node, right_op))


def infix_parse_single(token_list):
    if len(token_list) == 0:
        raise ParseError('Attempt to parse from empty token list (trailing part missing?)')
    num_children, type_, *name = token_list.pop(0)
    if num_children == 2:
        raise ParseError("Parsing error: Binary operator at front (" + str(type_) + "), remainder: " + str(token_list))
    elif num_children == 1:
        child, token_list = infix_parse_single(token_list)
        return LTLFormulaUnaryOp(type_, child), token_list
    elif num_children == 0:
        if type_ == Token.AP:
            return LTLFormulaLeaf(type_, ap=name[0]), token_list
        else:
            return LTLFormulaLeaf(type_, ap=None), token_list
    else:
        if type_ == Token.RPAR:
            raise ParseError("Parsing error: RPAR at front, remainder: " + str(token_list))
        if type_ == Token.LPAR:
            return parse_infix_formula(token_list, expect_rpar=True)


def F_AND(x, y):
    return LTLFormulaBinaryOp(Token.AND, x, y)


def F_IMLIES(x, y):
    return LTLFormulaBinaryOp(Token.IMPLIES, x, y)


def F_NEXT(x):
    return LTLFormulaUnaryOp(Token.NEXT, x)


def F_GLOBALLY(x):
    return LTLFormulaUnaryOp(Token.GLOBALLY, x)


def F_NOT(x):
    return LTLFormulaUnaryOp(Token.NOT, x)


def F_AP(s):
    return LTLFormulaLeaf(Token.AP, s)
