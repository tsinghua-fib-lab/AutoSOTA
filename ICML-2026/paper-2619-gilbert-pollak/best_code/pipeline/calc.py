from typing import Dict, List, Tuple
from sympy import Symbol, expand, sqrt
import sympy
import re
import os, argparse, sys

class Split:

    def __init__(self, mp):
        self.S_minus = mp['S_minus']
        self.S_plus = mp['S_plus']
        self.T_star = mp['T_star']
    
    @property
    def points(self):
        return sum([list(e) for e in self.S_minus + self.T_star], []) + list(self.S_plus)
    
    def __str__(self):
        return f'S_minus: {self.S_minus}; S_plus: {self.S_plus}; T_star: {self.T_star}'

def subsets(a, add_empty):
    n = len(a)
    results = []
    for mask in range(0 if add_empty else 1, 2 ** n):
        b = []
        for i in range(n):
            if (mask >> i) & 1:
                b.append(a[i])
        results.append(b)
    return results

def parse_split(s: str) -> Split:
    mp = dict()
    for part in s.split(';'):
        part = part.strip()
        if not part:
            continue
        key, value = part.split(':', 1)
        key = key.strip()
        mp[key] = eval(value)
    return Split(mp)

splits: List[Split] = []
with open('splits.txt') as f:
    s = f.readline()
    while s:
        splits.append(parse_split(s))
        s = f.readline()

# print(f'total splits: {len(splits)}')

class Expr:

    def __init__(self, expr, sympy_e = None):
        if isinstance(expr, Expr):
            self.expr = expr.expr
            self.sympy_e = expr.sympy_e
        else:
            self.expr = str(expr)
            if sympy_e != None:
                self.sympy_e = sympy_e
            else:
                if expr == 'Sqrt[3]':
                    expr = sqrt(3)
                self.sympy_e = Symbol(expr, nonnegative=True) if isinstance(expr, str) else expr
    
    def __mul__(self, other):
        if isinstance(other, Vector):
            return NotImplemented
        return Expr(f'({self})*({other})', self.sympy_e * other.sympy_e)
    
    def __truediv__(self, other):
        return Expr(f'({self})/({other})', self.sympy_e / other.sympy_e)
    
    def __add__(self, other):
        return Expr(f'({self})+({other})', self.sympy_e + other.sympy_e)
    
    def __sub__(self, other):
        return Expr(f'({self})-({other})', self.sympy_e - other.sympy_e)
    
    def __pow__(self, k):
        return Expr(f'({self})^({k})', self.sympy_e ** k)
    
    def __neg__(self):
        return Expr(f'-({self})', -self.sympy_e)
    
    def sqrt(self):
        return Expr(f'Sqrt[{self}]', sqrt(self.sympy_e))
    
    def __lt__(self, other):
        return Expr(f'{self} < {other}', self.sympy_e < other.sympy_e)
    
    def __le__(self, other):
        return Expr(f'{self} <= {other}', self.sympy_e <= other.sympy_e)
    
    def __invert__(self):
        return Expr(f'!({self})', ~self.sympy_e)
    
    def __and__(self, other):
        return Expr(f'{self} && {other}', self.sympy_e & other.sympy_e)
    
    def __or__(self, other):
        return Expr(f'({self} || {other})', self.sympy_e | other.sympy_e)
    
    def __str__(self):
        return self.expr

def Max(*args: Expr):
    return Expr(f"Max[{', '.join(str(arg) for arg in args)}]", sympy.Max(*[arg.sympy_e for arg in args]))

def Min(*args: Expr):
    return Expr(f"Min[{', '.join(str(arg) for arg in args)}]", sympy.Min(*[arg.sympy_e for arg in args]))

class Vector:

    def __init__(self, x, y):
        self.x = Expr(x)
        self.y = Expr(y)
    
    def __add__(self, other):
        return Vector(
            self.x + other.x,
            self.y + other.y
        )
    
    def __sub__(self, other):
        return Vector(
            self.x - other.x,
            self.y - other.y
        )
    
    def __neg__(self):
        return Vector(-self.x, -self.y)
    
    def __mul__(self, k):
        return Vector(self.x * k, self.y * k)
    
    def __rmul__(self, k):
        return Vector(self.x * k, self.y * k)
    
    def rotate_ccw_60(self):
        x = self.x
        y = self.y
        return Vector(
            (x-Expr('Sqrt[3]')*y)/Expr(2),
            (Expr('Sqrt[3]')*x+y)/Expr(2)
        )
    
    def rotate_cw_60(self):
        x = self.x
        y = self.y
        return Vector(
            (x+Expr('Sqrt[3]')*y)/Expr(2),
            (y-Expr('Sqrt[3]')*x)/Expr(2)
        )
    
    def norm2(self):
        return self.x**2 + self.y**2
    
    def __str__(self):
        return f'({self.x}, {self.y})'

def cross(a: Vector, b: Vector):
    return a.x * b.y - a.y * b.x

# 邻接表，邻居按顺时针方向存
adj: Dict[str, List[str]] = dict()
adj['A'] = ['s']
adj['B'] = ['s']
adj['s'] = ['A', 'B', 'r']
adj['r'] = ['s', 'D', 'P']
adj['P'] = ['r', 'Q', 'R']
adj['D'] = ['r']
adj['Q'] = ['P']
adj['R'] = ['P']

points = ['A', 'B', 'D', 's', 'r', 'P', 'Q', 'R', 'X', 'Y']
edges = [('A', 's'), ('B', 's'), ('s', 'r'), ('r', 'D'), ('r', 'P'), ('P', 'Q'), ('P', 'R')]

dir: Dict[Tuple[str, str], Vector] = dict()
dir[('A', 's')] = Vector(1, 0)
dir[('s', 'A')] = -dir[('A', 's')]

length: Dict[str, Expr] = dict()

# DFS 求出每条边的方向
def dfs0(u, fa):
    if len(adj[u]) == 1:
        return
    i = adj[u].index(fa)
    if len(adj[u]) == 2:
        v = adj[u][1-i]
        dir[(u, v)] = dir[(fa, u)]
        dir[(v, u)] = -dir[(u, v)]
        dfs0(v, u)
    elif len(adj[u]) == 3:
        v1 = adj[u][(i+1)%3]
        v2 = adj[u][(i+2)%3]
        dir[(u, v1)] = dir[(fa, u)].rotate_ccw_60()
        dir[(v1, u)] = -dir[(u, v1)]
        dfs0(v1, u)
        dir[(u, v2)] = dir[(fa, u)].rotate_cw_60()
        dir[(v2, u)] = -dir[(u, v2)]
        dfs0(v2, u)
    else:
        assert False, "Degree cannot be greater than 3"

# 求出每条边的方向
dfs0('s', 'A')

def setlength(u, v, l):
    length[(u, v)] = Expr(l)
    length[(v, u)] = Expr(l)

vars = ('b', 'c', 'd', 's', 'e')

setlength('A', 's', 1)
setlength('B', 's', 'b')
setlength('s', 'r', 's')
setlength('r', 'P', 'c')
setlength('r', 'D', 'd')
setlength('P', 'R', 'e')
setlength('P', 'Q', 'f')

position: Dict[str, Vector] = dict()
position['A'] = Vector(0, 0)

edge_set_anc: Dict[str, set] = dict()
edge_set_anc['A'] = set()

# DFS 求出每个点的位置
def dfs1(u, fa):
    for v in adj[u]:
        if v == fa:
            continue
        position[v] = position[u] + length[(u, v)] * dir[(u, v)]
        if str(length[(u, v)]) != '1':
            edge_set_anc[v] = edge_set_anc[u].union(str(length[(u, v)]))
        else:
            edge_set_anc[v] = edge_set_anc[u]
        dfs1(v, u)

# 求出每个点的位置
dfs1('A', None)

# u 到 v 的向量
def vec(u, v):
    return position[v] - position[u]

dist2_upper_bound: Dict[Tuple[str, str], Expr] = dict()

# u 到 v 的距离的平方
def dist2(u, v) -> Expr:
    if u in position and v in position:
        return vec(u, v).norm2()
    if (u, v) in dist2_upper_bound:
        return dist2_upper_bound[(u, v)]
    if (v, u) in dist2_upper_bound:
        return dist2_upper_bound[(v, u)]
    return Expr('1e18')

def edge_set(u, v) -> set:
    return edge_set_anc[u] ^ edge_set_anc[v]

# ∠ABC > 120° 的条件
def angle_gt_120(A, B, C) -> Expr:
    AB2 = dist2(A, B)
    BC2 = dist2(B, C)
    AC2 = dist2(A, C)
    return AB2+BC2+(AB2*BC2).sqrt()-AC2<Expr(0)

rho = Expr('rho')

# 检查 split 对应的函数是否关于 var 单调
def check_mono(split: Split, var: str):
    if 'X' in split.points or 'Y' in split.points:
        return False
    cnt_s_minus = 0
    cnt_s_plus = 0
    cnt_t = 0
    for u, v in split.S_minus:
        if var in edge_set(u, v):
            cnt_s_minus = 1
    for u, v in split.T_star:
        if var in edge_set(u, v):
            cnt_t += 1
    S_plus = split.S_plus
    assert len(S_plus) <= 3
    for u, v in zip(S_plus, S_plus[1:]):
        if var in edge_set(u, v):
            cnt_s_plus = 1
    if cnt_t + cnt_s_minus + cnt_s_plus == 0:
        return True
    if cnt_t + cnt_s_plus > 1:
        return False
    return cnt_s_minus > 0

def conv(e: Expr):
    e = e.sympy_e
    if e.func == sympy.And:
        e = sympy.And(*[expand(arg) for arg in e.args])
    else:
        e = expand(e)
    return re.sub(r'([^*\s()]+)\*\*2', r'\1*\1', str(e)).replace('Max', 'max').replace('Min', 'min').replace('True', 'true').replace('sqrt(3)', 'sqrt3').replace('|', '||').replace('&', '&&')

def steiner_of_three(A, B, C):
    codes = []
    def D(code: str): codes.append(code)
    AB2 = dist2(A, B)
    BC2 = dist2(B, C)
    AC2 = dist2(A, C)
    D(f'ld L_s_plus;')
    D(f'ld AB2 = {conv(AB2)};')
    D(f'ld BC2 = {conv(BC2)};')
    D(f'ld CA2 = {conv(AC2)};')
    D(f'if(AB2 + BC2 + sqrt(AB2 * BC2) < CA2) L_s_plus = sqrt(AB2)+sqrt(BC2);')
    D(f'else if(BC2 + CA2 + sqrt(BC2 * CA2) < AB2) L_s_plus = sqrt(BC2)+sqrt(CA2);')
    D(f'else if(CA2 + AB2 + sqrt(CA2 * AB2) < BC2) L_s_plus = sqrt(CA2)+sqrt(AB2);')
    D(f'else {{')
    D(f'if({conv(cross(vec(A, B), vec(A, C)) / Expr("Sqrt[3]") * Expr(2) <= Expr(0))}) ')
    D(f'L_s_plus = {conv((position[B] + vec(B, C).rotate_ccw_60() - position[A]).norm2().sqrt())};')
    D(f'else ')
    D(f'L_s_plus = {conv((position[C] + vec(C, B).rotate_ccw_60() - position[A]).norm2().sqrt())};')
    D(f'}}')
    return codes

# 由 split 这个划分得出的不等式
def induction_cond(split: Split, s_plus_calc = None):
    L_s_minus = sum([length[(u, v)] for u, v in split.S_minus], Expr(0))
    L_t = sum([dist2(u, v).sqrt() for u, v in split.T_star], Expr(0))
    codes = []
    def D(code: str): codes.append(code)
    D(f'ld L_s_minus = {conv(L_s_minus)};')
    D(f'ld L_t = {conv(L_t)};')
    if s_plus_calc is not None:
        D(s_plus_calc)
    elif len(split.S_plus) <= 1:
        L_s_plus = Expr(0)
        D(f'ld L_s_plus = 0;')
        # print(f'{conv(rho*L_t + L_s_plus - L_s_minus)},')
    elif len(split.S_plus) == 2:
        L_s_plus = dist2(*split.S_plus).sqrt()
        D(f'ld L_s_plus = {conv(L_s_plus)};')
        # print(f'{conv(rho*L_t + L_s_plus - L_s_minus)},')
    elif len(split.S_plus) == 3:
        A, B, C = split.S_plus
        codes.extend(steiner_of_three(A, B, C))
    else:
        assert False, "Size of S_plus cannot be greater than 3"
    D(f'return rho*L_t + L_s_plus - L_s_minus;')
    return codes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--splus', action='store_true', help='add a new s_plus, cond and s_plus should be stored at tmp/s_cond and tmp/s_plus')
    parser.add_argument('--regular-point', action='store_true', help='add a new regular point, cond should be stored at tmp/x_cond')

    args = parser.parse_args()
    if not args.splus and not args.regular_point:
        print('At least one of --splus and --regular-point should be set.')
        print('Run `python calc.py --help` to see the usage.')
        exit(0)
    elif args.splus and args.regular_point:
        print('Only one of --splus and --regular-point should be set.')
        print('Run `python calc.py --help` to see the usage.')
        exit(0)

    file_id = 0
    dir = 'formulas'
    for file in os.listdir(dir):
        if file.startswith('F'):
            file_id = max(file_id, int(file[1:]))
    with open(os.path.join(dir, f'F{file_id}'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                assert line.startswith(f'const int M{file_id} = ') and line.endswith(';')
                id = int(line[line.find('=') + 1 : -1])
                break

    file_id += 1
    with open(os.path.join(dir, f'F{file_id}'), 'w', encoding='utf-8') as f:
    
        # S_plus with size 4
        if args.splus:
            
            with open('tmp/s_cond', 'r', encoding='utf-8') as ff:
                s_cond = ff.read()
            with open('tmp/s_plus', 'r', encoding='utf-8') as ff:
                s_plus = set(eval(ff.read()))

            f.write(f"""\
namespace __f{file_id}_detail {{
{s_cond}
}}""")
            f.write('\n')

            for i in range(len(splits)):
                split = splits[i]
                if set(split.S_plus) != s_plus:
                    continue
                exist_X = False
                exist_Y = False
                for (u, v) in split.T_star:
                    if u == 'X' or v == 'X':
                        exist_X = True
                for (u, v) in split.T_star:
                    if u == 'Y' or v == 'Y':
                        exist_Y = True
                if exist_X or exist_Y:
                    continue
                codes = induction_cond(split, f"""\
ld L_s_plus;
if(!__f{file_id}_detail::steiner_cond(b, c, d, s, e)) L_s_plus = INF;
else L_s_plus = __f{file_id}_detail::steiner_length(b, c, d, s, e);""")
                codes.insert(0, f"""\
template<> struct F<{id}> {{
ld operator()(ull mono_mask, ld b, ld c, ld d, ld s, ld e, ld f) {{
if(F_VAL == 1 || mono_mask != 0) return INF;""")
                codes.append('}};\n')
                id += 1

                f.write('\n'.join(codes))
                f.write('\n')

        # regular point
        elif args.regular_point:

            with open('tmp/x_cond', 'r', encoding='utf-8') as ff:
                X_cond = ff.read()
            f.write(f"""\
namespace __f{file_id}_detail {{
{X_cond}
}}""")
            f.write('\n')

            for i in range(len(splits)):
                split = splits[i]
                if len(split.S_plus) > 3:
                    continue

                mono_mask = 0
                for j, var in enumerate(vars):
                    if check_mono(split, var):
                        mono_mask += 1<<j

                exist_X = False
                exist_Y = False
                for (u, v) in split.T_star:
                    if u == 'X' or v == 'X':
                        exist_X = True
                for (u, v) in split.T_star:
                    if u == 'Y' or v == 'Y':
                        exist_Y = True
                cond = None
                if exist_X:
                    conde = f'__f{file_id}_detail::X_cond(s,c,e)'
                    if cond:
                        cond = f'{cond} & {conde}'
                    else:
                        cond = conde
                    dist2_upper_bound[('A', 'X')] = Expr('A_X')**2
                    dist2_upper_bound[('D', 'X')] = Expr('D_X')**2
                if exist_Y:
                    condf = '(F_VAL == 1 ? true : false)'
                    if cond:
                        cond = f'{cond} & {condf}'
                    else:
                        cond = condf
                    dist2_upper_bound[('D', 'Y')] = Max(dist2('D', 'P'), dist2('D', 'Q'))
                if cond is None:
                    cond = 'true'
                codes = induction_cond(split)
                codes.insert(0, f"""\
template<> struct F<{id}> {{
ld operator()(ull mono_mask, ld b, ld c, ld d, ld s, ld e, ld f) {{
if(!({cond}) || (mono_mask | {mono_mask}) != {mono_mask}) return INF;
ld A_X = __f{file_id}_detail::AX_upper_bound(s, c, e);
ld D_X = __f{file_id}_detail::DX_upper_bound(s, c, e, d);
""")
                codes.append('}};\n')
                id += 1

                f.write('\n'.join(codes))
                f.write('\n')

        f.write(f'const int M{file_id} = {id};')
        f.write('\n')
