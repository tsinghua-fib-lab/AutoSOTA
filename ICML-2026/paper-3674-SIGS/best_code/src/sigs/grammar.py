import torch
from torch.autograd import Variable
from nltk import CFG, Nonterminal


grammar = """S -> S '+' T
S -> S '*' T
S -> S '/' T
S -> S '-' T
S -> T
S -> '-' T

T -> '(' S ')'
T -> '(' S ')^2'
T -> 'sin(' S ')'
T -> 'exp(' S ')'
T -> 'log(' S ')'
T -> 'cos(' S ')'
T -> 'sqrt(' S ')'
T -> 'tanh(' S ')'

T -> T '^' D 
T -> 'pi'
T -> 'x'
T -> 'y'
T -> 't'
T -> 'x^2'
T -> 'x^3' 
T -> 'y^2' 
T -> 'y^3' 

T -> D
T -> D '.' D

T -> '-' D
T -> '-' D '.' D



T -> T D

D -> D '0' | D '1' | D '2' | D '3' | D '4' | D '5' | D '6' | D '7' | D '8' | D '9'
D -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'

D -> 'e-1' | 'e-2' | 'e-3' | 'e-4'

Nothing -> None
"""

GCFG = CFG.fromstring(grammar)

S, T, D = Nonterminal('S'), Nonterminal('T'),  Nonterminal('D')

def get_mask(nonterminal, grammar, as_variable=False):
    if isinstance(nonterminal, Nonterminal):
        mask = [rule.lhs() == nonterminal for rule in grammar.productions()]
        mask = Variable(torch.FloatTensor(mask)) if as_variable else mask
        return mask
    else:
        raise ValueError('Input must be instance of nltk.Nonterminal')

if __name__ == '__main__':
    # Usage:
    GCFG = nltk.CFG.fromstring(grammar)

    print(get_mask(T))
    

