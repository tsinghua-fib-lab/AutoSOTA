import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import numpy as np
from logdiff.score.sampling_compositional import And, Or_CI, Or_ME, Not
from typing import List, Dict

import random

class ComplexQueryGenerator:
    not_prob = 0.05

    def __init__(self, group_names: List[str], attribute_classes: Dict[str, type], attribute_options: Dict[str, int], seed: int = 42):
        self.group_names = group_names
        self.attribute_classes = attribute_classes
        self.attribute_options = attribute_options

        np.random.seed(seed)
        random.seed(seed)


    def get_atom(self, atom_type_name):
        options = self.attribute_options[atom_type_name]
        AtomClass = self.attribute_classes[atom_type_name]
        atom_val = np.random.randint(options)
        return Not(AtomClass(atom_val)) if np.random.rand() < self.not_prob else AtomClass(atom_val)


    def gen_complex_query(self, expressions=3):
        op = np.random.choice(["AND" , "OR_ME", "OR_MI"])
        
        if op == "AND":
            return self.generate_AND(expressions)
        elif op == "OR_ME":
            return self.generate_OR_mutual_exclusive(expressions)
        elif op == "OR_MI":
            return self.generate_OR_conditional_independent(expressions)


    def __get_number_expressions_split(self, no_expressions):
        rem = max(0, no_expressions - 1)
        base, odd = divmod(rem, 2)
        k = np.random.randint(2)
        no_expressions_left = base + odd * k
        no_expressions_right = base + odd * (1 - k)
        return no_expressions_left, no_expressions_right


    def generate_AND(self, no_expressions, use_attribute=None, sampling_groups=None):
        sampling_groups = self.group_names.copy() if sampling_groups is None else sampling_groups
        attr_name = use_attribute if use_attribute is not None else np.random.choice(sampling_groups)
        
        if attr_name in sampling_groups: 
            sampling_groups.remove(attr_name)

        if len(sampling_groups) == 0:
            raise ValueError("Not enough sampling groups to generate AND expression.")

        # no expressions -> generate atom
        if no_expressions <= 0:
            atom = self.get_atom(attr_name)
            return atom
        # 1 expression -> generate AND of two atoms
        elif no_expressions == 1:
            return And(self.get_atom(attr_name), self.get_atom(np.random.choice(sampling_groups)))
        # nested expressions
        else: 
            no_expressions_left, no_expressions_right = self.__get_number_expressions_split(no_expressions)

            random.shuffle(sampling_groups)
            mid = len(sampling_groups) // 2
            groups_left = list(set(sampling_groups[:mid] + [attr_name]))
            groups_right = sampling_groups[mid:]

            # left logic -> has to include use_attribute
            # Generate AND
            if np.random.rand() < 0.5 and len(groups_left) >= 2:
                left_expr = self.generate_AND(no_expressions_left, use_attribute=use_attribute, sampling_groups=groups_left)
            # Generate OR
            else:
                # Generate mutual exclusive OR 
                if np.random.rand() < 0.5 or use_attribute is not None or len(groups_left) < 2:
                    left_expr = self.generate_OR_mutual_exclusive(no_expressions_left, use_attribute=use_attribute, sampling_groups=groups_left)
                # Generate mutual independent OR
                else:
                    left_expr = self.generate_OR_conditional_independent(no_expressions_left, sampling_groups=groups_left)

            # right logic -> does not have to include use_attribute
            # Generate AND
            if np.random.rand() < 0.5 and len(groups_right) >= 2:
                right_expr = self.generate_AND(no_expressions_right, use_attribute=None, sampling_groups=groups_right)
            # Generate OR
            else:
                # Generate mutual exclusive OR 
                if np.random.rand() < 0.5 or use_attribute is not None or len(groups_right) < 2:
                    right_expr = self.generate_OR_mutual_exclusive(no_expressions_right, use_attribute=None, sampling_groups=groups_right)
                # Generate mutual independent OR
                else:
                    right_expr = self.generate_OR_conditional_independent(no_expressions_right, sampling_groups=groups_right)

        gen_expr = And(left_expr, right_expr)
        return Not(gen_expr) if np.random.rand() < self.not_prob else gen_expr


    def generate_OR_mutual_exclusive(self, no_expressions, use_attribute=None, sampling_groups=None):
        sampling_groups = self.group_names.copy() if sampling_groups is None else sampling_groups
        attr_name = use_attribute if use_attribute is not None else np.random.choice(sampling_groups)

        # no expressions -> generate atom
        if no_expressions <= 0:
            atom = self.get_atom(attr_name)
            return atom
        # 1 expression -> generate AND of two atoms
        elif no_expressions == 1:
            return Or_ME(self.get_atom(attr_name), self.get_atom(attr_name))
        # nested expressions
        else: 
            no_expressions_left, no_expressions_right = self.__get_number_expressions_split(no_expressions)

            random.shuffle(sampling_groups)
            mid = len(sampling_groups) // 2
            groups_left = list(set(sampling_groups[:mid] + [attr_name]))
            groups_right = sampling_groups[mid:]

            # left logic -> has to include use_attribute
            # Generate AND
            if np.random.rand() < 0.5 and len(groups_left) >= 2:
                left_expr = self.generate_AND(no_expressions_left, use_attribute=attr_name, sampling_groups=groups_left)
            # Generate OR (mutual exclusive)
            else:
                left_expr = self.generate_OR_mutual_exclusive(no_expressions_left, use_attribute=attr_name, sampling_groups=groups_left)
               
            # right logic -> has to include use_attribute
            # Generate AND
            if np.random.rand() < 0.5 and len(groups_right) >= 2:
                right_expr = self.generate_AND(no_expressions_right, use_attribute=attr_name, sampling_groups=groups_right)
            # Generate OR (mutual exclusive)
            else:
                right_expr = self.generate_OR_mutual_exclusive(no_expressions_right, use_attribute=attr_name, sampling_groups=groups_right)
            
        gen_expr = Or_ME(left_expr, right_expr)
        return Not(gen_expr) if np.random.rand() < self.not_prob else gen_expr
 

    def generate_OR_conditional_independent(self, no_expressions, sampling_groups=None):
        sampling_groups = self.group_names.copy() if sampling_groups is None else sampling_groups
        attr_name = np.random.choice(sampling_groups)
        if attr_name in sampling_groups: 
            sampling_groups.remove(attr_name)

        if len(sampling_groups) == 0:
            raise ValueError("Not enough sampling groups to generate conditional independent OR expression.")

        # no expressions -> generate atom
        if no_expressions <= 0:
            atom = self.get_atom(attr_name)
            return atom
        # 1 expression -> generate AND of two atoms
        elif no_expressions == 1:
            return Or_CI(self.get_atom(attr_name), self.get_atom(np.random.choice(sampling_groups)))
        # nested expressions
        else: 
            no_expressions_left, no_expressions_right = self.__get_number_expressions_split(no_expressions)

            random.shuffle(sampling_groups)
            mid = len(sampling_groups) // 2
            groups_left = list(set(sampling_groups[:mid] + [attr_name]))
            groups_right = sampling_groups[mid:]

            # left logic -> has to include use_attribute
            # Generate AND
            if np.random.rand() < 0.5 and len(groups_left) >= 2:
                left_expr = self.generate_AND(no_expressions_left, use_attribute=None, sampling_groups=groups_left)
            # Generate OR
            else:
                # Generate mutual exclusive OR 
                if np.random.rand() < 0.5 or len(groups_left) < 2:
                    left_expr = self.generate_OR_mutual_exclusive(no_expressions_left, use_attribute=None, sampling_groups=groups_left)
                # Generate conditional independent OR
                else:
                    left_expr = self.generate_OR_conditional_independent(no_expressions_left, sampling_groups=groups_left)

            # right logic -> does not have to include use_attribute
            # Generate AND
            if np.random.rand() < 0.5 and len(groups_right) >= 2:
                right_expr = self.generate_AND(no_expressions_right, use_attribute=None, sampling_groups=groups_right)
            # Generate OR
            else:
                # Generate mutual exclusive OR 
                if np.random.rand() < 0.5 or len(groups_right) < 2:
                    right_expr = self.generate_OR_mutual_exclusive(no_expressions_right, use_attribute=None, sampling_groups=groups_right)
                # Generate conditional independent OR
                else:
                    right_expr = self.generate_OR_conditional_independent(no_expressions_right, sampling_groups=groups_right)

        gen_expr = Or_CI(left_expr, right_expr)
        return Not(gen_expr) if np.random.rand() < self.not_prob else gen_expr
    
if __name__ == "__main__":
    # Example usage and tesing purpose
    generator = ComplexQueryGenerator(
        group_names=["A", "B", "C", "D"],
        attribute_classes={
            "A": lambda x: f"A={x}",
            "B": lambda x: f"B={x}",
            "C": lambda x: f"C={x}",
            "D": lambda x: f"D={x}",
        },
        attribute_options={
            "A": 5,
            "B": 5,
            "C": 5,
            "D": 5,
        }
    )

    for i in range(5):
        expr = generator.gen_complex_query(expressions=4)
        print(f"{i}: {expr}")