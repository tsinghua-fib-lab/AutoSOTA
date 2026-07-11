
"""
Stage I symbolic discovery engine (paper §3.1).

FlexibleVectorSampler draws candidate ansätze from a pre-clustered latent
space.  The Grammar-VAE encodes ~50k expressions into R^32; k-means partitions
that space by MathClass (variable signature).  At search time the sampler
picks subclusters, decodes random latent vectors into symbolic expressions,
and returns them for PDE residual scoring (see evaluator.py).
"""

import numpy as np
import random
import itertools
import torch
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from enum import Enum
import time
import cma, sys
from sigs.utils import (
    FileUtils, ExpressionUtils, ClusteringUtils, ModelUtils
)


class FlexibleVectorSampler:
    """
    Stage I sampler: draws symbolic ansätze from the Grammar-VAE latent space.

    Workflow (Algorithm 1 in paper):
      1. Load pre-computed latent clusters keyed by MathClass.
      2. call sample_from_subclusters() or sample_coherent_sum_expressions()
         to decode latent vectors into candidate expressions.
      3. Score candidates with DerivativeEvaluator (evaluator.py) and rank by
         combined PDE + IC + BC RMSE.
      4. Optionally refine via refine_best_clusters_and_sample() or the full
         iterative_optimization_workflow() (Algorithm 1 outer loop).
    """
    
    def __init__(self, cluster_file: str = 'math_class_clusters.pkl', model=None, device: str = 'cuda'):
        self.clusters_data = FileUtils.load_pickle(cluster_file)
        if self.clusters_data is None:
            raise ValueError(f"Could not load cluster file: {cluster_file}")
        self.sampling_results = {}
        self.current_sample_id = 0
        self.rejected_count = 0
        self.stored_subclusters = {}
        self.optimization_history = []
        self.decoder_model = None
        self.model_device = 'cuda'

    def create_subclusters(self, category: str, n_clusters: int = 5) -> Dict:
        """Create subclusters for a category using k-means clustering."""
        vectors = np.array(self.clusters_data[category]['vectors'])
        expressions = self.clusters_data[category]['expressions']
        
        return ClusteringUtils.create_subclusters(vectors, expressions, n_clusters)

    def check_expression_validity(self, expr: str) -> bool:
        """Enhanced validation to catch problematic expressions."""
        
        # Original validation
        if not ExpressionUtils.validate_expression(expr):
            return False
        
        # Check for malformed patterns
        import re
        invalid_patterns = [
            r'\d+\.e-\d+e-\d+',  # Double scientific notation
            r'\d+\.e\+\d+e-\d+',
            r'log\(-',            # log of negative
            r'log\(0\)',          # log of zero
            r'\*\)',              # multiplication followed by closing paren
            r'\(\*',              # opening paren followed by multiplication
            r'sin\(\s*-\d+\.\d+\s*\)\s*\*',  # sin of negative number times something
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, expr):
                return False
        
        # Check for balanced parentheses
        if expr.count('(') != expr.count(')'):
            return False
        
        # Test parsing with SymEngine
        try:
            import symengine as se
            se_expr = se.sympify(expr.replace('^', '**'))
            
            # Check if expression is too simple (just a constant)
            if se_expr.is_number and abs(float(se_expr)) < 1e-10:
                return False
                
        except Exception:
            return False
        
        return True
        
    def sample_coherent_sum_expressions(
        self,
        expression_template: str,
        role_categories: Dict[str, str], 
        role_subclusters: Dict[str, int],
        n_sum_terms: int = 3,
        sum_operator: str = '+',
        n_samples: int = 1000,
        seed: Optional[int] = None,
        model = None,
        trig_only: bool = False,
    ) -> int:
        """Sample expressions where each individual sample has internal coherence."""
        ClusteringUtils.setup_random_seeds(seed)
        
        # Extract roles from template
        template_roles = self._extract_roles_from_template(expression_template)
        if set(template_roles) != set(role_categories.keys()):
            raise ValueError(f"Template roles {template_roles} don't match role_categories keys {list(role_categories.keys())}")
        
        # Create subclusters for each category
        all_subclusters = {}
        for role, category in role_categories.items():
            if category not in all_subclusters:
                if category not in self.clusters_data:
                    raise ValueError(f"Category '{category}' not found in clusters data")
                
                n_clusters = role_subclusters[category]
                subs = self.create_subclusters(category, n_clusters)
                all_subclusters[category] = subs
                
                #print(f"\nSubclusters for {category} (role: {role}):")
                for sub_idx, data in subs.items():
                    exprs = data['expressions']
                    examples = random.sample(exprs, min(2, len(exprs)))
                    #print(f"  Subcluster {sub_idx}: {examples}")

        # Calculate all possible subcluster combinations
        subcluster_factors = []
        categories_ordered = []
        
        for role in template_roles:
            category = role_categories[role]
            if category not in categories_ordered:
                categories_ordered.append(category)
                subcluster_factors.append(role_subclusters[category])
        
        all_combinations = list(itertools.product(*[range(f) for f in subcluster_factors]))
        
        #print(f"\nSampling Strategy:")
        #print(f"📋 Template: {expression_template}")
        #print(f"🔢 Sum terms per sample: {n_sum_terms}")
        #print(f"🎲 Total subcluster combinations: {len(all_combinations)}")
        
        combined_expressions = []
        combined_vectors = []
        sample_metadata = []
        
        samples_generated = 0
        tried_combinations = set()
        while samples_generated < n_samples:
            combination = random.choice(all_combinations)

            tried_combinations.add(combination)  # Track what we've tried
            
            # Map combination to role subclusters
            role_subcluster_mapping = {}
            combo_idx = 0
            
            for role in template_roles:
                category = role_categories[role]
                if category not in [role_categories[r] for r in role_subcluster_mapping.keys()]:
                    role_subcluster_mapping[role] = combination[combo_idx]
                    combo_idx += 1
                else:
                    for existing_role, existing_category in role_categories.items():
                        if existing_category == category and existing_role in role_subcluster_mapping:
                            role_subcluster_mapping[role] = role_subcluster_mapping[existing_role]
                            break
            
            # Generate sum terms for this sample
            # NEW: Option to use different subclusters for each term
            use_different_subclusters = getattr(self, '_use_different_subclusters_per_term', False)
            
            if use_different_subclusters:
                # Pick n_sum_terms DIFFERENT subclusters for each role
                role_subcluster_choices = {}
                for role in template_roles:
                    category = role_categories[role]
                    n_available = len(all_subclusters[category])
                    # Sample without replacement
                    chosen_subclusters = random.sample(range(n_available), min(n_sum_terms, n_available))
                    role_subcluster_choices[role] = chosen_subclusters
            
            sum_terms = []
            term_vectors = []

            for term_idx in range(n_sum_terms):
                role_expressions = {}
                role_vectors = {}
                
                for role in template_roles:
                    category = role_categories[role]
                    
                    if use_different_subclusters:
                        # Use a different subcluster for each term
                        subcluster_idx = role_subcluster_choices[role][term_idx % len(role_subcluster_choices[role])]
                    else:
                        # Original behavior: same subcluster for all terms
                        subcluster_idx = role_subcluster_mapping[role]
                    
                    subcluster = all_subclusters[category][subcluster_idx]
                    expr_idx = random.randrange(len(subcluster['expressions']))
                    
                    role_expressions[role] = subcluster['expressions'][expr_idx]
                    role_vectors[role] = subcluster['vectors'][expr_idx]
                
                # Substitute roles in template
                term_expr = expression_template
                term_vector_parts = []
                
                for role in template_roles:
                    term_expr = term_expr.replace(role, role_expressions[role])
                    term_vector_parts.append(role_vectors[role])
                
                # # Check if the current term contains both x and y
                # if 'x' not in term_expr or 'y' not in term_expr:
                #     # Skip to the next iteration of the outer loop and try a different combination
                #     continue
                
                # CE- STRICT CHECK: Every single (A*B) term must have x AND y
                if 'x' not in term_expr or 'y' not in term_expr:
                     # This specific combination failed the strict requirement.
                     # Break the inner loop. The outer loop will see len(sum_terms) < n_sum_terms
                     # and reject the whole sample.
                     break
                sum_terms.append(f"({term_expr})")
                term_vectors.append(np.concatenate(term_vector_parts))

            # If we didn't get enough terms, skip this whole expression
            if len(sum_terms) < n_sum_terms:
                self.rejected_count += 1
                continue
            # # ---------------------------------------------------------
            # # NEW CHECK: Ensure x and y exist across the sum components
            # # ---------------------------------------------------------
            # all_terms_combined = "".join(sum_terms)
            # if 'x' not in all_terms_combined or 'y' not in all_terms_combined:
            #     # This combination failed to produce both variables
            #     self.rejected_count += 1
            #     continue

            # Combine terms with sum operator
            if len(sum_terms) == 1:
                final_expr = sum_terms[0]
            else:
                final_expr = sum_terms[0]
                for i in range(1, len(sum_terms)):
                    final_expr += f" {sum_operator} {sum_terms[i]}"

            # mask = "sin(pi*x)*sin(pi*y)"  # masking expression
            mask = "1"  # masking expression
            # # final_expr = f"{mask}+({final_expr})*sin(pi*x)*sin(pi*y)"  # masking expression
            final_expr = f"{mask}*({final_expr})"  # masking expression
            # final_expr = f"{final_expr}"  # masking expression

            # Final validity check
            # if not self.check_expression_validity(final_expr) :
            #     self.rejected_count += 1
            #     continue
            # if not self.check_expression_validity(final_expr)  or  'y' not in final_expr or 'sin' not in final_expr:
            # 1) Must be valid
            if not self.check_expression_validity(final_expr)  :
                self.rejected_count += 1
                continue
            # 2) Must contain at least one trig function (configurable)
            # Count trig occurrences in the string
            # n_sin = final_expr.count("sin(")
            # n_cos = final_expr.count("cos(")
            # n_tan = final_expr.count("tan(")
            # n_trig = n_sin + n_cos + n_tan

            # # Count x and y variable instances
            # n_x = final_expr.count("4*pi")
            # n_y = final_expr.count("2*pi")

            # # If trig_only requested, enforce that expression is trig-only
            # if trig_only:
            #     # reject expressions containing non-trig functions
            #     banned = ['exp(', 'log(']
            #     if any(b in final_expr for b in banned):
            #         self.rejected_count += 1
            #         continue

            #     # require at least one trig occurrence
            #     if n_trig < 4:
            #         self.rejected_count += 1
            #         continue
                
            #     # require minimum number of x and y instances
            #     if n_y < 2:
            #         self.rejected_count += 1
            #         continue
            #     if n_x < 2:
            #         self.rejected_count += 1
            #         continue
            # else:
            #     # 3) Enforce minimum number of trig functions (existing heuristic)
            #     if n_trig < n_sum_terms*2:
            #         self.rejected_count += 1
            #         continue
            # if ('sin' not in final_expr) and ('cos' not in final_expr):
            #     self.rejected_count += 1
            #     continue

            # if('exp' in final_expr):
            #     self.rejected_count += 1
            #     continue

            
            combined_expressions.append(final_expr)
            # combined_vectors.append(np.concatenate(term_vectors))
            combined_vectors.append(np.sum(term_vectors, axis=0))

            sample_metadata.append({
                'subcluster_combination': combination,
                'role_subcluster_mapping': role_subcluster_mapping.copy(),
                'term_vectors': term_vectors
            })
            
            samples_generated += 1
            
            if samples_generated % 100 == 0:
                #print(f"Generated {samples_generated}/{n_samples} samples...")
                pass

        # Store results
        self.sampling_results[self.current_sample_id] = {
            'expressions': combined_expressions,
            'vectors': np.array(combined_vectors),
            'sample_metadata': sample_metadata,
            'expression_indices': list(range(len(combined_expressions))),
            'expression_template': expression_template,
            'role_categories': role_categories,
            'role_subclusters': role_subclusters,
            'n_sum_terms': n_sum_terms,
            'sum_operator': sum_operator,
            'sampling_type': 'coherent_sum',
            'subcluster_indices': [metadata.get('subcluster_combination', ()) for metadata in sample_metadata]  # ADD THIS LINE

            
        }
        
        sample_id = self.current_sample_id
        self.stored_subclusters[sample_id] = all_subclusters
        self.current_sample_id += 1
        
        #print(f"❌ Rejected: {self.rejected_count}")

        if samples_generated % 1000 == 0:
            coverage = len(tried_combinations) / len(all_combinations)
            #print(f"Generated {samples_generated}/{n_samples} samples, "
                # f"tried {len(tried_combinations)}/{len(all_combinations)} combinations "
                # f"({coverage:.2%} coverage)")

        # At the end
        final_coverage = len(tried_combinations) / len(all_combinations)
        #print(f"Final combination coverage: {final_coverage:.2%}")
        
        return sample_id
    
    def sample_jitter_from_cluster(
        self,
        sample_id: int,
        category: str,
        subcluster_idx: int,
        n_samples: int = 100,
        noise_scale: float = 0.05,
        seed: Optional[int] = None
    ) -> int:
        """
        Generate slight Gaussian-noise variants of a single cluster's vectors.
        """
        import numpy as np, random
        ClusteringUtils.setup_random_seeds(seed)

        # Retrieve stored subcluster
        data = self.stored_subclusters[sample_id][category][subcluster_idx]
        base_vecs = np.array(data['vectors'])
        base_exprs = data['expressions']

        new_vecs, new_exprs = [], []
        for _ in range(n_samples):
            i = random.randrange(len(base_vecs))
            v_new = base_vecs[i] + np.random.normal(0, noise_scale, base_vecs.shape[1])
            new_vecs.append(v_new)
            new_exprs.append(f"JITTER({base_exprs[i]},σ={noise_scale})")

        new_id = self.current_sample_id
        self.sampling_results[new_id] = {
            'expressions': new_exprs,
            'vectors': np.stack(new_vecs),
            'subcluster_indices': [(subcluster_idx,)] * n_samples,
            'expression_indices': list(range(n_samples)),
            'sampling_type': 'jittered_cluster',
            'category': category,
            'subcluster': subcluster_idx,
            'noise_scale': noise_scale
        }
        self.stored_subclusters[new_id] = {category: {subcluster_idx: data}}
        self.current_sample_id += 1
        return new_id


    def optimize_sum_terms_cmaes(
    self,
    initial_latents: np.ndarray,
    n_terms: int,
    expression_template: str,
    evaluation_function,
    sigma0: float = 0.5,
    popsize: int = 20,
    max_generations: int = 100,
    seed: Optional[int] = None
) -> Dict:
        """
        CMA-ES over n_terms sub-latents. `initial_latents` shape = (n_terms*latent_dim,).
        Splits that big vector into n_terms pieces, decodes each piece, glues using
        expression_template (a Python format string with {0},{1}…{n_terms-1}), then scores.
        """
        import cma, sys
        #print(f"Optimizing {n_terms} terms with CMA-ES using template: {expression_template}", file=sys.stderr)

        total_dim = initial_latents.shape[0]
        assert total_dim % n_terms == 0, "initial_latents must be divisible by n_terms"
        latent_dim = total_dim // n_terms

        def obj(flat_z, gen=None, idx=None):
            # split into per-term latents
            zs = np.split(flat_z, n_terms)
            # decode each term
            terms = [
                ModelUtils.decode_latent_vectors(
                    z[None,:], self.decoder_model, self.model_device,
                    batch_size=1, seed=seed
                )[0]
                for z in zs
            ]
            # assemble full expression
            expr = expression_template.format(*terms)
            tag = f"[gen{gen} cand{idx}]" if gen is not None else ""
            #print(f"{tag} DEC: {expr}", file=sys.stderr)
            _, _, loss, _ = evaluation_function([expr])
            return float(loss)

        es = cma.CMAEvolutionStrategy(initial_latents, sigma0,
                                    {'popsize': popsize, 'seed': seed})
        for gen in range(max_generations):
            sols   = es.ask()
            losses = [obj(sol, gen, i) for i, sol in enumerate(sols)]
            es.tell(sols, losses)
            if es.stop():
                break

        best_z    = es.result.xbest
        best_loss = obj(best_z, gen, 'best')
        # decode final best
        zs = np.split(best_z, n_terms)
        best_terms = [
            ModelUtils.decode_latent_vectors(
                z[None,:], self.decoder_model, self.model_device,
                batch_size=1, seed=seed
            )[0]
            for z in zs
        ]
        best_expr = expression_template.format(*best_terms)

        return {
            'best_latent':      best_z,
            'best_terms':       best_terms,
            'best_expression':  best_expr,
            'best_loss':        best_loss,
            'generations':      gen+1,
            'popsize':          popsize
        }
    def _extract_roles_from_template(self, template: str) -> List[str]:
        """Extract role names from expression template."""
        import re
        roles = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', template)
        math_funcs = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs'}
        return [role for role in roles if role not in math_funcs]

    def sample_from_subclusters(
        self,
        categories: Dict,  # Can be Dict[str, int] or Dict[MathClass, int]
        n_samples: int = 100,
        operator: str = '+',
        operators: Optional[List[str]] = None,
        category_instances: Optional[Dict] = None,  # Can be Dict[str, int] or Dict[MathClass, int]
        seed: Optional[int] = None,
        model = None,
        # New optional arguments:
        preferred_subclusters: Optional[Dict] = None,  # e.g. {'SPATIOTEMPORAL_3D_instance_0':[0,1], 'SPATIOTEMPORAL_3D_instance_1':[2,3]}
        distinct_instances: bool = False,  # enforce different subcluster indices for instances of same base category
    ) -> int:
        """Sample from subclusters and combine expressions with equal representation."""
        ClusteringUtils.setup_random_seeds(seed)
        
        # Normalize categories to strings
        def normalize_category_key(key):
            return key.name if isinstance(key, Enum) else key

        categories = {normalize_category_key(k): v for k, v in categories.items()}

        if category_instances is not None:
            category_instances = {normalize_category_key(k): v for k, v in category_instances.items()}

        # Validate categories
        for category in categories:
            if category not in self.clusters_data:
                raise ValueError(f"Category '{category}' not found in clusters data. Available categories: {list(self.clusters_data.keys())}")

        # Handle category instances - expand categories to include multiple instances
        if category_instances is not None:
            expanded_categories = {}
            expanded_category_names = []
            category_key_mapping = {}  # Map instance names back to original category keys
            
            for category, n_instances in category_instances.items():
                if category not in categories:
                    raise ValueError(f"Category '{category}' in category_instances must also be in categories dict")
                
                for i in range(n_instances):
                    instance_name = f"{category}_instance_{i}"
                    expanded_categories[instance_name] = categories[category]
                    expanded_category_names.append(instance_name)
                    category_key_mapping[instance_name] = category  # Store original category key
            
            # Add any categories not in category_instances (single instance)
            for category, n_clusters in categories.items():
                if category not in category_instances:
                    expanded_categories[category] = n_clusters
                    expanded_category_names.append(category)
                    category_key_mapping[category] = category  # Store original category key
            
            working_categories = expanded_categories
            category_names = expanded_category_names
            #print(f"Expanded categories: {working_categories}")
        else:
            working_categories = categories
            category_names = list(categories.keys())
            category_key_mapping = {cat: cat for cat in category_names}  # Identity mapping

        # Handle operators - if operators list provided, use it; otherwise use single operator
        if operators is not None:
            if len(operators) != len(category_names) - 1:
                raise ValueError(f"operators list must have {len(operators)-1} elements for {len(category_names)} categories")
            ops_to_use = operators
        else:
            ops_to_use = [operator] * (len(category_names) - 1)

        # Create subclusters for each base category (not each instance)
        all_subclusters = {}
        base_categories = set()
        
        for cat_name in category_names:
            # Get the original category key
            base_cat = category_key_mapping[cat_name]
            base_categories.add(base_cat)
            
            if base_cat not in all_subclusters:
                n_clusters = categories[base_cat]
                subs = self.create_subclusters(base_cat, n_clusters)
                all_subclusters[base_cat] = subs
                
                #print(f"\nExample expressions from each subcluster of {base_cat}:")
                for sub_idx, data in subs.items():
                    exprs = data['expressions']
                    examples = random.sample(exprs, min(3, len(exprs)))
                    #print(f"  Subcluster {sub_idx}:")
                    for ex in examples:
                        #print(f"     {ex}")
                        pass

        combined_expressions = []
        combined_vectors = []
        subcluster_indices = []
        expression_indices = []
        
        # Calculate combinations considering instances
        total_combinations = np.prod([working_categories[cat] for cat in category_names])
        samples_per_combination = n_samples // total_combinations
        remaining_samples = n_samples % total_combinations

        # Generate all possible subcluster combinations
        # Optionally restrict available subcluster indices per (instance) category via preferred_subclusters
        def _allowed_indices_for(cat_name):
            # preferred_subclusters can specify either instance names or base category names
            if preferred_subclusters is None:
                return list(range(working_categories[cat_name]))
            # try instance-specific first
            if cat_name in preferred_subclusters:
                return list(preferred_subclusters[cat_name])
            # fall back to base category name
            base_cat = category_key_mapping[cat_name]
            if base_cat in preferred_subclusters:
                return list(preferred_subclusters[base_cat])
            return list(range(working_categories[cat_name]))

        index_ranges = [_allowed_indices_for(cat) for cat in category_names]
        subcluster_combinations = list(itertools.product(*index_ranges))

        # If distinct_instances is requested, filter combinations so that per-base-category
        # the indices chosen for its instances are all distinct (no repeated subcluster index).
        if distinct_instances:
            filtered_combinations = []
            # build mapping from base_cat -> list of positions (indices into category_names)
            base_to_positions = {}
            for pos, cat_name in enumerate(category_names):
                base = category_key_mapping[cat_name]
                base_to_positions.setdefault(base, []).append(pos)

            for combo in subcluster_combinations:
                ok = True
                for base, positions in base_to_positions.items():
                    if len(positions) <= 1:
                        continue
                    chosen = [combo[p] for p in positions]
                    # If any duplicates among chosen indices, skip this combo
                    if len(set(chosen)) != len(chosen):
                        ok = False
                        break
                if ok:
                    filtered_combinations.append(combo)

            # Only replace if some combinations remain after filtering
            if filtered_combinations:
                subcluster_combinations = filtered_combinations
        random.shuffle(subcluster_combinations)

        samples_generated = 0
        
        # First ensure equal representation
        for subclusters in subcluster_combinations:
            for _ in range(samples_per_combination):
                selected_exprs = {}
                selected_vecs = {}
                selected_orig_indices = {}
                
                for cat_idx, cat_name in enumerate(category_names):
                    # Get the original category key
                    base_cat = category_key_mapping[cat_name]
                    
                    subcluster = all_subclusters[base_cat][subclusters[cat_idx]]
                    idx = random.randrange(len(subcluster['expressions']))
                    selected_exprs[cat_name] = subcluster['expressions'][idx]
                    selected_vecs[cat_name] = subcluster['vectors'][idx]
                    selected_orig_indices[cat_name] = subcluster['expression_indices'][idx]

                # Combine expressions with potentially different operators
                expr_parts = [f"({selected_exprs[cat]})" for cat in category_names]
                
                # Build expression with specified operators
                combined_expr = expr_parts[0]
                for i, op in enumerate(ops_to_use):
                    combined_expr += f" {op} {expr_parts[i+1]}"
                
                if not self.check_expression_validity(combined_expr):
                    self.rejected_count += 1
                    continue
                    
                combined_expressions.append(combined_expr)
                combined_vectors.append(np.concatenate([selected_vecs[cat] for cat in category_names]))
                subcluster_indices.append(subclusters)
                expression_indices.append(tuple(selected_orig_indices.values()))
                
                samples_generated += 1

        # Handle remaining samples randomly
        while samples_generated < n_samples:
            subclusters = random.choice(subcluster_combinations)
            selected_exprs = {}
            selected_vecs = {}
            selected_orig_indices = {}
            
            for cat_idx, cat_name in enumerate(category_names):
                # Get the original category key
                base_cat = category_key_mapping[cat_name]
                
                subcluster = all_subclusters[base_cat][subclusters[cat_idx]]
                idx = random.randrange(len(subcluster['expressions']))
                selected_exprs[cat_name] = subcluster['expressions'][idx]
                selected_vecs[cat_name] = subcluster['vectors'][idx]
                selected_orig_indices[cat_name] = subcluster['expression_indices'][idx]

            # Combine expressions with potentially different operators
            expr_parts = [f"({selected_exprs[cat]})" for cat in category_names]
            
            # Build expression with specified operators
            combined_expr = expr_parts[0]
            for i, op in enumerate(ops_to_use):
                combined_expr += f" {op} {expr_parts[i+1]}"
            
            if not self.check_expression_validity(combined_expr):
            #     self.rejected_count += 1
            #     continue
            # #print(f"Testing expression: {combined_expr}")
            # if 'y' not in combined_expr   :
                # #print(f"Rejected expression: {combined_expr}")
                self.rejected_count += 1
                continue
                
            combined_expressions.append(combined_expr)
            combined_vectors.append(np.concatenate([selected_vecs[cat] for cat in category_names]))
            subcluster_indices.append(subclusters)
            expression_indices.append(tuple(selected_orig_indices.values()))
            
            samples_generated += 1

        # Store results with indices for later evaluation
        self.sampling_results[self.current_sample_id] = {
            'expressions': combined_expressions,
            'vectors': np.array(combined_vectors),
            'subcluster_indices': subcluster_indices,
            'expression_indices': expression_indices,
            'categories': categories,
            'category_instances': category_instances,
            'operator': operator,
            'operators_used': ops_to_use,
            'sampling_type': 'subcluster_combination',
        }
        sample_id = self.current_sample_id
        # Store the subclusters that were actually used
        self.stored_subclusters[sample_id] = all_subclusters
        self.current_sample_id += 1
        
        #print(f"Generated {len(combined_expressions)} valid expressions")
        #print(f"Rejected {self.rejected_count} expressions containing S, T, or D")
        #print(f"Samples per subcluster combination: {samples_per_combination}")
        #print(f"Remaining samples distributed randomly: {remaining_samples}")
        #print(f"Operators used: {ops_to_use}")
        
        return sample_id

    def refine_best_clusters_and_sample(
            self,
            sample_id: int,
            best_expression_idx: int,
            m_clusters_per_factor: int = 10,  # micro-clusters per factor
            n_samples: int = 1000,
            operator: str = '+',
            seed: Optional[int] = None
        ) -> int:
        """
        Generalized refinement that works for any number of factors (2, 3, 4+).
        Automatically uses hybrid sampling when requesting more samples than unique combinations allow.
        """
        ClusteringUtils.setup_random_seeds(seed)
        
        if sample_id not in self.stored_subclusters:
            
            # Find the original sample with subclusters by tracing back
            original_sample_id = None
            current_id = sample_id
            
            for _ in range(10):  # Max 10 levels deep
                if current_id in self.stored_subclusters:
                    original_sample_id = current_id
                    break
                    
                # Look for refined_from in sampling results
                if current_id in self.sampling_results and 'refined_from' in self.sampling_results[current_id]:
                    current_id = self.sampling_results[current_id]['refined_from']
                else:
                    break
            
            if original_sample_id is not None:
                self.stored_subclusters[sample_id] = self.stored_subclusters[original_sample_id]
            else:
                raise ValueError(f"No stored subclusters for sample_id {sample_id} and cannot find original sample")
        # Get the best subcluster combination
        original_subclusters = self.stored_subclusters[sample_id]
        best_combination = self.get_best_subclusters(sample_id, best_expression_idx)
        
        # Handle both tuple and non-tuple cases
        if not isinstance(best_combination, (tuple, list)):
            best_combination = (best_combination,)
        
        # ROBUST FIX: Extract categories in the correct order and validate indices
        # ROBUST FIX: Extract categories in the correct order and validate indices
        sampling_result = self.sampling_results[sample_id]
        cats = []

        # Try multiple strategies to get categories
        if 'categories' in sampling_result and sampling_result['categories']:
            cats = list(sampling_result['categories'].keys())
        elif 'role_categories' in sampling_result:
            role_categories = sampling_result['role_categories']
            template = sampling_result['expression_template']
            template_roles = self._extract_roles_from_template(template)
            for role in template_roles:
                category = role_categories[role]
                if category not in cats:
                    cats.append(category)
        elif 'refined_from' in sampling_result:
            # Get from original sample
            original_id = sampling_result['refined_from']
            if original_id in self.sampling_results:
                original_result = self.sampling_results[original_id]
                if 'categories' in original_result and original_result['categories']:
                    cats = list(original_result['categories'].keys())
                elif 'role_categories' in original_result:
                    role_categories = original_result['role_categories']
                    template = original_result['expression_template']
                    template_roles = self._extract_roles_from_template(template)
                    for role in template_roles:
                        category = role_categories[role]
                        if category not in cats:
                            cats.append(category)

        # FALLBACK: Use stored_subclusters if still empty
        if not cats and original_subclusters:
            cats = list(original_subclusters.keys())
            #print(f"🔧 FALLBACK: Using stored_subclusters keys as categories: {cats}")

        # Final validation
        if not cats:
            raise ValueError(f"Cannot determine categories for sample_id {sample_id}. Keys available: {list(sampling_result.keys())}")
        #print(f"Final validated combination: {best_combination}")

        # ADD THIS DEBUG SECTION:
        #print(f"DEBUG INFO:")
        #print(f"  sample_id: {sample_id}")
        #print(f"  sampling_result keys: {list(sampling_result.keys())}")
        #print(f"  categories found: {cats}")
        #print(f"  best_combination length: {len(best_combination)}")
        #print(f"  stored_subclusters keys: {list(original_subclusters.keys()) if original_subclusters else 'None'}")
        # Add validation for empty categories
        if not cats:
            raise ValueError(f"No categories found for sample_id {sample_id}. Available keys: {list(sampling_result.keys())}")
        #print(f"\n=== GENERALIZED REFINEMENT ===")
        #print(f"Best combination: {best_combination}")
        #print(f"Categories: {cats}")
        #print(f"Number of factors: {len(best_combination)}")
        
        # Extract data for each factor
        factor_data = []
        for i, cat in enumerate(cats):
            if i < len(best_combination):
                subcluster_idx = best_combination[i]
                subs = original_subclusters[cat][subcluster_idx]
                
                vecs = np.array(subs['vectors'])
                exprs = subs['expressions']
                
                factor_data.append({
                    'category': cat,
                    'vectors': vecs,
                    'expressions': exprs,
                    'subcluster_idx': subcluster_idx
                })
                
                #print(f"\nFactor {i} ({cat}[{subcluster_idx}]):")
                #print(f"  - Contains {len(exprs)} expressions")
                #print(f"  - Sample expressions: {exprs[:3]}")
        
        # Micro-cluster each factor
        micro_clusters = []
        for i, factor in enumerate(factor_data):
            vecs = factor['vectors']
            exprs = factor['expressions']
            
            if len(vecs) < m_clusters_per_factor:
                #print(f"WARNING: Factor {i} has only {len(vecs)} expressions, using {len(vecs)} clusters")
                n_clusters = len(vecs)
            else:
                n_clusters = m_clusters_per_factor
            
            km = KMeans(n_clusters=n_clusters, random_state=seed).fit(vecs)
            labels = km.labels_
            
            members = {k: np.where(labels == k)[0] for k in range(n_clusters)}
            
            micro_clusters.append({
                'members': members,
                'expressions': exprs,
                'vectors': vecs,
                'n_clusters': n_clusters
            })
            
            #print(f"Factor {i}: Created {n_clusters} micro-clusters")
        
        # Generate all possible micro-cluster combinations
        cluster_ranges = [range(mc['n_clusters']) for mc in micro_clusters]
        all_combos = list(itertools.product(*cluster_ranges))
        random.shuffle(all_combos)
        
        # Calculate maximum possible unique combinations
        max_unique_combinations = 1
        for mc in micro_clusters:
            cluster_sizes = [len(members) for members in mc['members'].values()]
            max_unique_combinations *= np.prod(cluster_sizes)
        
        total_combos = len(all_combos)
        per_combo = n_samples // total_combos
        extra = n_samples % total_combos
        
        #print(f"Sampling strategy analysis:")
        #print(f"  - Micro-cluster combinations: {total_combos}")
        #print(f"  - Max unique vector combinations: {max_unique_combinations:,}")
        #print(f"  - Requested samples: {n_samples:,}")
        #print(f"  - Samples per micro-combo: {per_combo}")
        
        # Check if we need model-based resampling
        use_model_resampling = (n_samples > max_unique_combinations * 0.8 and 
                               max_unique_combinations < n_samples)
        
        if use_model_resampling:
            pass
        
        # Sample expressions
        new_exprs = []
        new_vecs = []
        new_subcluster_indices = []
        count = 0
        used_combinations = set()  # Track used vector combinations
        
        def make_one_sample(combo, force_unique=False):
            """Generate one sample from a micro-cluster combination"""
            # check what parameters we have
            current_result = self.sampling_results[sample_id]
            
            # #print(f"  sample_id: {sample_id}")
            # #print(f"  has expression_template: {'expression_template' in current_result}")
            # #print(f"  sampling_type: {current_result.get('sampling_type', 'UNKNOWN')}")
            # #print(f"  n_sum_terms: {current_result.get('n_sum_terms', 'MISSING')}")
            
            max_attempts = 50 if force_unique else 1
            
            for attempt in range(max_attempts):
                selected_parts = []
                selected_vecs = []
                vector_indices = []
                
                for factor_idx, cluster_idx in enumerate(combo):
                    mc = micro_clusters[factor_idx]
                    members = mc['members'][cluster_idx]
                    
                    # Randomly select from this micro-cluster
                    member_idx = random.choice(members)
                    selected_parts.append(mc['expressions'][member_idx])
                    selected_vecs.append(mc['vectors'][member_idx])
                    vector_indices.append(member_idx)
                
                # Check if this combination has been used (if we care about uniqueness)
                combination_key = tuple(vector_indices) if force_unique else None
                if force_unique and combination_key in used_combinations:
                    continue  # Try again
                
                # Combine based on the original operator/template
                # Combine based on the original operator/template
                if 'expression_template' in self.sampling_results[sample_id]:
                    # coherent sum case: preserve sum structure
                    template = self.sampling_results[sample_id]['expression_template']
                    role_categories = self.sampling_results[sample_id]['role_categories']
                    n_sum_terms = self.sampling_results[sample_id].get('n_sum_terms', 2)
                    sum_operator = self.sampling_results[sample_id].get('sum_operator', '+')
                    template_roles = self._extract_roles_from_template(template)
                    
                    # Generate multiple sum terms (coherent structure)
                    sum_terms = []
                    for term_idx in range(n_sum_terms):
                        # Each term uses the same selected_parts but potentially different instances
                        if term_idx == 0:
                            # First term uses the selected parts as-is
                            term_parts = selected_parts.copy()
                        else:
                            # Additional terms: re-sample from same micro-clusters
                            term_parts = []
                            for factor_idx in range(len(selected_parts)):
                                # Re-sample from the same micro-cluster
                                mc = micro_clusters[factor_idx]
                                cluster_idx = combo[factor_idx]
                                members = mc['members'][cluster_idx]
                                member_idx = random.choice(members)
                                term_parts.append(mc['expressions'][member_idx])
                        
                        # Create this term by substituting into template
                        term_expr = template
                        for role_idx, role in enumerate(template_roles):
                            if role_idx < len(term_parts):
                                term_expr = term_expr.replace(role, term_parts[role_idx])
                        
                        sum_terms.append(f"({term_expr})")
                    
                    # Combine terms with sum operator
                    combined_expr = sum_terms[0]
                    for i in range(1, len(sum_terms)):
                        combined_expr += f" {sum_operator} {sum_terms[i]}"
                        
                else:
                    # Handle simple case (unchanged)
                    ops = self.sampling_results[sample_id].get('operators_used', [operator] * (len(selected_parts) - 1))
                    combined_expr = f"({selected_parts[0]})"
                    for i, op in enumerate(ops):
                        combined_expr += f" {op} ({selected_parts[i + 1]})"
                
                combined_vec = np.concatenate(selected_vecs)
                
                if force_unique and combination_key is not None:
                    used_combinations.add(combination_key)
                # combined_expr = f" ({combined_expr}) * sin(pi*x)*sin(pi*y)"
                combined_expr = f" ({combined_expr}) "
                return combined_expr, combined_vec, True
            
            return None, None, False  # Failed to find unique combination
        
        # Phase 1: Fill unique combinations first (if we have enough)
        unique_sampling_target = min(n_samples, max_unique_combinations)
        
        #print(f"Phase 1: Generating {unique_sampling_target:,} unique combinations...")
        
        # Fill samples per combination (trying to be unique)
        for combo in all_combos:
            samples_for_this_combo = min(per_combo, unique_sampling_target - count)
            
            for _ in range(samples_for_this_combo):
                expr, vec, success = make_one_sample(combo, force_unique=True)
                if success and self.check_expression_validity(expr):
                    new_exprs.append(expr)
                    new_vecs.append(vec)
                    new_subcluster_indices.append(combo)
                    count += 1
                
                if count >= unique_sampling_target:
                    break
            
            if count >= unique_sampling_target:
                break
        
        #print(f"Phase 1 complete: {count:,} unique samples generated")
        
        # Phase 2: Model-based resampling for remaining samples
        if count < n_samples and use_model_resampling:
            remaining_samples = n_samples - count
            #print(f"Phase 2: Generating {remaining_samples:,} additional samples via model interpolation...")
            
            # Use the vectors we've already generated for interpolation
            if len(new_vecs) >= 2:
                interpolated_samples = self._generate_interpolated_samples(
                    existing_vecs=np.array(new_vecs),
                    n_samples=remaining_samples,
                    model=None,  # We'll do interpolation without decoding for now
                    interpolation_strength=0.3
                )
                
                # Add the interpolated samples
                for interp_vec in interpolated_samples:
                    # For now, just mark these as interpolated combinations
                    new_vecs.append(interp_vec)
                    new_exprs.append(f"INTERPOLATED_SAMPLE_{len(new_exprs)}")  # Placeholder
                    new_subcluster_indices.append(('interpolated',))
                    count += 1
                
                #print(f"Phase 2 complete: {remaining_samples:,} interpolated samples added")
            else:
                #print("Not enough vectors for interpolation, filling with repeats...")
                pass
                
        # Phase 3: Fill any remaining with repeats (fallback)
        if count < n_samples:
            remaining = n_samples - count
            #print(f"Phase 3: Filling {remaining:,} remaining samples with repeats...")
            pass
            
            combo_idx = 0
            while count < n_samples:
                combo = all_combos[combo_idx % len(all_combos)]
                expr, vec, _ = make_one_sample(combo, force_unique=False)
                if self.check_expression_validity(expr):
                    new_exprs.append(expr)
                    new_vecs.append(vec)
                    new_subcluster_indices.append(combo)
                    count += 1
                combo_idx += 1
        
        # Store results
        new_id = self.current_sample_id

        # find coherent sum parameters from original sampling record
        original_sample_id = sample_id
        while original_sample_id in self.sampling_results:
            if 'expression_template' in self.sampling_results[original_sample_id]:
                break
            elif 'refined_from' in self.sampling_results[original_sample_id]:
                original_sample_id = self.sampling_results[original_sample_id]['refined_from']
            else:
                break

        # Get original coherent sum parameters
        original_result = self.sampling_results[original_sample_id]

        self.sampling_results[new_id] = {
            'expressions': new_exprs,
            'vectors': np.stack(new_vecs),
            'subcluster_indices': new_subcluster_indices,
            'expression_indices': list(range(len(new_exprs))),
            
            # inherit coherent sum structure
            'expression_template': original_result.get('expression_template'),
            'role_categories': original_result.get('role_categories', {}),
            'role_subclusters': original_result.get('role_subclusters', {}),
            'n_sum_terms': original_result.get('n_sum_terms', 2),
            'sum_operator': original_result.get('sum_operator', '+'),
            'sampling_type': original_result.get('sampling_type', 'refined_clusters'),
            
            'categories': self.sampling_results[sample_id].get('categories', {}),
            'operator': operator,
            'refined_from': sample_id,
            'refined_around': best_combination,
        }

        #print(f"  - expression_template: {self.sampling_results[new_id].get('expression_template', 'MISSING')}")
        #print(f"  - n_sum_terms: {self.sampling_results[new_id].get('n_sum_terms', 'MISSING')}")
        #print(f"  - sampling_type: {self.sampling_results[new_id].get('sampling_type', 'MISSING')}")

        
        # self.current_sample_id += 1
        # #print(f"→ Created refined sample batch {new_id} with {len(new_exprs)} expressions")
        # #print(f"Sample refined expressions: {new_exprs[:3]}")
        # self.current_sample_id += 1
        # # Store subclusters for future refinement
        self.current_sample_id += 1
        self.stored_subclusters[new_id] = original_subclusters
      

        #print(f"→ Created refined sample batch {new_id} with {len(new_exprs)} expressions")
        #print(f"Sample refined expressions: {new_exprs[:3]}")

        return new_id

    def _generate_interpolated_samples(
        self,
        existing_vecs: np.ndarray,
        n_samples: int,
        model=None,
        interpolation_strength: float = 0.3
    ) -> List[np.ndarray]:
        """
        Generate new samples by interpolating between existing vectors.
        This is used when we need more samples than unique combinations allow.
        """
        if len(existing_vecs) < 2:
            # Not enough vectors for interpolation, just add noise
            return [existing_vecs[0] + np.random.normal(0, 0.01, existing_vecs[0].shape) 
                    for _ in range(n_samples)]
        
        interpolated_samples = []
        
        for _ in range(n_samples):
            # Select 2-3 random vectors to interpolate between
            n_to_combine = min(np.random.randint(2, 4), len(existing_vecs))
            indices = np.random.choice(len(existing_vecs), n_to_combine, replace=False)
            selected_vecs = existing_vecs[indices]
            
            # Generate convex combination weights
            weights = np.random.dirichlet(np.ones(n_to_combine))
            
            # Apply interpolation strength (blend with one of the original vectors)
            base_idx = np.random.choice(n_to_combine)
            base_weight = weights[base_idx]
            other_weights = weights.copy()
            other_weights[base_idx] = 0
            
            # Adjust weights based on interpolation strength
            final_weights = np.zeros_like(weights)
            final_weights[base_idx] = base_weight * (1 - interpolation_strength) + interpolation_strength * base_weight
            final_weights += other_weights * interpolation_strength
            
            # Renormalize
            final_weights = final_weights / final_weights.sum()
            
            # Create interpolated vector
            interpolated = np.sum([w * vec for w, vec in zip(final_weights, selected_vecs)], axis=0)
            interpolated_samples.append(interpolated)
        
        return interpolated_samples

    def get_sampling_results(self, sample_id: int) -> Tuple[List[str], np.ndarray, List[Tuple[int, int]], List[int]]:
        """Retrieve sampling results and indices by sample ID."""
        if sample_id not in self.sampling_results:
            raise ValueError(f"Sample ID {sample_id} not found")
        
        results = self.sampling_results[sample_id]
        
        # Handle both old and new data formats
        if 'subcluster_indices' in results:
            # Original format from sample_from_subclusters()
            subcluster_indices = results['subcluster_indices']
        elif 'sample_metadata' in results:
            # New format from sample_coherent_sum_expressions()
            # Extract subcluster combinations from metadata
            subcluster_indices = []
            for metadata in results['sample_metadata']:
                if 'subcluster_combination' in metadata:
                    subcluster_indices.append(metadata['subcluster_combination'])
                else:
                    subcluster_indices.append(metadata.get('role_subcluster_mapping', {}))
        else:
            # Fallback - create dummy indices
            subcluster_indices = list(range(len(results['expressions'])))
        
        return (
            results['expressions'],
            results['vectors'],
            subcluster_indices,
            results['expression_indices']
        )

    def get_best_subclusters(self, sample_id: int, best_expression_idx: int):
        """Get subcluster indices for the best performing expression with validation."""
        if sample_id not in self.sampling_results:
            raise ValueError(f"Sample ID {sample_id} not found")
        
        results = self.sampling_results[sample_id]
        
        # Validate expression index
        if best_expression_idx >= len(results['subcluster_indices']):
            raise IndexError(f"Expression index {best_expression_idx} out of range (max: {len(results['subcluster_indices'])-1})")
        
        raw_combination = results['subcluster_indices'][best_expression_idx]
        
        # Handle different data formats
        if isinstance(raw_combination, dict):
            # Handle role_subcluster_mapping format from coherent sum
            if 'role_categories' in results:
                role_categories = results['role_categories']
                template = results['expression_template'] 
                template_roles = self._extract_roles_from_template(template)
                
                # Convert role mapping to ordered tuple
                ordered_indices = []
                processed_categories = set()
                
                for role in template_roles:
                    category = role_categories[role]
                    if category not in processed_categories:
                        if role in raw_combination:
                            ordered_indices.append(raw_combination[role])
                        else:
                            # Fallback to 0 if role not found
                            ordered_indices.append(0)
                        processed_categories.add(category)
                
                return tuple(ordered_indices)
            else:
                # Convert dict to tuple (fallback)
                return tuple(raw_combination.values())
        
        elif isinstance(raw_combination, (tuple, list)):
            # Already in correct format
            return tuple(raw_combination)
        
        else:
            # Single value - wrap in tuple
            return (raw_combination,)
    def iterative_optimization_workflow(
        self,
        initial_sampling_config: Dict,
        evaluation_function,
        max_iterations: int = 5,
        refinement_factor: float = 0.5,
        convergence_threshold: float = 1e-6,
        verbose: bool = True
    ) -> Dict:
        """
        Complete iterative optimization workflow for mathematical expression discovery.
        """
        start_time = time.time()
        self.optimization_history = []

        # Ensure base_samples is always available
        base_samples = initial_sampling_config['params'].get('n_samples', 10000)

        current_best_loss = float('inf')
        current_best_expr = None
        current_best_sample_id = None

        if verbose:
            #print("=" * 60)
            #print("=" * 60)
            pass

        # Phase 1: Initial broad sampling
        if verbose:
            
            pass

        method = initial_sampling_config['method']
        params = initial_sampling_config['params']
        if method == 'coherent_sum':
            current_sample_id = self.sample_coherent_sum_expressions(**params)
        elif method == 'subclusters':
            current_sample_id = self.sample_from_subclusters(**params)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Phase 2: Iterative refinement
        for iteration in range(max_iterations):
            if verbose:
                #print("-" * 40)
                pass

            # Evaluate current batch
            exprs, vecs, sub_idxs, expr_idxs = self.get_sampling_results(current_sample_id)
            # exprs = [e for e in exprs if "exp" in e and "sin" in e and 'sqrt' not in e]
            if verbose:
                #print(f"Evaluating {len(exprs):,} expressions...")
                pass

            eval_start = time.time()
            #print(f"Evaluation function: {evaluation_function.__name__}")
            best_idx, best_expr, best_loss, all_losses = evaluation_function(exprs)
            eval_time = time.time() - eval_start

            # Track progress
            iteration_data = {
                'iteration': iteration + 1,
                'sample_id': current_sample_id,
                'best_idx': best_idx,
                'best_expr': best_expr,
                'best_loss': best_loss,
                'n_expressions': len(exprs),
                'evaluation_time': eval_time,
                'sampling_type': self.sampling_results[current_sample_id].get('sampling_type', 'unknown')
            }
            self.optimization_history.append(iteration_data)

            if verbose:
                pass

            # Check for global improvement
            improvement = current_best_loss - best_loss
            if best_loss < current_best_loss:
                current_best_loss = best_loss
                current_best_expr = best_expr
                current_best_sample_id = current_sample_id

                if verbose:
                    pass

                # Refine around best clusters if not last iteration
                if iteration < max_iterations - 1:
                    refined_samples = max(1000, int(base_samples * (refinement_factor ** iteration)))
                    if verbose:
                        pass
                    current_sample_id = self.refine_best_clusters_and_sample(
                        sample_id=current_sample_id,
                        best_expression_idx=best_idx,
                        m_clusters_per_factor=10 + iteration * 3,
                        n_samples=refined_samples,
                        operator=params.get('sum_operator', '+')
                    )
            else:
                if verbose:
                    #print(f"❌ No improvement. Best remains: {current_best_loss:.8f}")
                    pass

                # Check convergence
                if improvement < convergence_threshold:
                    if verbose:
                        #print(f"🏁 Converged! Improvement {improvement:.8f} < threshold {convergence_threshold}")
                        pass
                    break

                # Exploration if not last iteration
                if iteration < max_iterations - 1:
                    # Option A: further refine clusters
                    exploration_samples = max(1000, int(base_samples * (refinement_factor ** (iteration + 1))))
                    if verbose:
                        #print(f"🌟 Exploration refine with {exploration_samples:,} samples...")
                        pass
                    current_sample_id = self.refine_best_clusters_and_sample(
                        sample_id=current_sample_id,
                        best_expression_idx=best_idx,
                        m_clusters_per_factor=8,
                        n_samples=exploration_samples,
                        operator=params.get('sum_operator', '+')
                    )

                    # Determine best subcluster for jitter/CMA-ES
                    best_combo = self.get_best_subclusters(current_sample_id, best_idx)
                    best_sub_idx = best_combo[0] if isinstance(best_combo, (tuple, list)) else best_combo
                    best_latent = vecs[best_idx]

                    # Option B1: Local jitter
                    # Use a safe default category or extract from coherent params
                    jitter_cat = (list(params.get('role_categories', {}).values())[0]
                                  if 'role_categories' in params
                                  else list(self.clusters_data.keys())[0])
                    current_sample_id = self.sample_jitter_from_cluster(
                        sample_id=current_sample_id,
                        category=jitter_cat,
                        subcluster_idx=best_sub_idx,
                        n_samples=500,
                        noise_scale=0.1,
                        seed=42
                    )

                else:
                        if verbose:
                            #print(f"❌ No improvement. Best remains: {current_best_loss:.8f}")
                            pass

                        # 1) Grab how many terms you used initially
                        cfg     = initial_sampling_config['params']
                        n_terms = cfg['n_sum_terms']

                        # 2) Build a Python-format template for {}-injection:
                        sum_template = "(" + " + ".join(f"{{{i}}}" for i in range(n_terms)) + ")"

                        # 3) Extract the current best latent vector:
                        #    your sampler stores all vectors in sampling_results[…]['vectors']
                        best_vec = self.sampling_results[current_best_sample_id]['vectors'][best_idx]

                        # 4) Run the term-wise CMA-ES
                        cma_out = self.optimize_sum_terms_cmaes(
                            initial_latents     = best_vec,
                            n_terms             = n_terms,
                            expression_template = sum_template,
                            evaluation_function = evaluation_function,
                            sigma0              = 0.3,
                            popsize             = 30,
                            max_generations     = 20,
                            seed                = cfg.get('seed', None)
                        )

                        # 5) If CMA-ES gave you a better loss, update your global best
                        if cma_out['best_loss'] < current_best_loss:
                            current_best_loss = cma_out['best_loss']
                            current_best_expr = cma_out['best_expression']
                            # note: this is now a fresh decode, not a cluster batch
                            #print(f"CMA-ES best: {cma_out['best_expression']}  loss: {cma_out['best_loss']:.6f}")
                   
                        # then break or continue as you prefer
                        break

                    

        total_time = time.time() - start_time

        # Compile final results
        results = {
            'best_expression': current_best_expr,
            'best_loss': current_best_loss,
            'best_sample_id': current_best_sample_id,
            'optimization_history': self.optimization_history,
            'total_time': total_time,
            'total_iterations': len(self.optimization_history),
            'converged': improvement < convergence_threshold if 'improvement' in locals() else False
        }

        if verbose:
            #print("\n" + "=" * 60)
            #print("=" * 60)
            
            for i, hist in enumerate(self.optimization_history):
                status = "*" if hist['best_loss'] == current_best_loss else "  "
                #print(f"{status} {i+1}: Loss {hist['best_loss']:.6f} | {hist['sampling_type']} | {hist['n_expressions']:,} exprs | {hist['evaluation_time']:.1f}s")

        return results



