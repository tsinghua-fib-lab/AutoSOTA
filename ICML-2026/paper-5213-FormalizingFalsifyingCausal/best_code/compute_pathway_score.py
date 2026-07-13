#!/usr/bin/env python3
"""
Compute the pathway explanation score from Section 5 of:
"Formalizing and Falsifying Causal Pathways of Rare Events"
Haghighat & Janzing, ICML 2026.

Formula (Definition 3.3):
  E^K_{R->t} = 1 - log(P(B=1 | do(B_R=1))) / log(P(B_t=1))

Supports:
  --config <path>   : load pathway config from JSON (default: pathway_config.json)
  --diagnose        : print sensitivity analysis (elasticities, partial derivatives)
  --validate        : verify output matches expected baseline score
  --batch <path>    : evaluate multiple configs from a JSON array, rank by score
"""

import math
import sys
import json
import os
import argparse

# ─── Default baseline probabilities (fallback if no config) ───
DEFAULT_PROBS = {
    "P_B_given_A": 0.55,
    "P_C_given_B": 0.80,
    "P_D_given_C": 0.05,
    "P_E_given_D": 0.20,
}
DEFAULT_TARGET_PROB = 0.0005
DEFAULT_ROOT_CAUSE = ["A"]
DEFAULT_EDGES = [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]]
DEFAULT_NODES = ["A", "B", "C", "D", "E"]

# Edge to probability key mapping (order-dependent for default chain)
EDGE_PROB_KEYS = {
    ("A", "B"): "P_B_given_A",
    ("B", "C"): "P_C_given_B",
    ("C", "D"): "P_D_given_C",
    ("D", "E"): "P_E_given_D",
}


def load_config(config_path):
    """Load pathway configuration from JSON file."""
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def extract_chain_probs(config):
    """Extract ordered list of conditional probabilities from config edges."""
    edges = config.get("pathway", {}).get("edges", DEFAULT_EDGES)
    probs_dict = config.get("probabilities", {})

    probs = []
    for i, edge in enumerate(edges):
        if isinstance(edge, list):
            from_node, to_node = edge[0], edge[1]
            edge_key = (from_node, to_node)
            # Look up by mapping or naming convention
            prob_key = EDGE_PROB_KEYS.get(edge_key)
            if prob_key and prob_key in probs_dict:
                probs.append(probs_dict[prob_key])
            else:
                # Try naming convention: P_{to}_given_{from}
                key = "P_{}_given_{}".format(to_node, from_node)
                if key in probs_dict:
                    probs.append(probs_dict[key])
                elif len(edges) == 4 and i < 4:
                    # Fall back to positional for default 4-edge chain
                    def_keys = list(DEFAULT_PROBS.keys())
                    if i < len(def_keys) and def_keys[i] in probs_dict:
                        probs.append(probs_dict[def_keys[i]])
                    else:
                        raise KeyError(
                            "No probability found for edge {}->{}".format(from_node, to_node)
                        )
                else:
                    raise KeyError(
                        "No probability found for edge {}->{}".format(from_node, to_node)
                    )
        elif isinstance(edge, dict):
            probs.append(edge.get("prob", 0.0))
    return probs


def pathway_explanation_score(probs, target_prob):
    """Compute the pathway explanation score for a product of conditional probs."""
    joint_cond = 1.0
    for p in probs:
        joint_cond *= p

    # Numerical safety: clip extremely small values
    eps = 1e-300
    joint_cond = max(joint_cond, eps)
    target_prob = max(target_prob, eps)

    num = math.log(joint_cond)
    den = math.log(target_prob)
    score = 1.0 - num / den

    return score, joint_cond


def sensitivity_analysis(probs, target_prob):
    """Compute partial derivatives and elasticities for each parameter."""
    eps = 1e-300
    ln_target = math.log(max(target_prob, eps))

    results = []
    for i, p in enumerate(probs):
        dscore_dp = -1.0 / (max(p, eps) * ln_target)
        # Elasticity: % change in score per 1% change in p
        joint = 1.0
        for pj in probs:
            joint *= pj
        joint = max(joint, eps)
        base_score = 1.0 - math.log(joint) / ln_target
        if abs(base_score) > 1e-10:
            elasticity = dscore_dp * p / base_score
        else:
            elasticity = float("inf")
        results.append({
            "index": i,
            "value": p,
            "dscore_dp": dscore_dp,
            "elasticity": elasticity,
        })

    # Compute score under +/-10% perturbation for each
    for r in results:
        i = r["index"]
        perturbed = list(probs)
        perturbed[i] = min(probs[i] * 1.10, 0.999)
        s, _ = pathway_explanation_score(perturbed, target_prob)
        r["score_if_plus10pct"] = s
        perturbed[i] = max(probs[i] * 0.90, 0.001)
        s, _ = pathway_explanation_score(perturbed, target_prob)
        r["score_if_minus10pct"] = s

    # Sensitivity to target_prob
    joint = 1.0
    for p in probs:
        joint *= p
    joint = max(joint, eps)
    dscore_dtarget = math.log(joint) / (max(target_prob, eps) * (ln_target ** 2))

    s_plus, _ = pathway_explanation_score(probs, min(target_prob * 1.10, 0.999))
    s_minus, _ = pathway_explanation_score(probs, max(target_prob * 0.90, 1e-10))

    results.append({
        "index": -1,
        "parameter": "P(E) target marginal",
        "value": target_prob,
        "dscore_dp": dscore_dtarget,
        "elasticity": None,
        "score_if_plus10pct": s_plus,
        "score_if_minus10pct": s_minus,
    })

    return results


def print_diagnostics(probs, target_prob, root_cause, edges):
    """Print detailed sensitivity analysis."""
    sep = "=" * 70
    print()
    print(sep)
    print("SENSITIVITY ANALYSIS")
    print(sep)
    print()

    sa = sensitivity_analysis(probs, target_prob)

    # Rank by elasticity magnitude
    param_results = [r for r in sa if r["index"] >= 0]
    param_results.sort(key=lambda r: abs(r["elasticity"]), reverse=True)

    header = "{0:<6} {1:<6} {2:<10} {3:<14} {4:<12} {5:<14} {6:<14}".format(
        "Rank", "Param", "Value", "dScore/dp", "Elasticity", "+10% Score", "-10% Score"
    )
    print(header)
    print("-" * 76)
    for rank, r in enumerate(param_results, 1):
        idx = r["index"]
        if idx < len(edges):
            edge_label = "{}->{}".format(edges[idx][0], edges[idx][1])
        else:
            edge_label = "p{}".format(idx)
        print("{0:<6} {1:<6} {2:<10.4f} {3:<14.6f} {4:<12.6f} {5:<14.4f} {6:<14.4f}".format(
            rank, edge_label, r["value"], r["dscore_dp"], r["elasticity"],
            r["score_if_plus10pct"], r["score_if_minus10pct"]
        ))

    # Target prob sensitivity
    tr = sa[-1]
    print("{0:<6} {1:<6} {2:<10.6f} {3:<14} {4:<12} {5:<14.4f} {6:<14.4f}".format(
        "", "P(E)", tr["value"], "(see below)", "N/A",
        tr["score_if_plus10pct"], tr["score_if_minus10pct"]
    ))
    print()

    print("Optimization priority (highest elasticity first):")
    for rank, r in enumerate(param_results, 1):
        idx = r["index"]
        if idx < len(edges):
            edge_label = "P({}|{})".format(edges[idx][1], edges[idx][0])
        else:
            edge_label = "p{}".format(idx)
        direction = "INCREASE" if r["dscore_dp"] > 0 else "DECREASE"
        print("  {}. {}: {} (elasticity = {:.4f})".format(
            rank, edge_label, direction, r["elasticity"]
        ))

    print()
    print("Target P(E) sensitivity: +10% -> score {:.4f}, -10% -> score {:.4f}".format(
        tr["score_if_plus10pct"], tr["score_if_minus10pct"]
    ))
    print()


def print_score_report(probs, target_prob, root_cause, edges, nodes, config=None):
    """Print the full score report."""
    score, joint_cond = pathway_explanation_score(probs, target_prob)

    sep = "=" * 70
    print(sep)
    print("Pathway Explanation Score - Section 5 (Homelessness Example)")
    if config and config.get("pathway", {}).get("description"):
        print("Context: {}".format(config["pathway"]["description"]))
    print(sep)
    print()

    # Print pathway structure
    edge_strs = ["{}->{}".format(f, t) for f, t in edges]
    print("Pathway edges: {}".format(", ".join(edge_strs)))
    print("Root cause: {}".format(root_cause))
    print("Target:    E (chronic homelessness, 18+ months on streets)")
    print()

    print("Conditional probabilities:")
    for i, p in enumerate(probs):
        if i < len(edges):
            print("  P({}|{}) = {}".format(edges[i][1], edges[i][0], p))
    print("  P(E)    = {}  (marginal target probability)".format(target_prob))
    print()

    product_str = " x ".join(str(p) for p in probs)
    rc_str = "+".join(root_cause)
    print("  P(B=1 | do({}=1)) = {}".format(rc_str, product_str))
    print("                    = {:.6f}".format(joint_cond))
    print()

    ln_joint = math.log(joint_cond)
    ln_target = math.log(target_prob)
    ratio = ln_joint / ln_target
    print("  ln(P(pathway | do(root))) = ln({:.6f}) = {:.6f}".format(joint_cond, ln_joint))
    print("  ln(P(target))              = ln({}) = {:.6f}".format(target_prob, ln_target))
    print()
    print("  Pathway Explanation Score = 1 - ln({:.6f}) / ln({})".format(joint_cond, target_prob))
    print("                             = 1 - ({:.6f}) / ({:.6f})".format(ln_joint, ln_target))
    print("                             = 1 - {:.6f}".format(ratio))
    print("                             = {:.6f}".format(score))
    print()
    print(sep)
    print("  RESULT: Pathway Explanation Score = {:.4f}".format(score))
    print(sep)

    return score, joint_cond


def validate_baseline(score, expected=0.2861, tol=0.001):
    """Verify score matches baseline within tolerance."""
    if abs(score - expected) > tol:
        print("VALIDATION FAILED: score {:.4f} differs from baseline {:.4f}".format(score, expected))
        return False
    print("VALIDATION PASSED: score {:.4f} matches baseline {:.4f} within {}".format(score, expected, tol))
    return True


def batch_evaluate(batch_path):
    """Evaluate multiple configs and rank by score."""
    with open(batch_path, "r") as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        print("ERROR: batch config must be a JSON array")
        sys.exit(1)

    results = []
    for i, cfg in enumerate(configs):
        try:
            probs = extract_chain_probs(cfg)
            target_prob = cfg.get("target", {}).get("marginal_prob", DEFAULT_TARGET_PROB)
            root_cause = cfg.get("root_cause", DEFAULT_ROOT_CAUSE)
            edges = cfg.get("pathway", {}).get("edges", DEFAULT_EDGES)
            nodes = cfg.get("pathway", {}).get("nodes", DEFAULT_NODES)
            score, joint = pathway_explanation_score(probs, target_prob)
            results.append({
                "index": i,
                "config": cfg,
                "score": score,
                "joint": joint,
                "probs": probs,
                "target_prob": target_prob,
                "root_cause": root_cause,
                "edges": edges,
                "rationale": cfg.get("_rationale", ""),
            })
        except Exception as e:
            print("Config {}: ERROR - {}".format(i, e))
            results.append({
                "index": i,
                "config": cfg,
                "score": None,
                "error": str(e),
            })

    # Filter and sort
    valid = [r for r in results if r["score"] is not None]
    valid.sort(key=lambda r: r["score"], reverse=True)

    print("=" * 70)
    print("BATCH RESULTS ({} valid of {} total)".format(len(valid), len(configs)))
    print("=" * 70)
    header = "{0:<6} {1:<10} {2:<12} {3:<15} {4:<30} {5}".format(
        "Rank", "Score", "Joint", "Root Cause", "Edges", "Rationale"
    )
    print(header)
    print("-" * 100)
    for rank, r in enumerate(valid, 1):
        edges_str = ", ".join("{}->{}".format(f, t) for f, t in r["edges"])
        rc_str = "+".join(r["root_cause"])
        print("{0:<6} {1:<10.4f} {2:<12.6f} {3:<15} {4:<30} {5}".format(
            rank, r["score"], r["joint"], rc_str, edges_str, r["rationale"][:50]
        ))

    # Print failures
    failed = [r for r in results if r["score"] is None]
    if failed:
        print("\nFailed: {} configs".format(len(failed)))
        for r in failed:
            print("  Config {}: {}".format(r["index"], r.get("error", "unknown")))

    # Best result
    if valid:
        best = valid[0]
        print()
        print("BEST: Score = {:.4f} (config index {})".format(best["score"], best["index"]))
        print("  Root cause: {}".format(best["root_cause"]))
        print("  Edges: {}".format(best["edges"]))
        print("  Probs: {}".format(best["probs"]))
        print("  Target P(E): {}".format(best["target_prob"]))
        print()
        print("=" * 70)
        print("  RESULT: Pathway Explanation Score = {:.4f}".format(best["score"]))
        print("=" * 70)

    # Save full results
    out_path = "/repo/batch_pathway_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": valid, "failed": len(failed)}, f, indent=2)
    print("\nFull results written to {}".format(out_path))

    return valid


def main():
    parser = argparse.ArgumentParser(description="Pathway Explanation Score")
    parser.add_argument("--config", type=str, default="pathway_config.json",
                        help="Path to JSON config file")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print sensitivity analysis")
    parser.add_argument("--validate", action="store_true",
                        help="Validate against expected baseline score")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch evaluate multiple configs from JSON array")
    args = parser.parse_args()

    # Batch mode
    if args.batch:
        batch_evaluate(args.batch)
        return

    # Load config or use defaults
    config = load_config(args.config)

    if config:
        probs = extract_chain_probs(config)
        target_prob = config.get("target", {}).get("marginal_prob", DEFAULT_TARGET_PROB)
        root_cause = config.get("root_cause", DEFAULT_ROOT_CAUSE)
        edges = config.get("pathway", {}).get("edges", DEFAULT_EDGES)
        nodes = config.get("pathway", {}).get("nodes", DEFAULT_NODES)
    else:
        def_keys = list(DEFAULT_PROBS.keys())
        probs = [DEFAULT_PROBS[k] for k in def_keys]
        target_prob = DEFAULT_TARGET_PROB
        root_cause = DEFAULT_ROOT_CAUSE
        edges = DEFAULT_EDGES
        nodes = DEFAULT_NODES

    # Print score report
    score, joint_cond = print_score_report(probs, target_prob, root_cause, edges, nodes, config)

    # Optional diagnostics
    if args.diagnose:
        print_diagnostics(probs, target_prob, root_cause, edges)

    # Optional validation
    if args.validate:
        print()
        validate_baseline(score)

    # Write result file (always, for downstream parsing)
    with open("/repo/pathway_score_result.txt", "w") as f:
        f.write("{:.6f}\n".format(score))
    print("\nResult written to /repo/pathway_score_result.txt")


if __name__ == "__main__":
    main()
