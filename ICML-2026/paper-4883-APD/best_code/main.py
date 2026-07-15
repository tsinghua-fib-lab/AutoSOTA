# gcot_controller.py
from __future__ import annotations
import argparse
import copy
from dataclasses import field
import json
import traceback
import types
from typing import Callable, Dict, Any, List
from lib.eval import eval_ppl_wikitext
from transformers import AutoModelForCausalLM
from lib.api import ChatAPI
import torch
import torch.nn as nn
from lib.data import get_loaders
from lib.util import check_sparsity

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="cuda:0",
    )
    
    model.seqlen = model.config.max_position_embeddings 
    return model


class Node:
    stage: str 
    code: str    
    reward: float = -1e9
    parent: int | None = None
    id: int = -1
    ctx: Dict[str, Any] = field(default_factory=dict)

class TreeController:
    def __init__(self, llm, width=5, depth=3, device="cuda"):
        self.llm = llm
        self.WIDTH = width
        self.DEPTH = depth
        self.device = device
        self.nodes: List[Node] = []

    def _exec_fn(self, code: str) -> Dict[str, Any]:
        scope: Dict[str, Any] = {}
        exec(code, scope)
        return scope

    def expand_stage(self, stage: str, parent_ctx: Dict[str, Any], k: int) -> List[Node]:
        nodes = []
        if stage == "A":
            prompt = PROMPT_ANALYSIS
        elif stage == "H":
            prompt = PROMPT_HYP.format(stat_keys=json.dumps(parent_ctx["stat_keys"]))
        elif stage == "F":
            prompt = PROMPT_FORMULA
        elif stage == "C":
            prompt = PROMPT_COMPUTE
        else:
            raise ValueError(stage)

        for b in range(k):
            code = self.llm.chat(SYS, prompt + f"\n# branch={b}")
            nodes.append(Node(stage=stage, code=code, parent=parent_ctx.get("parent_id")))
        return nodes

    def build_leaf(self, chain_codes):
        # concatenate unique code blocks + LEAF wrapper
        full_code = "\n\n".join(chain_codes) + "\n\n" + self.llm.chat(SYS, PROMPT_LEAF)
        return full_code

    def run(self, args, train_loader, val_loader, sparsity, rounds) -> Dict[str, Any]:
        best = {"masks": None, "reward": float("-inf"), "code": None}

        for r in range(rounds):
            A_nodes = self.expand_stage("A", {}, self.WIDTH)
            for n_id, n in enumerate(A_nodes):
                scope = self._exec_fn(n.code)
                assert "analysis" in scope
                ldict = {"W": torch.ones(2,2), "G": None, "A": None}
                keys = list(scope["analysis"](ldict).keys())
                n.ctx = {"stat_keys": keys, "parent_id": None}
                n.id = len(self.nodes); self.nodes.append(n)

            H_pool = []
            for a in A_nodes[:self.WIDTH]:
                H_pool += self.expand_stage("H", a.ctx | {"parent_id": a.id}, self.WIDTH)
            F_pool = []
            for h in H_pool[:self.WIDTH]:
                F_pool += self.expand_stage("F", {"parent_id": h.id}, self.WIDTH)
            C_pool = []
            for f in F_pool[:self.WIDTH]:
                C_pool += self.expand_stage("C", {"parent_id": f.id}, self.WIDTH)

            leaves_built = 0
            for a in A_nodes[:self.WIDTH]:
                for h in H_pool[:self.WIDTH]:
                    for f in F_pool[:self.WIDTH]:
                        for c in C_pool[:self.WIDTH]:
                            torch.cuda.empty_cache()
                            chain = [a.code, h.code, f.code, c.code]
                            full_code = self.build_leaf(chain)
                            scope = self._exec_fn(full_code)
                            fn = scope["gcot_self_prune"] if "gcot_self_prune" in scope else scope.get("prune_once")
                            if fn is None:
                                continue
                            model = get_llm(args.model, args.cache)
                            try:
                                out = fn(model.model, train_loader, val_loader, float(sparsity), self.device)
                            except Exception as err:
                                print(traceback.format_exc())
                            reward = eval_ppl_wikitext(model.model, train_loader, device=args.device)
                            if check_sparsity(model.model) - args.sparsity > 0.01:
                                reward = float("-inf")
                            if reward > best["reward"]:
                                best["reward"] = reward
                                best["masks"]  = {k:v for k,v in out.items() if k != "__reward__"}
                                best["code"]   = full_code
                            leaves_built += 1
                            if leaves_built >= self.WIDTH: 
                                break
                        if leaves_built >= self.WIDTH: break
                    if leaves_built >= self.WIDTH: break
                if leaves_built >= self.WIDTH: break

        return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100, help="Number of LLM attempts.")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Global sparsity target.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--key", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str, default='wikitext2')
    parser.add_argument("--path", type=str)
    parser.add_argument("--nsamples", type=int, default=2)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--cache", type=str)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)

    args = parser.parse_args()


    trainloader, valloader = get_loaders(args.dataset, args.nsamples, path=args.path)



    llm = ChatAPI(args.key)

    ctl = TreeController(llm, width=args.width, depth=args.depth, device=args.device)
    res = ctl.run(args, trainloader, valloader, sparsity=args.sparsity, rounds=args.rounds)

    print("Best reward:", res["reward"])

if __name__ == "__main__":
    main()
