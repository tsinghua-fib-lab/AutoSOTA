
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import RealModeConfig
from .judge import build_judge
from .model_clients import build_model_client
from .questions import QuestionProvider

logger = logging.getLogger(__name__)


class LLMComparisonEnvironment:


    def __init__(self, config: RealModeConfig) -> None:
        self.config = config
        self.question_provider = QuestionProvider(
            questions=config.questions,
            question_file=config.question_file,
            shuffle=config.question_shuffle,
        )
        
        self.model_clients = []
        failed_models = []
        
        for model_cfg in config.models:
            try:
                client = build_model_client(model_cfg, config.huggingface)
                self.model_clients.append(client)
                logger.info(f"✅ Successfully loaded model: {model_cfg.name}")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {model_cfg.name}")
                logger.error(f"    Error: {e}")
                failed_models.append(model_cfg.name)
                logger.warning(f"⚠️  Skipping this model, continuing to load other models...")
        
        if not self.model_clients:
            raise RuntimeError(
                "All model loading failed! Please check model names and network connections.\n"
                f"Failed models: {', '.join(failed_models)}"
            )
        
        if failed_models:
            logger.warning(f"\n{'='*70}")
            logger.warning(f"Warning: The following models failed to load and have been skipped:")
            for model_name in failed_models:
                logger.warning(f"  - {model_name}")
            logger.warning(f"Successfully loaded models: {len(self.model_clients)}/{len(config.models)}")
            logger.warning(f"{'='*70}\n")
        
        self.judge = build_judge(config.judge)
        self.interactions: List[Dict[str, Any]] = []

    def compare(
        self,
        j: int,
        i: int,
        t: Optional[int] = None,
        sampling_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        model_j = self.model_clients[j]
        model_i = self.model_clients[i]
        question = self.question_provider.sample()
        answer_j = model_j.generate(question)
        answer_i = model_i.generate(question)

        verdict = self.judge.decide(
            question=question,
            answer_j=answer_j,
            answer_i=answer_i,
            model_j=model_j.name,
            model_i=model_i.name,
        )
        winner = verdict["winner"]
        obs = self._winner_to_obs(winner)

        payload = {
            "obs": obs,
            "winner": winner,
            "question": question,
            "model_j": model_j.name,
            "model_i": model_i.name,
            "answer_j": answer_j,
            "answer_i": answer_i,
            "verdict": verdict,
            "time_index": t,
            "sampling_method": sampling_method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.interactions.append(payload)
        return payload

    @staticmethod
    def _winner_to_obs(winner: str) -> int:
        if winner == "model_j":
            return 1
        if winner == "model_i":
            return 0
        if winner == "tie":
            from random import random

            return int(random() >= 0.5)
        raise ValueError(f"Unknown verdict result: {winner}")

    def save_interactions(self, output_dir: Path, append_to_file: str = None, start_index: int = 0) -> Path:

        output_dir.mkdir(parents=True, exist_ok=True)
        
        if append_to_file:
            
            file_path = Path(append_to_file)
            mode = "a"
            logger.info(f"Appending interactions to existing file (starting from index {start_index})")
        else:
            file_path = output_dir / f"interactions_{self._timestamp()}.jsonl"
            mode = "w"
            start_index = 0
        
        new_interactions = [r for r in self.interactions if r.get('time_index', 0) >= start_index]
        
        with file_path.open(mode, encoding="utf-8") as f:
            for record in new_interactions:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(new_interactions)} new interactions")
        return file_path

    def save_partial_order_graph(self, output_dir: Path, partial_order: list, format: str = 'png') -> Optional[Path]:

        try:
            import graphviz
        except ImportError:
            logger.warning("Graphviz is not installed, skipping DAG visualization. Install method: pip install graphviz")
            return None
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        m = len(self.model_clients)
        model_names = [client.name.split('/')[-1] for client in self.model_clients]
        
        adj_matrix = np.zeros((m, m), dtype=bool)
        for i, j in partial_order:
            adj_matrix[i, j] = True
        
        better_count = adj_matrix.sum(axis=0)
        max_layers = int(better_count.max()) + 1
        layers = [[] for _ in range(max_layers)]
        
        for idx in range(m):
            layer_num = int(better_count[idx])
            layers[layer_num].append(idx)
        
        layers = [layer for layer in layers if layer]
        
        logger.info(f"DAG layering result: {len(layers)} layers")
        for layer_idx, layer in enumerate(layers):
            logger.info(f"  Layer {layer_idx} ({len(layer)} models): {[model_names[i] for i in layer]}")
        
        dot = graphviz.Digraph(comment='Model Partial Order')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        dot.attr(fontsize='12')
        dot.attr(ranksep='1.0')
        
        for layer_idx, layer in enumerate(layers):
            with dot.subgraph() as s:
                s.attr(rank='same')
                for node_idx in layer:
                    if layer_idx == 0:
                        fillcolor = 'gold'
                    elif layer_idx == len(layers) - 1:
                        fillcolor = 'lightcoral'
                    else:
                        fillcolor = 'lightblue'
                    
                    s.node(str(node_idx), model_names[node_idx], 
                          fillcolor=fillcolor, 
                          style='rounded,filled')
        
        node_to_layer = {}
        for layer_idx, layer in enumerate(layers):
            for node_idx in layer:
                node_to_layer[node_idx] = layer_idx
        
        edges_drawn = 0
        edges_skipped = 0
        
        for i, j in partial_order:
            layer_i = node_to_layer.get(i, -1)
            layer_j = node_to_layer.get(j, -1)
            
            if layer_j == layer_i + 1:
                dot.edge(str(i), str(j), color='darkgreen', penwidth='2.0')
                edges_drawn += 1
            elif layer_j > layer_i + 1:
                edges_skipped += 1
        
        logger.info(f"DAG drawing: {len(layers)} layers, {edges_drawn} edges (skipped {edges_skipped} transitive edges)")
        
        file_path = output_dir / f"partial_order_dag_{self._timestamp()}"
        try:
            dot.render(file_path, format=format, cleanup=True)
            return Path(str(file_path) + f'.{format}')
        except Exception as e:
            logger.warning(f"DAG visualization saving failed: {e}")
            return None
    
    def save_interactions_csv(self, output_dir: Path, append_to_file: str = None, start_index: int = 0) -> Path:

        import csv
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if append_to_file:
            
            file_path = Path(append_to_file)
            mode = "a"  
            write_header = False
            logger.info(f"Appending CSV records to existing file (starting from index {start_index})")
        else:
            
            file_path = output_dir / f"interactions_{self._timestamp()}.csv"
            mode = "w"
            write_header = True
            start_index = 0
        
        if not self.interactions:
            logger.warning("No interactions to save, cannot save CSV")
            return file_path
        
        fieldnames = [
            "time_index",
            "timestamp",
            "sampling_method",
            "question",
            "model_j",
            "model_i",
            "answer_j",
            "answer_i",
            "winner",
            "obs",
            "judge_reasoning",
            "judge_confidence",
        ]
        
        new_interactions = [r for r in self.interactions if r.get('time_index', 0) > start_index]
        
        encoding = "utf-8-sig" if write_header else "utf-8"
        
        with file_path.open(mode, encoding=encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            for record in new_interactions:
                row = {
                    "time_index": record.get("time_index", ""),
                    "timestamp": record.get("timestamp", ""),
                    "sampling_method": record.get("sampling_method", ""),
                    "question": record.get("question", ""),
                    "model_j": record.get("model_j", ""),
                    "model_i": record.get("model_i", ""),
                    "answer_j": record.get("answer_j", ""),
                    "answer_i": record.get("answer_i", ""),
                    "winner": record.get("winner", ""),
                    "obs": record.get("obs", ""),
                    "judge_reasoning": record.get("verdict", {}).get("reasoning", ""),
                    "judge_confidence": record.get("verdict", {}).get("confidence", ""),
                }
                writer.writerow(row)
        
        logger.info(f"Saved {len(new_interactions)} new interactions to CSV")
        return file_path

    def save_config_snapshot(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"real_mode_config_{self._timestamp()}.yaml"
        import yaml

        with file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config.to_dict(), f, allow_unicode=True)
        return file_path

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def summary(self) -> Dict[str, Any]:
        return {
            "num_interactions": len(self.interactions),
            "models": [client.name for client in self.model_clients],
            "question_pool_size": len(self.question_provider),
            "judge": self.judge.to_dict(),
        }


