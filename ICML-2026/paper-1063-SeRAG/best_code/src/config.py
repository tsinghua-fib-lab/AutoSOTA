from dataclasses import dataclass
from typing import Union
from src.utils import LLM_Model

 
@dataclass
class SeRAGConfig:
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None
    llm_name: str = "gpt-4o-mini"
    chunk_token_size: int = 1000
    spacy_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 3
    k_dim: int = 2
    gamma_coarse: float = 0.4
    gamma_fine: float = 0.6
    retrieval_k_coarse: int = 10
    initial_entity_score: float = 1.0
    alpha_semantic: float = 0.45
    beta_logical: float = 0.45
    gamma_distance: float = 0.1
    semantic_k: int = 20
    distance_sigma: int = 5
    min_common_entities: int = 1
    coarse_adaptive_factor: float = 0.0  # 0=disabled

