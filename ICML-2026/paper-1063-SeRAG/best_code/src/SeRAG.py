# SeRAG.py
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re
import logging
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple, Set

from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
from src.struct_entropy import StructEntropy, CommunityNode 
from src.ner import SpacyNER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

class SeRAG:
    def __init__(self, global_config):
        self.config = global_config
        self.dataset_name = global_config.dataset_name
        logger.info(f"Initializing SeRAG with config: {self.config}")
        self.alpha_semantic = getattr(self.config, 'alpha_semantic', 0.7)
        self.beta_logical = getattr(self.config, 'beta_logical', 0.2)
        self.gamma_distance = getattr(self.config, 'gamma_distance', 0.1)
        
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        
        self.spacy_ner = SpacyNER(self.config.spacy_model) 
        
        self.struct_entropy_model: StructEntropy = None
        self.encoding_tree_root: CommunityNode = None
        
        self.community_summaries: Dict[int, np.ndarray] = {} 
        self.chunk_se_weights: Dict[str, float] = {} 
        
        self.passage_hash_id_to_entities = {} 
        self.hash_id_to_chunk_index = {} 
        self.chunk_index_to_hash_id = {} 

        self.ner_filepath = os.path.join(self.config.working_dir, self.dataset_name, "ner_results.json")

    def load_embedding_store(self):
        working_dir = getattr(self.config, 'working_dir', '.')
        dataset_name = getattr(self.config, 'dataset_name', 'default_dataset')
        
        self.chunk_embedding_store = EmbeddingStore(
            self.config.embedding_model, 
            db_filename=os.path.join(working_dir, dataset_name, "chunk_embedding.parquet"), 
            batch_size=self.config.batch_size, 
            namespace="chunk"
        )
        self.entity_embedding_store = EmbeddingStore(
            self.config.embedding_model, 
            db_filename=os.path.join(working_dir, dataset_name, "entity_embedding.parquet"), 
            batch_size=self.config.batch_size, 
            namespace="entity"
        )

    
    def _calculate_semantic_edges(self, all_chunk_embeddings: np.ndarray, all_chunk_hash_ids: List[str]) -> Dict[Tuple[str, str], float]:
        logger.info("Calculating Semantic Edges (KNN).")
        K = getattr(self.config, 'semantic_k', 20) 
        norm_embeddings = normalize(all_chunk_embeddings, axis=1, norm='l2')
        knn = NearestNeighbors(n_neighbors=K + 1, metric='cosine', algorithm='auto')
        knn.fit(norm_embeddings)
        distances, indices = knn.kneighbors(norm_embeddings)
        
        semantic_edges = {}
        for i in tqdm(range(len(all_chunk_hash_ids)), desc="Processing Semantic Edges"):
            source_hash = all_chunk_hash_ids[i]
            for j in range(1, K + 1):
                target_hash = all_chunk_hash_ids[indices[i, j]]
                similarity = 1.0 - distances[i, j]
                edge = tuple(sorted((source_hash, target_hash)))
                if edge not in semantic_edges or similarity > semantic_edges[edge]:
                    semantic_edges[edge] = similarity
        return semantic_edges

    def _calculate_logical_edges(self, all_chunk_hash_ids: List[str]) -> Dict[Tuple[str, str], float]:
        logger.info("Calculating Logical Edges (Entity Co-occurrence).")
        entity_text_to_hash_id = self.entity_embedding_store.text_to_hash_id
        chunk_hash_id_to_entities = self.passage_hash_id_to_entities 
        
        logical_edges = {}
        for i in tqdm(range(len(all_chunk_hash_ids)), desc="Processing Logical Edges"):
            hash_i = all_chunk_hash_ids[i]
            valid_ents_i = {entity_text_to_hash_id[e] for e in chunk_hash_id_to_entities.get(hash_i, set()) if e in entity_text_to_hash_id}
            if not valid_ents_i: continue

            for j in range(i + 1, len(all_chunk_hash_ids)):
                hash_j = all_chunk_hash_ids[j]
                valid_ents_j = {entity_text_to_hash_id[e] for e in chunk_hash_id_to_entities.get(hash_j, set()) if e in entity_text_to_hash_id}
                if not valid_ents_j: continue
                
                common = valid_ents_i.intersection(valid_ents_j)
                min_common_entities = getattr(self.config, 'min_common_entities', 1)
                if len(common) >= min_common_entities:
                    # if common:
                    logical_edges[tuple(sorted((hash_i, hash_j)))] = len(common) / max(len(valid_ents_i), len(valid_ents_j))
        return logical_edges
    
    def _calculate_distance_edges(self, all_chunk_hash_ids: List[str]) -> Dict[Tuple[str, str], float]:
        logger.info("Calculating Distance Edges (Gaussian Similarity).")
        sigma = getattr(self.config, 'distance_sigma', 5) 
        window_size = sigma * 3
        sigma_sq_2 = 2 * (sigma ** 2)
        
        distance_edges = {}
        for i in tqdm(range(len(all_chunk_hash_ids)), desc="Processing Distance Edges"):
            h_i = all_chunk_hash_ids[i]
            idx_i = self.hash_id_to_chunk_index.get(h_i)
            if idx_i is None: continue
            for j in range(i + 1, min(i + 1 + window_size, len(all_chunk_hash_ids))):
                h_j = all_chunk_hash_ids[j]
                idx_j = self.hash_id_to_chunk_index.get(h_j)
                if idx_j is None: continue
                similarity = math.exp(-((idx_i - idx_j) ** 2) / sigma_sq_2)
                if similarity > 0.1:
                    distance_edges[tuple(sorted((h_i, h_j)))] = similarity 
        return distance_edges
    
    def _merge_and_normalize_edges(self, all_chunk_hash_ids, s_edges, l_edges, d_edges):
        alpha, beta, gamma = self.alpha_semantic, self.beta_logical, self.gamma_distance
        all_edges = set(s_edges.keys()) | set(l_edges.keys()) | set(d_edges.keys())
        
        final_edges_hashes = []
        final_weights = []
        for h_i, h_j in all_edges:
            w = (alpha * s_edges.get((h_i, h_j), 0.0) + 
                 beta * l_edges.get((h_i, h_j), 0.0) + 
                 gamma * d_edges.get((h_i, h_j), 0.0))
            if w > 1e-6:
                final_edges_hashes.append((h_i, h_j))
                final_weights.append(w)
        return final_edges_hashes, final_weights

    def _get_chunk_index_from_text(self, hash_id_to_chunk: Dict[str, str]):
        index_pattern = re.compile(r'^(\d+):')
        self.hash_id_to_chunk_index = {}
        for hash_id, text in hash_id_to_chunk.items():
            match = index_pattern.match(text.strip())
            if match:
                self.hash_id_to_chunk_index[hash_id] = int(match.group(1))
                
    
    def _get_summary_embedding(self, community_node: CommunityNode, chunk_embeddings: np.ndarray):
        if not community_node.node_ids: return None
        
        weighted_sum = np.zeros(chunk_embeddings.shape[1])
        total_weight = 0.0
        
        for idx in community_node.node_ids: 
            h_id = self.chunk_index_to_hash_id.get(idx)
            weight = max(0.0, self.chunk_se_weights.get(h_id, 0.0))
            
            weighted_sum += weight * chunk_embeddings[idx] 
            total_weight += weight
            
        if total_weight > 1e-6:
            avg_emb = weighted_sum / total_weight
            return normalize(avg_emb.reshape(1, -1), axis=1, norm='l2').flatten()
        return None
        
    def _traverse_and_compute_summaries(self, node: CommunityNode, chunk_embeddings: np.ndarray):
        summary_emb = self._get_summary_embedding(node, chunk_embeddings)
        if summary_emb is not None:
            self.community_summaries[node.ID] = summary_emb
        
        for child in node.children:
            self._traverse_and_compute_summaries(child, chunk_embeddings)


    def index(self, passages):
        self.chunk_embedding_store.insert_text(passages)
        h_to_chunk = self.chunk_embedding_store.get_hash_id_to_text()
        
        self._get_chunk_index_from_text(h_to_chunk) 
        self.chunk_index_to_hash_id = {idx: h for h, idx in self.hash_id_to_chunk_index.items()}
        
        all_h_ids = list(h_to_chunk.keys())
        all_embs = np.array(self.chunk_embedding_store.embeddings)

        existing_p_ent, _, new_p_ids = self.load_existing_data(all_h_ids)
        if new_p_ids:
            new_p_ent, _ = self.spacy_ner.batch_ner({k: h_to_chunk[k] for k in new_p_ids}, self.config.max_workers)
            existing_p_ent.update(new_p_ent)
        self.save_ner_results(existing_p_ent, {})
        
        entity_nodes = set()
        self.passage_hash_id_to_entities = existing_p_ent
        for ents in existing_p_ent.values(): entity_nodes.update(ents)
        self.entity_embedding_store.insert_text(list(entity_nodes))

        s_e = self._calculate_semantic_edges(all_embs, all_h_ids)
        l_e = self._calculate_logical_edges(all_h_ids)
        d_e = self._calculate_distance_edges(all_h_ids)
        print(f"Semantic edges count: {len(s_e)}")
        print(f"Logical edges count: {len(l_e)}")
        print(f"Distance edges count: {len(d_e)}")
        raw_edges, raw_ws = self._merge_and_normalize_edges(all_h_ids, s_e, l_e, d_e)
        
        mapped_edges, final_ws = [], []
        for i, (u_h, v_h) in enumerate(raw_edges):
            if u_h in self.hash_id_to_chunk_index and v_h in self.hash_id_to_chunk_index:
                mapped_edges.append([self.hash_id_to_chunk_index[u_h], self.hash_id_to_chunk_index[v_h]])
                final_ws.append(raw_ws[i])
 
        e_tensor = torch.tensor(mapped_edges, dtype=torch.long, device=device)
        w_tensor = torch.tensor(final_ws, dtype=torch.float, device=device)
        logger.info(f"Edges shape: { e_tensor.shape}, Weights shape: {w_tensor.shape}")
        self.struct_entropy_model = StructEntropy(edges=e_tensor, weights=w_tensor)
        
        K = getattr(self.config, 'k_dim', 2)
        self.encoding_tree_root = self.struct_entropy_model.find_k_dim_entropy_tree(K)

        node_se_map = self.struct_entropy_model.calc_node_se_from_tree()

        se_values = node_se_map.cpu().detach().numpy() if torch.is_tensor(node_se_map) else node_se_map

        self.chunk_se_weights = {}
        for idx, val in enumerate(se_values):
            if idx in self.chunk_index_to_hash_id:
                h_id = self.chunk_index_to_hash_id[idx]
                self.chunk_se_weights[h_id] = float(val)
        
        self.community_summaries = {}
        self._traverse_and_compute_summaries(self.encoding_tree_root, all_embs)
        logger.info("Indexing completed.")
    
    def qa(self, questions):
        self._batch_llm_self_query(questions)
        retrieval_results = self.retrieve(questions)
        system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. \
            Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. \
            Conclude with "Answer: " using exact entity names from the source text. Provide a concise, definitive response without further elaboration. \
            Remember: If the context is irrelevant, use your internal knowledge to formulate the most plausible response."""

        all_messages = []
        for res in retrieval_results:
            context = "\n".join(res["sorted_passage"])
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\nQuestion: {res['question']}\nThought: "}
            ]
            all_messages.append(messages)
            
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(executor.map(self.llm_model.safe_infer, all_messages), total=len(all_messages), desc="QA Reading (Parallel)"))

        for ans, res in zip(all_qa_results, retrieval_results):
            safe_ans = ans if ans is not None else ""
            match = re.search(r'Answer:(.*)', safe_ans, re.DOTALL | re.IGNORECASE)
            res["pred_answer"] = match.group(1).strip() if match else safe_ans.strip()
            res["full_response"] = safe_ans
        return retrieval_results

    
    def retrieve(self, questions):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        all_pseudos = [q["pseudo_content"] for q in questions]
        all_pseudo_embs = self.config.embedding_model.encode(all_pseudos, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

        results = []
        for i, q_info in enumerate(tqdm(questions, desc="Retrieving")):
            
            question = q_info["question"]
            q_dense = all_pseudo_embs[i].flatten()

            seed_ids, seed_scores = self.get_seed_entities(question)
            candidate_communities = self._coarse_grained_matching(q_dense) 
            passages, scores = self._fine_grained_matching_and_fusion(q_dense, candidate_communities, seed_ids, seed_scores)
            
            results.append({
                "question": question,
                "sorted_passage": passages,
                "sorted_passage_scores": scores,
                "gold_answer": q_info.get("answer", None)
            })
        return results

    def _coarse_grained_matching(self, q_dense: np.ndarray) -> Dict[str, float]:
        if not self.community_summaries: return {}
        K_COARSE = getattr(self.config, 'retrieval_k_coarse', 10)
        
        c_ids = list(self.community_summaries.keys())
        sum_embs = np.array(list(self.community_summaries.values()))
        sims = np.dot(sum_embs, q_dense).flatten()
        
        top_indices = np.argsort(sims)[::-1][:K_COARSE]
        coarse_results = {}
        node_map = self._get_community_node_map(self.encoding_tree_root)
        
        # Adaptive threshold: only expand communities with similarity above dynamic cutoff
        adaptive_factor = getattr(self.config, "coarse_adaptive_factor", 0.0)
        max_sim = sims[top_indices[0]] if len(top_indices) > 0 else 0.0
        threshold = max_sim * adaptive_factor
        
        for idx in top_indices:
            if adaptive_factor > 0 and sims[idx] < threshold:
                continue  # skip communities below adaptive threshold
            node = node_map.get(c_ids[idx])
            if node:
                for chunk_idx in node.node_ids: # chunk_idx 是 int
                    h_id = self.chunk_index_to_hash_id.get(chunk_idx)
                    if h_id: coarse_results[h_id] = max(coarse_results.get(h_id, 0.0), sims[idx])
        return coarse_results

    def _fine_grained_matching_and_fusion(self, q_dense, candidate_chunks, seed_ids, seed_scores):
        g_coarse = getattr(self.config, 'gamma_coarse', 0.5)
        g_fine = getattr(self.config, 'gamma_fine', 0.5)
        top_k = self.config.retrieval_top_k
        
        candidate_h_ids = list(candidate_chunks.keys())
        if not candidate_h_ids: return [], []
        
        all_embs = np.array(self.chunk_embedding_store.embeddings)
        indices = [self.chunk_embedding_store.hash_id_to_idx[h] for h in candidate_h_ids]
        c_embs = all_embs[indices]
        
        dpr_sims = np.dot(c_embs, q_dense).flatten()
        dpr_norm = min_max_normalize(dpr_sims) if len(dpr_sims) > 1 else dpr_sims
        
        final_scores = {}
        for i, h_id in enumerate(candidate_h_ids):
            text = self.chunk_embedding_store.hash_id_to_text[h_id].lower()
            bonus = 0
            for s_id, s_score in zip(seed_ids, seed_scores):
                ent_text = self.entity_embedding_store.hash_id_to_text.get(s_id, "").lower()
                if ent_text:
                    count = text.count(ent_text)
                    if count > 0: bonus += s_score * math.log(1 + count)

            s_fine = 0.8 * dpr_norm[i] + math.log(1 + bonus)
            
            final_scores[h_id] = g_coarse * candidate_chunks[h_id] + g_fine * s_fine

        sorted_h = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]
        return [self.chunk_embedding_store.hash_id_to_text[h] for h in sorted_h], [final_scores[h] for h in sorted_h]

   
    def load_existing_data(self, chunk_hash_ids):
        existing_p = {}
        if os.path.exists(self.ner_filepath):
            with open(self.ner_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_p = {k: set(v) for k, v in data.get('passage_entities', {}).items()}
        new_ids = list(set(chunk_hash_ids) - set(existing_p.keys()))
        return existing_p, {}, new_ids

    def save_ner_results(self, p_ent, s_ent):
        os.makedirs(os.path.dirname(self.ner_filepath), exist_ok=True)
        with open(self.ner_filepath, 'w', encoding='utf-8') as f:
            json.dump({'passage_entities': {k: list(v) for k, v in p_ent.items()}, 'sentence_entities': {}}, f, ensure_ascii=False, indent=4)

    def _get_community_node_map(self, root: CommunityNode):
        m, q = {}, [root]
        while q:
            curr = q.pop(0)
            m[curr.ID] = curr
            q.extend(curr.children)
        return m


    def _batch_llm_self_query(self, questions):
 
        self_query_prompt = (
            "You are an expert at multi-step reasoning and information retrieval. "
            "For a given complex query, your task is to write a fact-dense passage that "
            "identifies intermediate entities and links them to provide a complete context. "
            "This passage will be used to improve document retrieval."
            "If you are unsure about the specific facts or entities, provide a broader categorical description instead of inventing details."
        )

        user_prompt = """Generate a factual, multi-hop descriptive passage for the following query.

Query: Jeremy Theobald and Christopher Nolan share what profession?
Passage: Jeremy Theobald is a British actor and film producer, known for his roles in early independent films. Christopher Nolan is a highly acclaimed film director, screenwriter, and producer. Both individuals overlap in the film industry as producers, having collaborated on projects like 'Following'.

Query: How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?
Passage: Ryu Hye-young is a South Korean actress who gained significant recognition for her role as Sung Bo-ra in the hit television series 'Reply 1988'. 'Reply 1988', which aired on tvN, consists of 20 episodes focused on the lives of five families in a Seoul neighborhood.

Query: Were Lonny and Allure both founded in the 1990s?
Passage: Lonny is an online lifestyle and home decor magazine that was founded in 2009 by Michelle Adams and Patrick Cline. In contrast, Allure is a major American women's magazine focused on beauty, which was founded in 1991 by Linda Wells. Therefore, only Allure was established during the 1990s.

Query: In what country was Lost Gravity manufactured?
Passage: Lost Gravity is a steel roller coaster located at Walibi Holland. It was manufactured by Mack Rides, a prominent German company specializing in amusement park rides and roller coasters. As a product of Mack Rides, Lost Gravity's origin of manufacture is Germany.

Query: {input_query}
Passage: """

        self_query_all_messages = []
    
        for q_info in questions:
            user_content = user_prompt.format(input_query=q_info['question'])
            self_query_all_messages.append([
                {"role": "system", "content": self_query_prompt},
                {"role": "user", "content": user_content}
            ])

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            sq_results = list(tqdm(
                executor.map(self.llm_model.self_query, self_query_all_messages), 
                total=len(self_query_all_messages), 
                desc="Self-Querying to generate descriptive sentences for questions"
            ))
   
        
        for q_info, sq_res in zip(questions, sq_results):
            res_content = sq_res.strip() if sq_res and isinstance(sq_res, str) else ""
            # print(f"Self-Query Result for '{q_info['question']}': {res_content}")
            q_info["pseudo_content"] = f"{q_info['question']} [SEP] {res_content}" if res_content else q_info["question"]
            # q_info["pseudo_content"] = f"{res_content}" if res_content else q_info["question"]

    def get_seed_entities(self, question):
        question_entities = list(self.spacy_ner.question_ner(question))
        if not question_entities:
            return [], []

        q_ent_embs = self.config.embedding_model.encode(
            question_entities, 
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        similarities = np.dot(self.entity_embeddings, q_ent_embs.T)
        
        seed_hash_ids = []
        seed_scores = []
        
        threshold = getattr(self.config, 'entity_sim_threshold', 0.6) 

        for i in range(len(question_entities)):
            best_idx = np.argmax(similarities[:, i])
            best_score = similarities[best_idx, i]

            if best_score >= threshold:
                h_id = self.entity_hash_ids[best_idx]
                if h_id not in seed_hash_ids:
                    seed_hash_ids.append(h_id)
                    seed_scores.append(best_score)
                    
        return seed_hash_ids, seed_scores
    
