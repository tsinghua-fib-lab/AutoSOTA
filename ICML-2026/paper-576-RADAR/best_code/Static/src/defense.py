import logging

from collections import Counter,defaultdict

from transformers import StoppingCriteriaList, MaxLengthCriteria, AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
stopword_set = set(stopwords.words('english'))
import time

import torch 
from itertools import combinations
from .helper import clean_str, StopOnTokens
import copy
from tqdm import tqdm
from torch import LongTensor, FloatTensor
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

import spacy
import os
import json
import random
from transformers import pipeline
from itertools import chain, combinations
import math

from src.decoding_methods import secure_decoding

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import time
# from .pairwise_em import PairwiseConflictEM  # not used, module missing

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from networkx.algorithms import approximation

import re

# 实现了一组用于 RAG 场景的防御/聚合策略（基类 + 多种具体方法），目的是在检索到的 top-k 文档可能被攻击/污染时仍尽量产出可靠答案。
# 所有防御都以相同输入格式 data_item（至少包含 question, topk_content, answer 等字段）为输入，输出最终的文本回答（字符串）。

logger = logging.getLogger('RRAG-main')

INJECTION = True # injection attack. if False, we consider passage modification attacks discussed in the appendix

def save_all_responses(save_path,response_list,data_item):
    all_data = []# it is a bit ugly... unnecessary read and write ; TODO: change it to jsonl instead
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        with open(save_path,'r') as f:
            all_data = json.load(f)
    all_data.append({"query":data_item['question'],
                     "answer":data_item['answer'],
                     "response":response_list})
    with open(save_path,'w') as f:
        json.dump(all_data,f,indent=4)

# 基类
class RRAG:

    def __init__(self,llm):
        self.llm = llm

    def query_undefended(self,data_item):
        query_prompt = self.llm.wrap_prompt(data_item,as_multi_choice='choices' in data_item)
        #response = None 
        response =  self.llm.query(query_prompt)
        logger.debug(f'Query_prompt:\n{query_prompt}')
        logger.debug(f'Response:\n{response}')
        logger.debug(f'Answer:\n{data_item["answer"]}')
        return response

    def query(self, data_item):
        raise NotImplementedError

    def _eval_response(self,response,data_item):
        answer = data_item['answer']
        response = clean_str(response)
        for ans in answer:
            if clean_str(ans) in response:
                return True 
        return False


class MinCutRRAG(RRAG):
    def __init__(self, llm, nli_model_path="DeBERTa-v3-large-mnli-fever-anli-ling-wanli", isolation_threshold=0.3, hrsim_alpha=0.0, score_rerank=False, contra_threshold=0.8, power_iters=10):
        super().__init__(llm)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_path).to(self.device)
        self.nli_model.eval()
        self.nli_batch_size = 32  # 默认 batch_size
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.isolation_threshold = isolation_threshold
        self.hrsim_alpha = hrsim_alpha
        self.score_rerank = score_rerank
        self.contra_threshold = contra_threshold
        self.power_iters = power_iters

    def query(self, data_item):
        docs = data_item['topk_content']
        k = len(docs)

        responses = []
        valid_docs = []  # 记录有效的文档索引
        for i in range(k):
            single_data_item = data_item.copy()
            single_data_item['topk_content'] = [docs[i]]
            single_prompt = self.llm.wrap_prompt(single_data_item, as_multi_choice='choices' in data_item, seperate=False)
            resp = self.llm._query(single_prompt)
            
            if "I don't know" in resp:  # 如果答案包含 "I don't know"，跳过该文档
                logger.info(f"[MinCut] Skipping document {i} as it contains 'I don't know'.")
                continue
            
            responses.append(resp)
            valid_docs.append(i)  # 记录有效文档的索引

        # 如果没有有效文档，直接对 LLM 进行原始询问并返回最终答案
        if not responses:
            logger.info("[MinCut] No valid documents found, querying LLM directly for the final answer.")
            final_prompt = self.llm.wrap_prompt(data_item, as_multi_choice='choices' in data_item, seperate=False)
            final_answer = self.llm._query(final_prompt)
            return final_answer  # 返回 LLM 查询结果

        logger.info(f"[MinCut] Responses: {responses}")
        logging.getLogger().handlers[0].flush()  # 强制刷新

        M, C = self._build_sim_and_conflict_matrices(data_item['question'], responses)
        logger.info(f"[MinCut] M: {M.tolist()}")
        
        total_docs = len(data_item['topk_content'])
        S, F = self.compute_scores_balanced(M, C, valid_docs=valid_docs, total_docs=total_docs)
        
        logger.info(f"[MinCut] S: {S.tolist()}")
        logger.info(f"[MinCut] F: {F.tolist()}")
        
        G = nx.DiGraph()  # Directed graph
        
        # Add source and sink nodes
        source = "source"
        sink = "sink"
        G.add_node(source)
        G.add_node(sink)
        
        # Add edges from source to documents (S_i values) only for valid docs
        for i in range(len(responses)):  # 使用 valid_docs 中的索引
            G.add_edge(source, i, capacity=S[i])  # S 和 F 中的索引依赖于 valid_docs
        
        # Add edges from documents to sink (F_i values) only for valid docs
        for i in range(len(responses)):  # 使用 valid_docs 中的索引
            G.add_edge(i, sink, capacity=F[i])  # S 和 F 中的索引依赖于 valid_docs
        
        # Add edges between documents (M_ij values) only for valid docs
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                G.add_edge(i, j, capacity=M[i, j])
                G.add_edge(j, i, capacity=M[i, j])
        
        min_cut_value, partition = nx.minimum_cut(G, source, sink)
        
        reachable, non_reachable = partition
        
        selected_docs = sorted([node for node in reachable if isinstance(node, int)])
        logger.info(f"[MinCut] Selected: {selected_docs}")
        
        if len(selected_docs) > 1:
            # Recompute embeddings and cos_sim for consistency (though could reuse from _build)
            selected_responses = [responses[i] for i in selected_docs]
            selected_embeddings = self.embed_model.encode(selected_responses)
            selected_cos_sim = util.cos_sim(selected_embeddings, selected_embeddings).cpu().numpy()

            # Calculate average off-diagonal cosine
            n_selected = len(selected_docs)
            off_diag_mask = ~np.eye(n_selected, dtype=bool)
            avg_cosine = np.mean(selected_cos_sim[off_diag_mask])
            logger.info(f"[MinCut] Average cosine of selected: {avg_cosine}")

            if avg_cosine < 1.0:
                # Exclude isolated: Compute per-doc average cosine to others (exclude self)
                doc_avg_cos = []
                for idx in range(n_selected):
                    others = [j for j in range(n_selected) if j != idx]
                    avg = np.mean(selected_cos_sim[idx, others])
                    doc_avg_cos.append(avg)

                # Exclude docs with avg_cos < 0.3 (arbitrary threshold for 'isolated')
                isolation_threshold = self.isolation_threshold
                new_selected = [selected_docs[idx] for idx in range(n_selected) if doc_avg_cos[idx] >= isolation_threshold]
                if new_selected != selected_docs:
                    selected_docs = sorted(new_selected)
                    logger.info(f"[MinCut] After excluding isolated: {selected_docs}")
        

        selected_data_item = data_item.copy()
        
        # Score-based re-ranking: sort selected docs by average M-matrix entailment
        if self.score_rerank and len(selected_docs) > 1:
            # Compute per-document average entailment to other selected docs
            avg_entail = []
            for idx_i, i in enumerate(selected_docs):
                row_entail = []
                for j in selected_docs:
                    if i != j and M[i, j] > 0:
                        row_entail.append(M[i, j])
                avg = np.mean(row_entail) if row_entail else 0.0
                avg_entail.append((idx_i, avg))
            # Sort by average entailment descending (highest confidence first)
            avg_entail.sort(key=lambda x: x[1], reverse=True)
            reordered = [selected_docs[idx] for idx, _ in avg_entail]
            if reordered != selected_docs:
                logger.info(f"[MinCut] Re-ranked docs by entailment: {[(d, s) for (_, d), (_, s) in zip(enumerate(selected_docs), avg_entail)]}")
                selected_docs = reordered
        
        selected_data_item['topk_content'] = [docs[valid_docs[i]] for i in selected_docs]  # 映射回原始文档
        
        final_prompt = self.llm.wrap_prompt(selected_data_item, as_multi_choice='choices' in data_item, seperate=False)
        final_answer = self.llm._query(final_prompt)
        return final_answer
    
    def _build_sim_and_conflict_matrices(self, question, responses):
        k = len(responses)
        # 初始化矩阵
        M = np.zeros((k, k), dtype=np.float32)
        C = np.zeros((k, k), dtype=np.float32)

        # 1. 预计算 Embeddings (用于过滤完全无关的噪声)
        embeddings = self.embed_model.encode(responses)
        cos_sim = util.cos_sim(embeddings, embeddings).cpu().numpy()

        # 2. 准备 NLI 推理对
        # 为了防御攻击，我们需要通过 NLI 来判断逻辑关系
        pairs = []
        indices = []
        
        for i in range(k):
            for j in range(k):
                if i == j: 
                    M[i, j] = 1.0 # 自己和自己完全一致
                    continue
                
                # 过滤掉 "I don't know" 这种无效回答
                # 它们既不支持别人，也不反驳别人，M=0, C=0
                if "I don't know" in responses[i] or "I don't know" in responses[j]:
                    continue

                # 构建 Prompt: 必须包含 question 以提供上下文
                premise = f"Question: {question} Answer: {responses[i]}"
                hypothesis = f"Question: {question} Answer: {responses[j]}"
                
                pairs.append((premise, hypothesis))
                indices.append((i, j))

        # 3. 批量 NLI 推理
        if len(pairs) > 0:
            
            batch_size = self.nli_batch_size
            nli_probs_map = {}
            
            for start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[start:start+batch_size]
                inputs = self.nli_tokenizer(
                    [p[0] for p in batch_pairs], 
                    [p[1] for p in batch_pairs], 
                    return_tensors='pt', truncation=True, padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.nli_model(**inputs)
                    # 使用 Softmax 归一化概率
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                
                # 存入临时字典
                for idx, prob in enumerate(probs):
                    real_idx = start + idx
                    i, j = indices[real_idx]
                    nli_probs_map[(i, j)] = prob

            # 4. 填充矩阵 (双向逻辑处理)
            for i in range(k):
                for j in range(i + 1, k):
                    if (i, j) not in nli_probs_map or (j, i) not in nli_probs_map:
                        continue

                    p_ij = nli_probs_map[(i, j)] 
                    p_ji = nli_probs_map[(j, i)]

                    contra_score = np.sqrt(p_ij[2] * p_ji[2]) 
                    
                    # 只有当矛盾概率显著大于中立概率时，才算有效矛盾
                    if contra_score > self.contra_threshold:
                        C[i, j] = C[j, i] = contra_score
                        M[i, j] = M[j, i] = 0.0 # 矛盾则不可能相似
                    else:
                        entail_score = np.sqrt(p_ij[0] * p_ji[0])
                        
                        M[i, j] = M[j, i] = entail_score
                        C[i, j] = C[j, i] = 0.0  # FIX: entailing pairs should not contribute to conflict matrix

        # HRSIM penalty: penalize edges where query-document similarities differ
        if self.hrsim_alpha > 0.0:
            question_embedding = self.embed_model.encode([question])[0]
            response_embeddings = self.embed_model.encode(responses)
            query_sims = []
            for resp_emb in response_embeddings:
                qs = float(np.dot(question_embedding, resp_emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(resp_emb) + 1e-8))
                query_sims.append(qs)
            for i in range(k):
                for j in range(i + 1, k):
                    if M[i, j] > 0:
                        penalty = self.hrsim_alpha * (query_sims[i] + query_sims[j])
                        M[i, j] = M[j, i] = max(M[i, j] - penalty, 0.0)

        return M, C
    def compute_scores_balanced(self, M, C, valid_docs, total_docs):
        k = len(valid_docs)
        if k == 1: return np.array([0.9]), np.array([0.1])

        # 1. 计算中心度
        adj = M + np.eye(k) * 0.01
        v = np.ones(k) / k
        for _ in range(self.power_iters):
            v = np.dot(adj, v); v /= (np.linalg.norm(v) + 1e-8)
        
        centrality = (v - v.min()) / (v.max() - v.min() + 1e-8)

        # 2. 计算基础 S 和 F
        S_raw = np.zeros(k)
        F_raw = np.zeros(k)

        for i in range(k):
            # S: 中心度 * 排名衰减
            rank_weight = np.exp(-valid_docs[i] / total_docs)
            rank_penalty = 1 - rank_weight  # 低排 → penalty ≈1，高排 → penalty ≈0
            S_raw[i] = (centrality[i] + 1e-8) * rank_weight
            
            weighted_conflict = 0
            for j in range(k):
                if i == j: continue
                # 对方的 centrality 代表了对方在共识中的“话语权”
                # 如果我跟一个“话语权”很高（处于共识中心）的人冲突，我的 F 应该很高
                weighted_conflict += C[i, j] * centrality[j]
            
            # 归一化冲突得分
            F_raw[i] = weighted_conflict / (np.sum(centrality) - centrality[i] + 1e-8)

        def final_scale(arr):
            if arr.ptp() < 1e-12: return np.ones_like(arr) * 0.5
            return (arr - arr.min()) / (arr.ptp() + 1e-12)

        S = final_scale(S_raw)
        F = final_scale(F_raw)

        S = np.clip(S, 0.01, 0.99)
        F = np.clip(F, 0.01, 0.99)

        return S, F


# 对每个检索文档单独让 LLM 给出独立回答（separate responses）；
# 用序贯 NLI（DeBERTa 多分类）判断两两回答是否“矛盾”；
# 构建有向图并迭代去掉出度大于剩余顶点/2 的节点；
# 从剩余节点中选入度为 0 的文档作为可信集，按原始顺序拼回去做最终回答。
# 复杂度O(k^2)
class GraphBasedRRAG(RRAG):

    def __init__(self,llm):
        self.llm = llm
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu" # for gpt-4o
        self.nli_tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/zs7353/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("/scratch/gpfs/zs7353/DeBERTa-v3-large-mnli-fever-anli-ling-wanli").to(device)

    def query(self, data_item):
        docs = data_item['topk_content']
        seperate_responses = self.llm.batch_query(self.llm.wrap_prompt(data_item,as_multi_choice='choices' in data_item,seperate=True))
        k = len(docs)
        # Build pairwise prompts to check for contradictory information
        prompts = []
        out_edges = {i: set() for i in range(k)}
        in_edges = {i: set() for i in range(k)}

        premises = []
        hypotheses = []
        pair_indices = []

        for i in range(k):
            for j in range(i + 1, k):
                premise = f"The answer to the question: {data_item['question']}\nis {seperate_responses[i]}."
                hypothesis = f"The answer to the question: {data_item['question']}\nis {seperate_responses[j]}."
                premises.append(premise)
                hypotheses.append(hypothesis)
                pair_indices.append((i, j))

        if premises:
            inputs = self.nli_tokenizer(premises, hypotheses, return_tensors='pt', truncation=True, padding=True)
            inputs = {key: value.to(self.nli_model.device) for key, value in inputs.items()}

            # Run the model on the batch
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

            # Process each batch item and update edges based on contradiction probability
            for idx, (i, j) in enumerate(pair_indices):
                contradiction_probability = probs[idx][2].item()
                if contradiction_probability >= 0.5 and "I don't know" not in seperate_responses[i] and "I don't know" not in seperate_responses[j]:
                    out_edges[i].add(j)
                    in_edges[j].add(i)
        
        # Iteratively remove vertices with out-degree greater than (number of remaining vertices)/2
        # remaining = set(range(k))
        remaining = set()
        # just don't take irrelevant docs? They are just noisy and useless
        for i in range(k):
            remaining.add(i)
        
        removal_occurred = True
        while removal_occurred:
            removal_occurred = False
            current_remaining = list(remaining)
            n_remaining = len(remaining)
            to_remove = []
            for v in current_remaining:
                current_out_degree = len(out_edges[v].intersection(remaining))
                if current_out_degree > math.floor(n_remaining / 2):
                    to_remove.append(v)
            if to_remove:
                removal_occurred = True
                for v in to_remove:
                    remaining.discard(v)
        
        # From the remaining documents, select those with in-degree 0
        selected = []
        for v in remaining:
            current_in_degree = len(in_edges[v].intersection(remaining))
            if current_in_degree == 0:
                selected.append(v)
        
        logger.info(selected)
        # Fallback: if no document has in-degree 0, use all remaining documents
        if not selected:
            selected = list(remaining)
        
        # Sort selected documents by their original rank order
        selected.sort()
        
        # Update the data_item to include only the selected documents
        new_data_item = data_item.copy()
        new_data_item['topk_content'] = [docs[i] for i in selected]
        
        # Create the final prompt using the LLM's wrap_prompt method
        ultimate_prompt = self.llm.wrap_prompt(new_data_item, as_multi_choice='choices' in data_item, seperate=False)
        
        # Return the final answer by querying the LLM
        final_answer = self.llm._query(ultimate_prompt)
        print("final_answer: ", final_answer)
        return final_answer

# 同样先对每个文档单独生成回答并用 NLI 判断是否矛盾，构建无向图（矛盾边）。
# 然后在候选顶点集合 z（非 "I don't know" 的回答集合）上穷举求最大独立集（MIS），若有多个取字典序最小者；
# 把 MIS 对应文档作为可信集合，拼接最终 prompt 并查询 LLM。
# 复杂度O(2^k)
class MISBasedRRAG(RRAG):

    def __init__(self, llm, err):
        self.llm = llm
        self.err = err
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"  # for gpt-4o
        self.nli_tokenizer = AutoTokenizer.from_pretrained("DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("DeBERTa-v3-large-mnli-fever-anli-ling-wanli").to(device)

    def query(self, data_item):
        # Retrieve the documents and get separate responses.
        docs = data_item['topk_content']
        start_time = time.perf_counter()
        seperate_responses = self.llm.batch_query(self.llm.wrap_prompt(data_item, as_multi_choice='choices' in data_item, seperate=True))
        end_time = time.perf_counter()
        logger.info(f"[MIS] Responses: {seperate_responses}")
        logging.getLogger().handlers[0].flush()  # 强制刷新
        k = len(docs)
        
        # Build an undirected graph: graph[i] holds all vertices j that contradict with document i.
        graph = {i: set() for i in range(k)}
        premises, hypotheses, pair_indices = [], [], [] 

        for i in range(k):
            for j in range(i + 1, k):
                premise = f"The answer to the question: {data_item['question']}\nis {seperate_responses[i]}."
                hypothesis = f"The answer to the question: {data_item['question']}\nis {seperate_responses[j]}."
                premises.append(premise)
                hypotheses.append(hypothesis)
                pair_indices.append((i, j))
        
        if premises:
            inputs = self.nli_tokenizer(premises, hypotheses, return_tensors='pt', truncation=True, padding=True)
            inputs = {key: value.to(self.nli_model.device) for key, value in inputs.items()}
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            end_time = time.perf_counter()
            print("time for NLI: ", end_time - start_time)
            
            # For each pair, add an undirected edge if the answers contradict.
            for idx, (i, j) in enumerate(pair_indices):
                contradiction_probability = probs[idx][2].item()
                x = random.random()
                if ("I don't know" not in seperate_responses[i] and "I don't know" not in seperate_responses[j]):
                    if contradiction_probability >= 0.5:
                        if x >= self.err:
                            graph[i].add(j)
                            graph[j].add(i)
                    else:
                        if x <= self.err:
                            graph[i].add(j)
                            graph[j].add(i)                            
                # if (contradiction_probability >= 0.5 and "I don't know" not in seperate_responses[i] and "I don't know" not in seperate_responses[j]):
                #     graph[i].add(j)
                #     graph[j].add(i)
        
        z = {i for i in range(k) if "I don't know" not in seperate_responses[i]}
        # z = {i for i in range(k)}
        
        # Compute the maximum independent set over the vertices.
        # Among all maximum independent sets, choose the one with the lexicographically smallest order.
        start_time = time.perf_counter()
        best_set = self._max_independent_set(graph, z)
        end_time = time.perf_counter()
        print("time for finding MIS: ", end_time - start_time)
        
        # Fallback: if best_set is empty, use all z documents.
        if not best_set:
            best_set = list(z)
            if not best_set:
                best_set = [i for i in range(k)]
        else:
            best_set = list(best_set)

        best_set.sort()  # sort in ascending order (better ranked docs have lower indices)
        logger.info(f"Selected document indices: {best_set}")
        
        # Update data_item with only the selected documents.
        new_data_item = data_item.copy()
        new_data_item['topk_content'] = [docs[i] for i in best_set]
        
        # Create the final prompt and query for the ultimate answer.
        ultimate_prompt = self.llm.wrap_prompt(new_data_item, as_multi_choice='choices' in data_item, seperate=False)
        print(ultimate_prompt)
        start_time = time.perf_counter()
        final_answer = self.llm._query(ultimate_prompt)
        end_time = time.perf_counter()
        print("time for the ultimate query: ", end_time - start_time)
        print("final_answer:", final_answer)
        return final_answer

    def _max_independent_set(self, graph, vertices):
        best_size = 0
        best_sets = []
        vertices_list = list(vertices)
        
        # Generate all subsets of vertices_list
        for subset in chain.from_iterable(combinations(vertices_list, r) for r in range(len(vertices_list) + 1)):
            subset = set(subset)
            if self._is_independent(subset, graph):
                subset_size = len(subset)
                if subset_size > best_size:
                    best_size = subset_size
                    best_sets = [tuple(sorted(subset))]
                elif subset_size == best_size:
                    best_sets.append(tuple(sorted(subset)))
                    
        # Return the lexicographically smallest independent set (as a tuple).
        if best_sets:
            return min(best_sets)
        else:
            return set()

    def _is_independent(self, subset, graph):
        for v in subset:
            for u in subset:
                if u != v and u in graph[v]:
                    return False
        return True

# 对每个文档分别预测（separate responses），把每个位置的预测按指数权重 gamma^i 加权计数，取最多票项作为最终预测（适用于多选题/离散选项）。
class WeightedMajorityVoting(RRAG):
    def __init__(self, llm, gamma=1):
        self.llm = llm
        self.gamma = gamma

    def query(self, data_item):
        # assume the prompt ask the LLM to output A., B., C., D., or E. No information found
        seperate_responses = self.llm.batch_query(self.llm.wrap_prompt(data_item,as_multi_choice='choices' in data_item,seperate=True))
        seperate_preds = []
        for response in seperate_responses:
            if "gpt" in self.llm.model_name: 
                if response.find('Answer') != -1:
                    response = response[(response.find('Answer')+7):].strip()
                else:
                    response = response.strip()
                if response[0] in 'ABCD':
                    seperate_preds.append(response[0]+'.')
                else:
                    seperate_preds.append('E.')
            else:
                response = response.strip()
                if len(response)>=2 and response[1]=='.' and response[0] in'ABCD':
                    seperate_preds.append(response[:2])
                else:
                    seperate_preds.append('E.')

        logger.debug(f'Seperate responses: {seperate_preds}')

        cntr = defaultdict(float)

        total_weight = 0
        total_weight_orig = 0
        for i, pred in enumerate(seperate_preds):
            if pred == 'E.':
                continue
            weight = self.gamma ** i  # First position weight=1, second=gamma, third=gamma^2, etc.
            total_weight += weight
            total_weight_orig += 1

        for i, pred in enumerate(seperate_preds):
            if pred == 'E.':
                continue 
            weight = self.gamma ** i      
            cntr[pred] += weight * total_weight_orig / total_weight
        
        cntr = Counter(cntr)
        cntr = cntr.most_common(2)

        if len(cntr)==0:
            pred = 'E.' # No information found.
        else:
            pred = cntr[0][0] 
        return pred

# 对每个文档单独生成回答，使用 spaCy 提取短语/关键字并以加权计数过滤出高频关键词（基于 absolute / relative 阈值）；
# 把关键词合并成 hints 放入 prompt 再做一次聚合查询以获取最终答案。
class WeightedKeywordAgg(RRAG):

    def __init__(self,llm,relative_threshold=0.3, absolute_threshold=3, abstention_threshold=1, gamma=1, longgen=False):
        self.llm = llm
        self.abstention_threshold = 1
        self.keyword_extractor = spacy.load("en_core_web_sm") 
        self.ignore_set = {'VERB','INTJ','ADP','AUX','CCONJ','DET','PART','PRON','SCONJ','PUNCT','SPACE'}
        self.absolute = absolute_threshold
        self.relative = relative_threshold
        self.gamma = gamma
        self.longgen = longgen # if it is long-form generation or short-form (we use slightly different prompt template)
        logger.info(f'abs: {absolute_threshold}, relative: {relative_threshold}')

    def query(self, data_item, abstention_threshold=None): 
        # override original threshold parameters if given
        abstention_threshold = abstention_threshold if abstention_threshold is not None else self.abstention_threshold
        if self.longgen:
            data_item['genhint'] = True # add a flag so that wrap_prompt() can retrieve the correct prompt template
        # make seperate predictions
        seperate_responses_raw = self.llm.batch_query(self.llm.wrap_prompt(data_item,as_multi_choice='choices' in data_item,seperate=True))
        abstained_idx = []
        seperate_responses = []
        logger.debug(f'Seperate responses:\n')
        total_weight = 0
        total_weight_orig = 0
        for i,x in enumerate(seperate_responses_raw):
            logger.debug(f'{i}: {x}\n')
            if "I don't" in x:
                abstained_idx.append(i)
            else:
                seperate_responses.append((x,  self.gamma ** i))
                total_weight +=  self.gamma ** i
                total_weight_orig += 1

        logger.debug(f'Number of retained responses: {len(seperate_responses)}')

        if len(seperate_responses) < abstention_threshold:
            logger.warning('Abstain from making response...')
            return "I don't know."
        
        def construct_phrase(token_list):
            ret = ''
            for token in token_list:
                ret+=token.lemma_+token.whitespace_
        # extract keyword/keyphrase
        all_extracted_phrase = []
        token_counter = defaultdict(int)
        for response, weight in seperate_responses:
            doc = self.keyword_extractor(response)
            phrase_list = [response.strip()] 
            tmp = []
            for token in doc:
                if token.pos_ in self.ignore_set:
                    if len(tmp)>0:
                        phrase = ''.join([x.lemma_+x.whitespace_ for x in tmp]).strip()
                        phrase_list.append(phrase)
                        phrase_list+=[x.lemma_ for x in tmp]
                        tmp = []
                else:
                    tmp.append(token)

            phrase = ''.join([x.lemma_+x.whitespace_ for x in tmp]).strip()
            phrase_list.append(phrase)
            phrase_list+=[x.lemma_ for x in tmp]
            phrase_list = set(phrase_list) # only consider unique keywords
            all_extracted_phrase.append(phrase_list)
            for phrase in phrase_list:
                token_counter[phrase]+=weight * total_weight_orig / total_weight

        # filtering 
        print(phrase_list)
        count_threshold = min(self.absolute,self.relative*len(seperate_responses))
        logger.debug(sorted(token_counter.items(), key=lambda x: (len(x[0]),x[0]), reverse=True))
        logger.debug(f'count_threshold,{count_threshold}')
        for token,count in list(token_counter.items()):
            if (count < count_threshold) or (token in punctuation) or (token in stopword_set)  or (self.longgen and ' ' not in token): # if it is long generation, we remove single words to reduce the size the keyword set...
                del token_counter[token]

        # generate keyword hints
        sorted_tokens = sorted(token_counter.items(), key=lambda x: (len(x[0]),x[0]), reverse=True)
        hints = ', '.join([f'{token}' for token,count in sorted_tokens])
        logger.debug(sorted_tokens)
        query_prompt = self.llm.wrap_prompt(data_item,as_multi_choice='choices' in data_item,hints=hints)
        logger.debug(f'Hint prompt:\n{query_prompt}')
        response = self.llm.query(query_prompt)
        logger.debug(f'Keyword aggregated response:\n{response}')

        return response

# 基于 model 的 secure_decoding（自定义解码器）对各个 prompt 做并行/批量分析；
# 先估计“我不知道”概率来过滤掉可能无关 prompt，然后对保留输入运行 secure_decoding（带 stopping criteria、eta、gamma 等超参），生成聚合输出。
class WeightedDecodingAgg(RRAG):
    def __init__(self,llm, eta, gamma=1, abstention_prob=None):
        self.llm = llm
        self.llm.model.secure_decoding = secure_decoding.__get__(self.llm.model, type(self.llm.model))
        self.temperature = 1.0 #args.temperature
        abstention_prob_list = {'/scratch/gpfs/zs7353/Llama-3.2-3B-Instruct': 0.99, 
                                '/scratch/gpfs/zs7353/Mistral-7B-Instruct-v0.2': 0.99, 
                                '/scratch/gpfs/zs7353/DeepSeek-R1-Distill-Qwen-7B': 0.99}
        if abstention_prob is None:
            self.abstention_prob = abstention_prob_list.get(llm.model_name, 0.99)
            logger.debug(f"Using default abstention probability: {self.abstention_prob}")

        self.gamma = gamma
        self.eta = eta
       
    def preprocess_input(self,data_item):
        prompt_list = self.llm.wrap_prompt(data_item,as_multi_choice='choices' in data_item,seperate=True)
        data_item_zero_shot = {"question": data_item["question"], "topk_content":[], "long_gen": True}
        prompt_zero_shot = self.llm.wrap_prompt(data_item_zero_shot,as_multi_choice='choices' in data_item,seperate=False)
        prompt_list.append(prompt_zero_shot)

        prompt_list_draft = [prompt + " I don't know" for prompt in prompt_list]

        # batched version 
        input_dict_draft = self.llm.tokenizer(prompt_list_draft, return_tensors="pt", padding=True).to("cuda")
        input_ids_draft = input_dict_draft.input_ids.to("cuda")
        attention_mask_draft = input_dict_draft.attention_mask.to("cuda")

        # compute the perplexity of the prompt "I don't know"
        with torch.no_grad():
            output_token_draft = self.llm.model(input_ids_draft, attention_mask=attention_mask_draft)
            logits_draft = output_token_draft.logits

        probs = torch.softmax(logits_draft, dim=-1)
        total_probability = torch.ones(input_ids_draft.shape[0]).to("cuda")

        input_dict = self.llm.tokenizer(prompt_list, return_tensors="pt", padding= True)
        start_index = input_dict.input_ids.size(1)

        for i in range(start_index, input_ids_draft.size(1) - 1):  # Exclude the last token since there's no next token to predict
            # Get the probability of the actual next token
            next_token_id = input_ids_draft[0, i + 1]  # The next token in the sequence
            next_token_prob = probs[:, i, next_token_id]
            total_probability *= next_token_prob

        #print(f"total_probability: {total_probability}")
        input_ids = input_dict.input_ids.to("cuda")
        attention_mask = input_dict.attention_mask.to("cuda")

        # filter the prompt with the probability of "I don't know" is greater than 0.9
        total_probability[-1] = 0.0 # last one is the zero-shot prompt
        ab_record = total_probability < self.abstention_prob
        input_ids = input_ids[ab_record]
        attention_mask = attention_mask[ab_record]
        return input_ids,attention_mask,ab_record

    def query(self, data_item):

        input_ids,attention_mask,ab_record = self.preprocess_input(data_item)

        if input_ids.shape[0] == 1: # only the no-retrieval prediction
            return "I don't know.", False
        
        # Initialize past_key_values for caching
        past_key_values = None
        generated_outputs = []

        stop_list = ["\n#", "\n##","\n###","\n####","\n#####"] + ["\n\n"] ################ seems to work fine
        stop_token_ids = [self.llm.tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
        stop_token_ids = [LongTensor(x).to("cuda") for x in stop_token_ids]
        stopping_criteria = StoppingCriteriaList([
            MaxLengthCriteria(max_length=len(input_ids[0]) + self.llm.max_output_tokens),
            StopOnTokens(stop_token_ids=stop_token_ids)
        ])
        
        generated_outputs = self.llm.model.secure_decoding(input_ids,
                                                           attention_mask=attention_mask,
                                                           stopping_criteria=stopping_criteria,
                                                           use_cache=False,
                                                           pad_token_id=self.llm.tokenizer.pad_token_id,
                                                           eos_token_id=self.llm.tokenizer.eos_token_id,
                                                           return_dict_in_generate=True,
                                                           temperature=self.temperature,
                                                           tokenizer=self.llm.tokenizer,
                                                           eta=self.eta,
                                                           gamma=self.gamma)

        generated_output_text = self.llm.tokenizer.decode(generated_outputs, skip_special_tokens=True)
        return generated_output_text
    
# 给 top-k 赋几何权重（gamma^i），多次从 top-k 按权采样子集，分别用 LLM 生成候选回答；
# 对这些候选回答计算 embedding（OpenAI 或 sentence-transformers），找最接近平均 embedding（centroid）的回答作为最终答案。
class RandomSamplingReQueryAgg(RRAG):
    def __init__(
        self, 
        llm,
        sample_size=5, 
        num_samples=3,
        gamma=1
    ):
        super().__init__(llm)
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.gamma = gamma

        self.use_openai = True
        self.openai_model = "text-embedding-ada-002"
        self.hf_model_name = "/scratch/gpfs/bi0600/all-mpnet-base-v2"

        if not self.use_openai:
            self.hf_model = SentenceTransformer( self.hf_model_name)
        else:
            self.client = OpenAI()

    def get_openai_embeddings(self, text_list):
        response = self.client.embeddings.create(
            model=self.openai_model,
            input=text_list
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def get_hf_embeddings(self, text_list):
        embeddings = self.hf_model.encode(text_list)
        return embeddings

    def query(self, data_item):
        question = data_item["question"]
        all_chunks = data_item["topk_content"]
        n = len(all_chunks)

        # 1) Assign geometric weights to chunks: gamma^i
        weights = np.array([self.gamma ** i for i in range(n)])
        weights /= weights.sum()  # normalize

        # 2) First-stage sampling: sample multiple subsets & query LLM
        sampled_responses = []
        for i in range(self.num_samples):
            sampled_chunks = list(
                np.random.choice(
                    all_chunks,
                    size=min(self.sample_size, n),
                    replace=False,
                    p=weights
                )
            )
            prompt = self.build_prompt(question, sampled_chunks)
            response = self.llm.query(prompt)
            sampled_responses.append(response)

        logger.debug(f"First-stage sampled responses:\n{sampled_responses}")

        # 3) Second-stage: pick the response closest to the mean embedding
        if self.use_openai:
            response_embeddings = self.get_openai_embeddings(sampled_responses)
        else:
            response_embeddings = self.get_hf_embeddings(sampled_responses)

        # Compute average (centroid) embedding
        response_embeddings = np.array(response_embeddings)
        avg_embedding = np.mean(response_embeddings, axis=0)

        # Find whichever response is closest to this centroid
        best_idx = None
        best_sim = -float("inf")
        for i, emb in enumerate(response_embeddings):
            cos_sim = dot(emb, avg_embedding) / (norm(emb) * norm(avg_embedding))
            if cos_sim > best_sim:
                best_idx = i
                best_sim = cos_sim
        
        final_response = sampled_responses[best_idx]
        logger.debug(f"Second-stage final response:\n{final_response}")
        return final_response

    def build_prompt(self, question, chunks):
        context_text = "\n\n".join(chunks)
        return f"Answer the following question based on the context below. It is very important that the answer should be based solely on evidence found in the context information. The answer should be as short as possible and can only use words found in the context information. \n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"


# 结合采样和关键词聚合：
# 多次按权采样、对采样得到的非弃权回答提取关键词并计权过滤，最后用关键词提示做最终查询。
class SamplingWithKeyWordAggregation(RRAG):
    def __init__(
        self, 
        llm,
        sample_size=5, 
        num_samples=3,
        gamma=1,
        relative_threshold=0.3,
        absolute_threshold=3,
        abstention_threshold=1,
    ):
        super().__init__(llm)
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.gamma = gamma

        self.keyword_extractor = spacy.load("en_core_web_sm") 
        self.ignore_set = {'VERB','INTJ','ADP','AUX','CCONJ','DET','PART','PRON','SCONJ','PUNCT','SPACE'}

        self.abstention_threshold = abstention_threshold
        self.absolute = absolute_threshold
        self.relative = relative_threshold
        self.gamma = gamma
        logger.debug(f'Sampling+keyword. abs: {absolute_threshold}, relative: {relative_threshold}')


    def query(self, data_item):
        question = data_item["question"]
        all_chunks = data_item["topk_content"]
        n = len(all_chunks)

        # 1) Assign geometric weights to chunks: gamma^i
        weights = np.array([self.gamma ** i for i in range(n)])
        weights /= weights.sum()  # normalize

        # 2) First-stage sampling: sample multiple subsets & query LLM
        sampled_responses = []
        total_weight = 0
        total_weight_orig = 0

        for i in range(self.num_samples):
            indices = np.random.choice(
                n,
                size=min(self.sample_size, n),
                replace=False,
                p=weights
            )
            sampled_chunks = [all_chunks[j] for j in indices]
            chunk_weight = weights[indices].sum()
            prompt = self.build_prompt(question, sampled_chunks)
            response = self.llm.query(prompt)

            if "I don't" not in response:
                sampled_responses.append((response, chunk_weight))
                total_weight += chunk_weight
                total_weight_orig += 1

        logger.debug(f"Sampled responses:\n{sampled_responses}")

        if len(sampled_responses) < self.abstention_threshold:
            logger.warning("Abstain from making response...")
            return "I don't know."

        # 3) Keyword aggregation
        token_counter = defaultdict(int)
        all_extracted_phrase = []

        for response, weight in sampled_responses:
            doc = self.keyword_extractor(response)
            phrase_list = [response.strip()]
            tmp = []

            for token in doc:
                if token.pos_ in self.ignore_set:
                    if len(tmp) > 0:
                        phrase = ''.join([x.lemma_ + x.whitespace_ for x in tmp]).strip()
                        phrase_list.append(phrase)
                        phrase_list += [x.lemma_ for x in tmp]
                        tmp = []
                else:
                    tmp.append(token)

            phrase = ''.join([x.lemma_ + x.whitespace_ for x in tmp]).strip()
            phrase_list.append(phrase)
            phrase_list += [x.lemma_ for x in tmp]
            phrase_list = set(phrase_list)

            all_extracted_phrase.append(phrase_list)
            for phrase in phrase_list:
                token_counter[phrase] += weight * total_weight_orig / total_weight

        # Filtering
        print(phrase_list)
        count_threshold = min(self.absolute, self.relative * len(sampled_responses))
        for token, count in list(token_counter.items()):
            if (count < count_threshold) or (token in punctuation) or (token in stopword_set):
                del token_counter[token]

        # Generate keyword-based final query
        sorted_tokens = sorted(token_counter.items(), key=lambda x: (len(x[0]), x[0]), reverse=True)
        hints = ', '.join([token for token, _ in sorted_tokens])
        logger.debug("Sorted tokens for hints:")
        logger.debug(sorted_tokens)
        hint_prompt = self.llm.wrap_prompt(data_item, as_multi_choice='choices' in data_item, hints=hints)
        logger.debug(f'Hint prompt:\n{hint_prompt}')
        final_response = self.llm.query(hint_prompt)

        logger.debug(f"Final response:\n{final_response}")
        return final_response

    def build_prompt(self, question, chunks):
        context_text = "\n\n".join(chunks)
        return f"""
        Given the context information below and not prior knowledge, answer the query with only keywords.
        If there is no relevant information, just say "I don't know".\n\n
        Context:\n
        {context_text}\n\n
        Query: {question}\n
        Answer:
        """
