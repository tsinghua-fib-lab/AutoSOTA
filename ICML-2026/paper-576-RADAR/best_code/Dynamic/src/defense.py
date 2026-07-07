import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
stopword_set = set(stopwords.words('english'))
import torch 
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import os
import json
from .helper import clean_str, StopOnTokens
from src.decoding_methods import secure_decoding
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logger = logging.getLogger('RRAG-main')
INJECTION = True  # injection attack. if False, we consider passage modification attacks discussed in the appendix


def save_all_responses(save_path, response_list, data_item):
    all_data = []
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        with open(save_path, 'r') as f:
            all_data = json.load(f)
    all_data.append({"query": data_item['question'],
                     "answer": data_item['answer'],
                     "response": response_list})
    with open(save_path, 'w') as f:
        json.dump(all_data, f, indent=4)


# 基类
class RRAG:
    def __init__(self, llm):
        self.llm = llm

    def query_undefended(self, data_item):
        query_prompt = self.llm.wrap_prompt(data_item, as_multi_choice='choices' in data_item)
        # response = None
        response = self.llm.query(query_prompt)
        logger.debug(f'Query_prompt:\n{query_prompt}')
        logger.debug(f'Response:\n{response}')
        logger.debug(f'Answer:\n{data_item["answer"]}')
        return response

    def query(self, data_item):
        raise NotImplementedError

    def _eval_response(self, response, data_item):
        answer = data_item['answer']
        response = clean_str(response)
        for ans in answer:
            if clean_str(ans) in response:
                return True
        return False


class MinCutRRAG(RRAG):
    def __init__(self, llm, nli_model_path="DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
        super().__init__(llm)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_path).to(self.device)
        self.nli_model.eval()
        self.nli_batch_size = 32  # 默认 batch_size
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

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
        for i in range(len(responses)):
            G.add_edge(source, i, capacity=S[i])

        # Add edges from documents to sink (F_i values) only for valid docs
        for i in range(len(responses)):
            G.add_edge(i, sink, capacity=F[i])

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
                isolation_threshold = 0.3
                new_selected = [selected_docs[idx] for idx in range(n_selected) if doc_avg_cos[idx] >= isolation_threshold]
                if new_selected != selected_docs:
                    selected_docs = sorted(new_selected)
                    logger.info(f"[MinCut] After excluding isolated: {selected_docs}")

        selected_data_item = data_item.copy()
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
                    M[i, j] = 1.0  # 自己和自己完全一致
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
            batch_size = 16
            nli_probs_map = {}
            for start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[start:start + batch_size]
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
                    if contra_score > 0.8:
                        C[i, j] = C[j, i] = contra_score
                        M[i, j] = M[j, i] = 0.0  # 矛盾则不可能相似
                    else:
                        entail_score = np.sqrt(p_ij[0] * p_ji[0])
                        M[i, j] = M[j, i] = entail_score
                        C[i, j] = C[j, i] = contra_score

        return M, C

    def compute_scores_balanced(self, M, C, valid_docs, total_docs):
        k = len(valid_docs)
        if k == 1:
            return np.array([0.9]), np.array([0.1])

        # 1. 计算中心度
        adj = M + np.eye(k) * 0.01
        v = np.ones(k) / k
        for _ in range(10):
            v = np.dot(adj, v)
            v /= (np.linalg.norm(v) + 1e-8)
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
                if i == j:
                    continue
                # 对方的 centrality 代表了对方在共识中的"话语权"
                # 如果我跟一个"话语权"很高（处于共识中心）的人冲突，我的 F 应该很高
                weighted_conflict += C[i, j] * centrality[j]

            # 归一化冲突得分
            F_raw[i] = weighted_conflict / (np.sum(centrality) - centrality[i] + 1e-8)

        def final_scale(arr):
            if arr.ptp() < 1e-12:
                return np.ones_like(arr) * 0.5
            return (arr - arr.min()) / (arr.ptp() + 1e-12)

        S = final_scale(S_raw)
        F = final_scale(F_raw)

        S = np.clip(S, 0.01, 0.99)
        F = np.clip(F, 0.01, 0.99)

        return S, F


class DynamicMinCutRRAG(MinCutRRAG):
    def __init__(self, llm, nli_model_path="DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
        super().__init__(llm, nli_model_path)

    def _bayesian_update(self, prior, likelihood):
        prior = np.clip(prior, 0.01, 0.99)
        numerator = likelihood * prior
        denominator = numerator + (1 - likelihood) * (1 - prior)
        return numerator / (denominator + 1e-9)

    def dynamic_query(self, data_item, previous_answer=None, previous_priors=None):
        docs = data_item['topk_content']
        question = data_item['question']
        k = len(docs)

        if previous_priors is None:
            previous_priors = {'S': 1.0, 'F': 0.0}

        # --- Step 1: 生成单文档答案，并过滤无效文档 ---
        responses = []
        doc_sources = []
        valid_docs = []

        # 1.1 处理新文档
        for i in range(k):
            single_data_item = data_item.copy()
            single_data_item['topk_content'] = [docs[i]]
            # 调用 LLM 生成单个文档的答案
            single_prompt = self.llm.wrap_prompt(single_data_item, as_multi_choice='choices' in data_item, seperate=False)
            resp = self.llm._query(single_prompt)

            if "I don't know" in resp:
                logger.info(f"[DynamicMinCut] Skipping new document {i} as it contains 'I don't know'.")
                continue

            responses.append(resp)
            doc_sources.append(docs[i])
            valid_docs.append(i)

        # 1.2 处理旧答案 (如果存在)
        old_answer_idx = -1
        if previous_answer and "I don't know" not in previous_answer:
            logger.info(f"[DynamicMinCut] Adding previous answer to the pool.")
            responses.append(previous_answer)
            doc_sources.append(f"Previous reliable conclusion: {previous_answer}")
            old_answer_idx = len(responses) - 1
            valid_docs.append(k)

        logger.info(f"[DynamicMinCut] Valid responses: {responses}")
        logging.getLogger().handlers[0].flush()

        # 如果没有任何有效响应（既没有有效新文档，也没有旧答案）
        if not responses:
            logger.info("[DynamicMinCut] No valid info found. Querying LLM directly.")
            final_prompt = self.llm.wrap_prompt(data_item, as_multi_choice='choices' in data_item, seperate=False)
            final_answer = self.llm._query(final_prompt)
            if previous_priors is None:
                previous_priors = {'S': 0.5, 'F': 0.5}
            return final_answer, previous_priors

        # --- Step 2: 使用 NLI 计算两两相似度 (M矩阵) ---
        M, C = self._build_sim_and_conflict_matrices(question, responses)
        logger.info(f"[DynamicMinCut] M: {M.tolist()}")

        # --- Step 3: 计算 S 和 F，并构建最小割图 ---
        # 3.1 计算原始 S 和 F
        S_cons = M.sum(axis=1) / len(responses)
        F_cons = C.sum(axis=1) / len(responses)

        k_valid = len(responses)
        total_docs = len(data_item['topk_content']) + (1 if previous_answer else 0)
        S, F = self.compute_scores_balanced(M, C, valid_docs=valid_docs, total_docs=total_docs)
        logger.info(f"[DynamicMinCut] new S: {S.tolist()}")
        logger.info(f"[DynamicMinCut] new F: {F.tolist()}")

        if old_answer_idx != -1:
            # 1. 获取先验 (Prior) P(H)
            prior_S = previous_priors.get('S', 1.0)
            prior_F = previous_priors.get('F', 0.0)

            # 2. 计算似然
            other_indices = [i for i in range(len(responses)) if i != old_answer_idx]
            if other_indices:
                # 似然 S：新文档平均有多支持旧答案？
                likelihood_S = np.mean(M[old_answer_idx, other_indices])
                # 似然 F：新文档平均有多反驳旧答案？
                likelihood_F = np.mean(C[old_answer_idx, other_indices])
            else:
                likelihood_S = 1.0
                likelihood_F = 0.0

            logger.info(f"[DynamicBayes] Prior S: {prior_S:.2f}, Likelihood S: {likelihood_S:.2f}")

            # 3. 计算后验
            posterior_S = self._bayesian_update(prior_S, likelihood_S)
            posterior_F = self._bayesian_update(prior_F, likelihood_F)

            # 4. 更新图的权重
            S[old_answer_idx] = posterior_S
            F[old_answer_idx] = posterior_F

            logger.info(f"[DynamicBayes] Posterior S: {posterior_S:.2f}, Posterior F: {posterior_F:.2f}")

        # 3.3 建图
        G = nx.DiGraph()
        source = "source"
        sink = "sink"
        G.add_node(source)
        G.add_node(sink)

        num_nodes = len(responses)
        for i in range(num_nodes):
            G.add_edge(source, i, capacity=S[i])
            G.add_edge(i, sink, capacity=F[i])
            for j in range(i + 1, num_nodes):
                G.add_edge(i, j, capacity=M[i, j])
                G.add_edge(j, i, capacity=M[i, j])

        # --- Step 4: 计算最小割，选出正确答案集合 ---
        min_cut_value, partition = nx.minimum_cut(G, source, sink)
        reachable, non_reachable = partition
        selected_indices = sorted([node for node in reachable if isinstance(node, int)])
        logger.info(f"[DynamicMinCut] Selected indices (before post-process): {selected_indices}")

        # --- Step 4.5: 后处理 —— 基于 cosine 相似度排除孤立节点 ---
        # 与静态版本逻辑一致：不保护旧答案，让它和新文档一样接受孤立点检验
        if len(selected_indices) > 1:
            selected_responses = [responses[i] for i in selected_indices]
            selected_embeddings = self.embed_model.encode(selected_responses)
            selected_cos_sim = util.cos_sim(selected_embeddings, selected_embeddings).cpu().numpy()

            n_selected = len(selected_indices)
            off_diag_mask = ~np.eye(n_selected, dtype=bool)
            avg_cosine = np.mean(selected_cos_sim[off_diag_mask])
            logger.info(f"[DynamicMinCut] Average cosine of selected: {avg_cosine}")

            if avg_cosine < 1.0:
                # 计算每个被选中节点对其他被选中节点的平均 cosine（排除自己）
                doc_avg_cos = []
                for idx in range(n_selected):
                    others = [j for j in range(n_selected) if j != idx]
                    avg = np.mean(selected_cos_sim[idx, others])
                    doc_avg_cos.append(avg)

                # 剔除平均 cosine < 0.3 的孤立节点
                isolation_threshold = 0.3
                new_selected = [selected_indices[idx] for idx in range(n_selected)
                                if doc_avg_cos[idx] >= isolation_threshold]
                if new_selected != selected_indices:
                    selected_indices = sorted(new_selected)
                    logger.info(f"[DynamicMinCut] After excluding isolated: {selected_indices}")

        logger.info(f"[DynamicMinCut] Selected indices (final): {selected_indices}")
        if old_answer_idx != -1:
            is_old_kept = old_answer_idx in selected_indices
            logger.info(f"[DynamicMinCut] Previous answer kept? {is_old_kept}")

        # --- Step 5: 最终生成 ---
        selected_contents = [responses[i] for i in selected_indices]

        if not selected_contents:
            logger.warning("[DynamicMinCut] Min-cut / post-process removed all nodes! Fallback to using all available info.")
            selected_contents = doc_sources

        context_str = ""
        for idx, content in enumerate(selected_contents):
            context_str += f"Answer {idx + 1}: {content}\n"

        if len(selected_contents) >= 3:
            final_prompt = (
                f"Question: {question}\n\n"
                f"The following are all the reference answers obtained (synthesize **strictly based on this content only**. "
                f"Do NOT add, correct, question, or challenge any information in it, even if you believe it may be outdated):\n"
                f"{context_str}\n\n"
                f"Strictly follow the reference answers provided above and synthesize the most consistent and main conclusion as the final answer.\n"
                f"Output **only the final answer itself**. Do NOT write any explanations, reminders, supplements, or comments about dates."
            )
            logger.info(f"[DynamicMinCut] Using STRICT prompt (>=3 references).")
        else:
            final_prompt = (
                f"Question: {question}\n\n"
                f"The following are reference answers retrieved from external sources. "
                f"Note: the number of references is limited, so they may be incomplete, biased, or potentially unreliable. "
                f"Use them as a reference, but you may also rely on your own internal knowledge to judge and answer:\n"
                f"{context_str}\n\n"
                f"Carefully consider the reference answers above together with your own knowledge. "
                f"If the references are consistent and reasonable, follow them. "
                f"If they appear unreliable, incomplete, or contradict well-established facts you know, "
                f"you may correct or supplement them based on your own knowledge to produce the most accurate answer.\n"
                f"Output **only the final answer itself**. Do NOT write any explanations, reminders, supplements, or comments about dates."
            )
            logger.info(f"[DynamicMinCut] Using LENIENT prompt ({len(selected_contents)} reference(s) only).")

        logger.info(f"[DynamicMinCut] Final Prompt: {final_prompt}")
        final_answer = self.llm._query(final_prompt)

        new_priors = {'S': 1.0, 'F': 0.0}
        current_responses = responses
        if current_responses:
            prior_list = current_responses + [final_answer]
            M_prior, C_prior = self._build_sim_and_conflict_matrices(question, prior_list)
            final_idx = len(prior_list) - 1
            avg_S = np.mean(M_prior[final_idx, :len(current_responses)])
            avg_F = np.mean(C_prior[final_idx, :len(current_responses)])
            new_priors['S'] = avg_S
            new_priors['F'] = avg_F
            logger.info(f"[DynamicPriors] New priors based on final_answer: S={avg_S:.3f}, F={avg_F:.3f}")

        return final_answer, new_priors