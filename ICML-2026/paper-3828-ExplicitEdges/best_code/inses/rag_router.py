import json
import re
from tqdm import tqdm
from datetime import datetime
from qdrant_vectordb import QdrantVectorDB
from neo4j_graphdb import Neo4jGraphDB
from llama_index.core.llms import LLM
from data_loader import DataLoader
from evaluator import Evaluator

class RAGRouter:
    """route to different RAG systems"""

    def __init__(
            self,
            llm,
            embed_model,
            similarity_top_k: int = 5,
            dataset_name: str = "2wiki",
            result_dir: str = "../results/",
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        self.dataset_name = dataset_name
        self.result_dir = result_dir

        self.qdrant_db = QdrantVectorDB(
            llm=self.llm,
            embed_model=self.embed_model,
            collection_name=self.dataset_name,
            similarity_top_k=self.similarity_top_k,
        )
        self.neo4j_db = Neo4jGraphDB(
            llm_model=self.llm,
            embed_model=self.embed_model,
        )

    def is_multi_hop(
            self,
            query: str,
    ) -> bool:
        """
        Classify whether a question likely requires >=3 reasoning hops.
        Uses a LlamaIndex LLM object with `.complete(prompt)`.
        Returns True (>=3 hops) or False (<=2 hops or parse error).
        """
        prompt = f"""You are a strict binary classifier of question complexity for QA routing.
        Return EXACTLY one token: True or False. No punctuation. No explanation.

        Definition:
        - "True" ONLY if the question is exceptionally complex and strictly requires >= 3 distinct reasoning hops.
        - "False" if it can be answered with 1-2 hops, involves basic aggregations, or if you are unsure.
        - DEFAULT RULE: When in doubt or if the complexity is borderline, strictly output False.

        Examples:
        Q: Who wrote 'Pride and Prejudice'?
        A: False
        Q: Capital of Spain?
        A: False
        Q: What is the currency of the country whose capital is Cairo?
        A: False
        Q: What is the capital of the country where New York is located?
        A: False
        Q: Which city has a larger population, the capital of France or the capital of Italy?
        A: False
        Q: Which scientist discovered the element isolated by the spouse of the founder of X-ray crystallography?
        A: True
        Q: Find the museum that houses the painting created by the student of the artist who co-founded Cubism, and then report the city of that museum.
        A: True

        Q: {query}
        A:
        """

        try:
            resp = self.llm.complete(prompt)
            text = (getattr(resp, "text", None) or str(resp)).strip().lower()
            return text == "true" if text in {"true", "false"} else False
        except Exception:
            return False

    def run_on_dataset(
            self,
            dataset_name: str,
            sample_size: int,
            llm: LLM,
            confidence_threshold: float = 0.8,
    ):
        qa, context = DataLoader(dataset_name=dataset_name, sample_size=sample_size).load()

        result_list = []
        for item in tqdm(qa):
            question = item["question"]
            multi_hop = self.is_multi_hop(question)

            # Run both RAG and GraphRAG for all queries (dual-path confidence selection)
            rag_answer = None
            graphrag_answer = None

            try:
                rag_answer = self.qdrant_db.generate_structured_response(
                    query=question,
                    llm=llm,
                )
            except Exception as e:
                rag_answer = {"reasoning": "Exception !!!", "answer": str(e), "confidence": 0.0}
                print(f"RAG error: {e}")

            try:
                graphrag_answer = self.neo4j_db.generate_structured_response(
                    query=question,
                    llm=llm,
                )
            except Exception as e:
                graphrag_answer = {"reasoning": "Exception !!!", "answer": str(e), "confidence": 0.0}
                print(f"GraphRAG error: {e}")

            # Select answer with higher confidence
            rag_conf = float(rag_answer.get("confidence", 0.0)) if rag_answer else 0.0
            graph_conf = float(graphrag_answer.get("confidence", 0.66)) if graphrag_answer else 0.0

            if rag_conf >= graph_conf:
                result = {**item, "rag_answer": rag_answer["answer"], "confidence": rag_conf}
            else:
                result = {**item, "rag_answer": graphrag_answer["answer"], "confidence": graph_conf}
            result_list.append(result)

        file_path = self.result_dir + dataset_name + "RAGRouter.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(result_list, file, ensure_ascii=False, indent=4)
            print(f"data has been saved to {file_path}")
        except Exception as e:
            print(f"file save error: {e}")


def main(llm, embed_model, dataset_name: str="2wiki", sample_size: int=1000):
    qa, context = DataLoader(dataset_name=dataset_name, sample_size=sample_size).load()
    rag_router = RAGRouter(
        llm=llm,
        embed_model=embed_model,
        similarity_top_k=5,
        dataset_name=dataset_name,
    )

    rag_router.qdrant_db.delete_collection(confirm=True)
    rag_router.qdrant_db.add_documents(context)

    rag_router.neo4j_db.add_documents_in_batches(context)

    rag_router.run_on_dataset(dataset_name, sample_size=sample_size, llm=llm)

    eval_filename = f"{dataset_name}RAGRouter"
    em_score, em_set = Evaluator().evaluate_file_by_em(
        json_file_name=eval_filename,
        answer1_name="answer",
        answer2_name="rag_answer",
    )
    llmjudge_score, llmjudge_set = Evaluator().evaluate_file_by_llm_judge(
        json_file_name=eval_filename,
        question_name="question",
        ground_truth_name="answer",
        prediction_name="rag_answer",
        llm=llm,
    )
    return em_score, llmjudge_score


if __name__ == "__main__":
    import os
    import argparse
    from dotenv import load_dotenv

    os.environ["no_proxy"] = "localhost,127.0.0.1"

    load_dotenv()

    # 1. Define the command-line argument parser
    parser = argparse.ArgumentParser(
        description="INSES: Multi-hop Reasoning Framework via RAG Routing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", type=str, default="2wiki", choices=["2wiki", "musique", "hotpotqa"],
                        help="The dataset to evaluate.")
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Number of samples to run from the dataset.")
    parser.add_argument("--llm_provider", type=str, default="zhipuai", choices=["zhipuai", "deepseek", "openai"],
                        help="The LLM provider to serve as the cognitive engine.")
    parser.add_argument("--model", type=str, default="glm-4",
                        help="The specific LLM model version (e.g., glm-4, deepseek-chat, gpt-4o).")

    args = parser.parse_args()

    # 2. Dynamic Instantiation of LLM
    print(f"[*] Initializing LLM Engine: {args.llm_provider.upper()} ({args.model})...")

    if args.llm_provider in ["zhipuai", "deepseek"]:
        from llm_factory import LLMFactory

        # Dynamically construct environment variable names, such as ZHIPUAI_API_KEY or DEEPSEEK_API_KEY
        env_key_name = f"{args.llm_provider.upper()}_API_KEY"
        api_key = os.getenv(env_key_name)

        if not api_key:
            raise ValueError(f"Missing {env_key_name}! Please configure it in your .env file or system environment.")

        eval_llm = LLMFactory.create_llm(
            provider=args.llm_provider,
            model=args.model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=1024
        )

    elif args.llm_provider == "openai":
        from llama_index.llms.openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY! Please configure it in your .env file.")

        eval_llm = OpenAI(
            model=args.model,
            api_key=api_key,
            temperature=0.0,
        )

    # 3. Initialize the local embedding model
    print("[*] Loading Embedding Model: BAAI/bge-base-en-v1.5...")

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    # The first run will automatically download the model weights in PyTorch format (.bin or .safetensors).
    bge_base = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # 4. Execute the main evaluation process
    print(f"[*] Starting evaluation on dataset: '{args.dataset}' (Sample size: {args.sample_size})")
    em_score, llmjudge_score = main(
        llm=eval_llm,
        embed_model=bge_base,
        dataset_name=args.dataset,
        sample_size=args.sample_size
    )

    # 5. Print the formatted final result.
    print("\n" + "=" * 50)
    print(" " * 15 + "INSES EVALUATION RESULTS")
    print("=" * 50)
    print(f" Dataset        : {args.dataset}")
    print(f" Sample Size    : {args.sample_size}")
    print(f" LLM Engine     : {args.llm_provider} ({args.model})")
    print("-" * 50)
    print(f" Exact Match (EM)   : {em_score:.4f}")
    print(f" LLM-as-a-Judge     : {llmjudge_score:.4f}")
    print("=" * 50 + "\n")

