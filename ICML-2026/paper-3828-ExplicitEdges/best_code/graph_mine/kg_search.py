import json
from typing import List, Dict, Any
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.schema import QueryBundle
from inses.inses_retriever import INSESRetriever


class KGSearch:
    def __init__(
            self,
            llm,
            embed_model,
            username="neo4j",
            password="password123",
            url="bolt://localhost:7687"
    ):

        # Initialize Neo4j graph store
        self.graph_store = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=url
        )

        self.llm = llm
        self.embed_model = embed_model

        # Initialize INSESRetriever
        self.search_retriever = INSESRetriever(
            graph_store=self.graph_store,
            embed_model=self.embed_model,
            llm=self.llm,
        )

        print("INSESRetriever initialized")

    # Use LLM to evaluate if the correct answer is in the context
    def gpt_evaluate_response(self, correct_answer, context):
        system_prompt = "You are an evaluator that checks if the Correct Answer can be deduced from the information in the Context."
        user_prompt = f"""
        Context:
        {context}

        Correct Answer:
        {correct_answer}

        Task:
        Determine whether the Context contains the information stated in the Correct Answer. 
        Respond with "1" if yes, and "0" if no. Do not provide any explanation, just the number.
        """

        # fallback: merge system+user into one prompt
        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.llm.complete(prompt)
        response_text = response.text.strip()
        return int(response_text)

    def test_single_question(self, question: str, question_id: int) -> Dict[str, Any]:
        """
        Args:
            question:
            question_id:

        Returns:
            dict
        """

        query_bundle = QueryBundle(query_str=question)

        # perform graph search by calling retrieve_from_graph
        all_selected_triplets = self.search_retriever.retrieve_from_graph(query_bundle)

        # analysis result
        triplets_text = [t.node.text for t in all_selected_triplets]
        context = " ".join(triplets_text)
        evaluation = self.gpt_evaluate_response(question, context)

        test_result = {
            "question_id": question_id,
            "question": question,
            #"visited_nodes_count": len(visited_nodes),
            "all_selected_triplets_count": len(all_selected_triplets),
            #"visited_nodes": " ".join([vv.id for vv in visited_nodes]),
            "all_selected_triplets": context,
            "evaluation": evaluation,
        }

        return test_result

    def run_all_tests(
            self,
            questions: List[str],
            out_file_name: str
    ):
        """
        Args:
            questions: List[str]
            out_file_name: str
        """

        all_results = []
        correct = 0
        for i, question in enumerate(questions, 1):
            result = self.test_single_question(question, i)
            all_results.append(result)
            correct = correct + int(result['evaluation'])

        accuracy = correct / len(questions)
        all_results.append({"accuracy": f"{accuracy * 100:.2f}%"})

        # save result to file
        try:
            with open(out_file_name, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"results have been saved to {out_file_name}")
        except Exception as e:
            print(f"save results error: {e}")
            raise

        return accuracy


if __name__ == "__main__":
    pass
