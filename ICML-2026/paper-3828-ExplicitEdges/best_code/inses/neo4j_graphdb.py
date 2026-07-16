import re
import json
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
    DynamicLLMPathExtractor,
)
from inses_retriever import INSESRetriever

from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
)

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM

from data_loader import DataLoader
from neo4j import GraphDatabase
from py2neo import Graph

DEFAULT_ENTITIES = [
    "PRODUCT",
    "MARKET",
    "TECHNOLOGY",
    "EVENT",
    "CONCEPT",
    "ORGANIZATION",
    "PERSON",
    "LOCATION",
    "TIME",
    "MISCELLANEOUS",
]

DEFAULT_RELATIONS = [
    "USED_BY",
    "USED_FOR",
    "LOCATED_IN",
    "PART_OF",
    "WORKED_ON",
    "HAS",
    "IS_A",
    "BORN_IN",
    "DIED_IN",
    "HAS_ALIAS",
]

class Neo4jGraphDB:
    def __init__(
            self,
            llm_model,
            embed_model,
            uri: str = "bolt://localhost:7687",
            username: str = "neo4j",
            password: str = "password123",
            database_name: str = "neo4j",
            docker_container_name: str = "my-neo4j",
            chunk_size: int = 1024,
            chunk_overlap: int = 20,
            result_dir: str = "../results/",
            **kwargs
    ):
        """
        Initialize Neo4j graph database

        Parameters:
            llm_model： LLM Model
            embedding_model： Embedding Model
            uri: Neo4j database URI
            username: Database username
            password: Database password
            database_name: Graph Database name
            chunk_size: Text chunk size
            chunk_overlap: Text chunk overlap size
            docker_container_name: Neo4j Docker container name
            **kwargs: Other parameters
        """
        self.llm = llm_model
        self.embed_model = embed_model
        self.uri = uri
        self.username = username
        self.password = password
        self.database_name = database_name
        self.docker_container_name = docker_container_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.result_dir = result_dir

        # Initialize graph store
        self.graph_store = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=uri,
            database=database_name,
        )

        # Initialize property graph index
        self.graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            llm=self.llm,
            embed_model=self.embed_model,
            use_async=False,
        )

        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Initialize extractors
        self._init_extractors()

        self.response_prompt = self._create_structured_response_prompt()

        # Test connection
        #self._test_connection()

    def get_graph_index(self):
        return self.graph_index

    def _init_extractors(self):
        """Initialize various extractors for extracting high-quality triples"""
        # Rule-based implicit path extractor
        self.implicit_extractor = ImplicitPathExtractor()

        # LLM-based path extractor
        self.llm_path_extractor = SimpleLLMPathExtractor(
            llm=self.llm,
            max_paths_per_chunk=20,
            num_workers=4,
        )

        # DynamicLLMPathExtractor extractor
        self.dynamic_llm_path_extractor = DynamicLLMPathExtractor(
            llm=self.llm,
            max_triplets_per_chunk=20,
            num_workers=4,
            allowed_entity_types=DEFAULT_ENTITIES,
            allowed_relation_types=DEFAULT_RELATIONS,
        )

    def add_documents(self, documents: List[Dict[str, str]], show_progress: bool = True) -> PropertyGraphIndex:
        """
        Add document list to graph database

        Parameters:
            documents: List of documents, each document is a dictionary containing 'title' and 'text'
            show_progress: show progress or not

        Returns:
            PropertyGraphIndex: Created graph index
        """

        # Create LlamaIndex document objects
        llama_docs = [
            Document(
                text=doc["text"],
                metadata={"title": doc["title"], "source": doc.get("source", "unknown")}
            ) for doc in documents
        ]

        # Parse documents into nodes
        nodes = self.node_parser.get_nodes_from_documents(llama_docs)

        # Create graph index
        self.graph_index = PropertyGraphIndex(
            nodes=nodes,  # optimized_nodes,
            property_graph_store=self.graph_store,
            llm=self.llm,
            embed_model=self.embed_model,
            extractors=[
                self.implicit_extractor,
                self.llm_path_extractor,
            ],
            use_async=False,
            show_progress=show_progress,
        )

        print(f"Successfully added {len(documents)} documents to graph database")
        return self.graph_index

    def add_documents_in_batches(self, documents: List[Dict[str, str]], bat_size: int = 1000):
        # Add documents to graph database
        print("=== Adding Documents in batches ===")

        # process documents in batches
        batch_size = bat_size  # batch size
        for i in range(0, len(documents), batch_size):
            doc_batch = documents[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")

            try:
                self.add_documents(doc_batch)
            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1} from {i} to {i+batch_size}: {e}")
                # skip this batch or save the error information
                continue

    def _create_structured_response_prompt(self) -> PromptTemplate:
        """create a custom prompt template and ensure the output is in JSON format"""

        template = """
        You are a helpful assistant that provides accurate and concise answers based on the provided knowledge graph information.

        Please answer the following query: {query_str}

        The following information is extracted from a knowledge graph, which contains entities, relationships, and relevant text:
        ---------------------
        {context_str}
        ---------------------

        Your response must be in JSON format with three fields:
        1. "reasoning": Your step-by-step reasoning process based on the knowledge graph information. Explain how the entities and relationships help answer the query.
        2. "answer": The final answer to the query, as concise as possible without unnecessary explanations.
        3. "confidence": A float between 0.0 and 1.0 indicating your confidence in the answer based on the quality and completeness of the supporting knowledge graph information.

        Example response format:
        {{
            "reasoning": "Step 1: Identified entity X and its relationship to entity Y. Step 2: Found that entity Z is connected to both X and Y. Step 3: Based on these relationships, concluded that...",
            "answer": "Concise answer here",
            "confidence": 0.85
        }}

        JSON Response:
        """
        return PromptTemplate(template)

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """extract JSON object"""

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no valid JSON is found, return the basic structure
        return {
            "reasoning": "Failed to extract JSON from response.",
            "answer": text
        }

    def generate_structured_response(
            self,
            query: str,
            llm,
            similarity_top_k: int = 5,
            response_mode: str = "compact",
            verbose: bool = False,
    ) -> Dict[str, str]:
        """
        Generates a structured JSON answer based on a query.

        Parameters:
            query: query
            llm: LLM
            similarity_top_k: top-k
            response_mode: compact/others
            verbose: whether to display verbose information

        Return:
            dict including reasoning and answer
        """
        # INSESRetriever is the implementation of INSES graph search algorithm
        inses_retriever = INSESRetriever(
            graph_store=self.graph_store,
            embed_model=self.embed_model,
            llm=llm,
        )

        # create a query engine using the as_query_engine method of PropertyGraphIndex
        query_engine = self.graph_index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode,
            text_qa_template=self.response_prompt,
            llm=llm,
            verbose=verbose,
            sub_retrievers=[inses_retriever]
        )

        # perform query
        response = query_engine.query(query)

        # response to text
        if hasattr(response, 'response'):
            response_text = response.response
        else:
            response_text = str(response)

        # parse text response with JSON
        structured_response = self._extract_json_from_text(response_text)

        # make sure the correct format is returned
        if not isinstance(structured_response,
                          dict) or "reasoning" not in structured_response or "answer" not in structured_response:
            structured_response = {
                "reasoning": "Response format was incorrect. Restructuring the response.",
                "answer": str(structured_response)
            }

        return structured_response

    def run_on_dataset(self, dataset_name: str, sample_size: int, llm):
        qa, context = DataLoader(dataset_name=dataset_name, sample_size=sample_size).load()
        print('len_qa, len_context: ', len(qa), len(context))

        result_list = []
        for item in tqdm(qa):
            question = item['question']
            try:
                graphrag_answer = self.generate_structured_response(
                    query=question,
                    llm=llm,
                    # similarity_top_k=5,
                )
            except Exception as e:
                graphrag_answer = {"reasoning": "Exception !!!", "answer": str(e)}
                print(e)
            result = {**item, "graphrag_answer": graphrag_answer['answer']}
            result_list.append(result)

        file_path = self.result_dir + dataset_name + "GraphRAG.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(result_list, file, ensure_ascii=False, indent=4)
            print(f"data has been saved to: {file_path}")
        except Exception as e:
            print(f"file save error: {e}")



    def _test_connection(self):
        """Test Neo4j connection"""
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            driver.verify_connectivity()
            driver.close()
            print("Successfully connected to Neo4j database")
        except Exception as e:
            print(f"Failed to connect to Neo4j database: {e}")
            raise

    def get_connection(self) -> Graph:
        """
        Get Neo4j database connection

        Returns:
            Graph: py2neo graph object for subsequent queries
        """
        return Graph(self.uri, auth=(self.username, self.password))

    def delete_database(self, confirm: bool = False) -> None:
        """
        Delete the entire graph database

        Parameters:
            confirm: Whether to confirm deletion, prompts user if False
        """
        if not confirm:
            response = input(
                "Are you sure you want to delete the entire graph database? This operation is irreversible. (y/n): ")
            if response.lower() != 'y':
                print("Deletion cancelled")
                return

        try:
            # Use py2neo to delete all nodes and relationships
            graph = self.get_connection()
            graph.run("MATCH (n) DETACH DELETE n")
            print("Graph database has been deleted")
        except Exception as e:
            print(f"Failed to delete graph database: {e}")
            raise

    def delete_nodes_by_label(self, label: str, confirm: bool = False) -> None:
        """
        Delete all nodes with a specific label

        Parameters:
            label: Node label
            confirm: Whether to confirm deletion, prompts user if False
        """
        if not confirm:
            response = input(
                f"Are you sure you want to delete all nodes with label '{label}'? This operation is irreversible. (y/n): ")
            if response.lower() != 'y':
                print("Deletion cancelled")
                return

        try:
            graph = self.get_connection()
            graph.run(f"MATCH (n:{label}) DETACH DELETE n")
            print(f"All nodes with label '{label}' have been deleted")
        except Exception as e:
            print(f"Failed to delete nodes: {e}")
            raise

    def execute_cypher_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute Cypher query

        Parameters:
            query: Cypher query statement
            params: Query parameters

        Returns:
            Any: Query result
        """
        try:
            graph = self.get_connection()
            result = graph.run(query, params)
            return result.data()
        except Exception as e:
            print(f"Failed to execute Cypher query: {e}")
            raise


# Usage example
if __name__ == "__main__":
    pass

