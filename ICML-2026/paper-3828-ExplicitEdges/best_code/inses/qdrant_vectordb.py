import os
import re
import json
from tqdm import tqdm
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import List, Dict, Optional, Union, Any
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM

from data_loader import DataLoader

class QdrantVectorDB:
    def __init__(
            self,
            llm,
            embed_model,
            collection_name: str = "2wiki",
            host: str = "localhost",
            port: int = 6333,
            similarity_top_k: int = 5,
            response_mode: str = "compact",
            verbose: bool = False,
            chunk_size: int = 1024,
            result_dir: str = "../results/",
            **kwargs
    ):
        """
        Initialize the Qdrant vector database

        Parameters:
            embed_model: embedding model
            collection_name: Qdrant collection name
            host: Qdrant host address
            port: Qdrant host port
            similarity_top_k: top-k
            chunk_size: text chunk size
            result_dir: output file path
            **kwargs: other parameters
        """
        self.llm = llm
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self.response_mode = response_mode
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.result_dir = result_dir

        self.response_prompt = self._create_structured_response_prompt()

        self.client = QdrantClient(host=host, port=port, **kwargs)

        # initialize qdrant vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embed_model=self.embed_model,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25"
        )

        #self.vector_store_index = self.init_vector_store_index()

        # initialize StorageContext
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # initialize node parser
        self.node_parser = SentenceSplitter(chunk_size=chunk_size)

    def get_vector_store_index(
            self
    ) -> Union[VectorStoreIndex, None]:
        """
        Check if the Qdrant collection exists. If so, returns a queryable VectorStoreIndex.

        return:
            VectorStoreIndex: If the collection exists, returns a queryable index object.
            None: If the collection does not exist
        """
        if not self.client.collection_exists(self.collection_name):
            print(f"Warning: Collection '{self.collection_name}' is not exit!")
            return None

        # create and return index
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )

    def add_documents(
            self,
            documents: List[Dict[str, str]],
            show_progress: bool = True,
    ) -> None:
        """
        add documents to vector database

        :parameters:
            documents: list of documents，dict of 'title' and 'text'
            show_progress: show progress or not
        """
        # Create LlamaIndex Document object
        llama_docs = [
            Document(
                text=doc["text"],
                metadata={"title": doc["title"]}
            ) for doc in documents
        ]

        print('len llama_docs:', len(llama_docs))

        # documents to nodes
        nodes = self.node_parser.get_nodes_from_documents(llama_docs)

        print('len nodes:', len(nodes))

        # create index and store nodes in it
        vector_store_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=show_progress,
        )

    def export_to_file(
            self,
            file_path: str
    ) -> None:
        """
        Export Qdrant database to JSON file

        parameters:
            file_path: export path name
        """
        # check if collection is existed
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} is not existed!")

        # get all records
        all_points = []
        offset = None

        while True:
            # get data by scroll
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=1000,
                with_payload=True,
                with_vectors=True
            )

            if not batch:
                break

            all_points.extend(batch)
            offset = next_offset

        # save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_points, f, ensure_ascii=False, indent=2)

    def import_from_file(
            self,
            file_path: str
    ) -> None:
        """
        import data from JSON file

        parameters:
            file_path: import file path
        """
        # check if file is existed
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} is not existed")

        # load data
        with open(file_path, 'r', encoding='utf-8') as f:
            points = json.load(f)

        if not points:
            return

        # get vector dimension
        vector_dim = len(points[0]['vector'])

        # delete collection
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        # create a new collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )

        # upload data
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )

        # re-initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embed_model=self.embed_model
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def delete_collection(
            self,
            collection_name: Optional[str] = None,
            confirm: bool = False
    ) -> bool:
        """
        Delete a collection of Qdrant database

        parameters:
            collection_name: The name of the collection to be deleted. If None, the collection name specified during initialization is used.
            confirm: Confirm deletion. If False, the user will be prompted to confirm.

        return:
            bool: returns True if the deletion is successful, otherwise return False
        """
        # the name of the collection to be deleted
        target_collection = collection_name if collection_name else self.collection_name

        # check if the collection is existed
        if not self.client.collection_exists(target_collection):
            print(f"Warning: Collection '{target_collection}' is not existed")
            return False

        # confirm to delete
        if not confirm:
            response = input(f"Sure to delete '{target_collection}' ? This operation is irreversible. (y/n): ")
            if response.lower() != 'y':
                print("delete cancelled")
                return False

        try:
            # delete collection
            self.client.delete_collection(target_collection)
            print(f"Successfully delete the collection '{target_collection}'")

            # If the current collection is deleted, the vector store should be reinitialized
            if target_collection == self.collection_name:
                self.vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embed_model=self.embed_model
                )
                self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            return True
        except Exception as e:
            print(f"Failed to delete collection: {e}")
            return False

    def _create_structured_response_prompt(self) -> PromptTemplate:
        """Create a custom prompt template, ensuring the output is in JSON format"""
        template = """
        You are a helpful assistant that provides accurate and concise answers based on the provided context.

        Please answer the following query: {query_str}

        Context information is below.
        ---------------------
        {context_str}
        ---------------------

        Your response must be in JSON format with three fields:
        1. "reasoning": Your step-by-step reasoning process based on the context.
        2. "answer": The final answer to the query, as concise as possible without unnecessary explanations.
        3. "confidence": The confidence level of your answer, where 0 means no confidence and 1 means complete certainty. If you cannot derive a reasonable answer from the provided context, the returned confidence level should be low.

        Example response format:
        {{
            "reasoning": "Step 1: ... Step 2: ... Step 3: ...",
            "answer": "Concise answer here",
            "confidence": 0.8
        }}

        JSON Response:
        """
        return PromptTemplate(template)

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON object from the text"""

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON objects using regular expressions
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
            "answer": text,
            "confidence": 0.0
        }

    def generate_structured_response(
            self,
            query: str,
            llm: LLM,
            similarity_top_k: int = 5,
            response_mode: str = "compact",
            verbose: bool = False
    ) -> Dict[str, str]:
        """
        Generates a structured JSON answer based on the query

        parameters:
            query: query
            llm: LLM
            similarity_top_k: top-k
            response_mode:
            verbose:

        return:
            dict including reasoning and answer
        """

        # create query engine
        query_engine = self.get_vector_store_index().as_query_engine(
            llm=llm,
            text_qa_template=self.response_prompt,
            similarity_top_k=similarity_top_k,
            response_mode=response_mode,
            verbose=verbose,
        )

        # perform query
        response = query_engine.query(query)

        # response to text
        if hasattr(response, 'response'):
            response_text = response.response
        else:
            response_text = str(response)

        # extract JSON from text
        structured_response = self._extract_json_from_text(response_text)

        # Make sure the correct format is returned
        if (
                not isinstance(structured_response, dict)
                or "reasoning" not in structured_response
                or "answer" not in structured_response
                or "confidence" not in structured_response
        ):
            structured_response = {
                "reasoning": "Response format was incorrect. Restructuring the response.",
                "answer": str(structured_response),
                "confidence": 0.0
            }

        return structured_response

    def init_vector_database(self):
        # Add connection check before creating QdrantVectorDB
        try:
            test_client = QdrantClient(host="localhost", port=6333)
            # Try to get the server status
            status = test_client.get_collections()
            print("Successfully connected to the Qdrant server")
            test_client.close()
        except Exception as e:
            print(f"Unable to connect to Qdrant server: {e}")
            print("Please make sure the Qdrant server is running and the connection parameters are correct")
            exit(1)

        self.delete_collection()

        print('vector db created.')

        #self.add_documents(context)
        print('doc added.')

    def run_on_dataset(self, dataset_name: str, sample_size: int, llm: LLM):
        qa, context = DataLoader(dataset_name=dataset_name, sample_size=sample_size).load()

        result_list = []
        for item in tqdm(qa):
            question = item['question']
            try:
                rag_answer = self.generate_structured_response(
                    query=question,
                    llm=llm,
                    similarity_top_k=self.similarity_top_k,
                )
            except Exception as e:
                rag_answer = {"reasoning": "Exception !!!", "answer": str(e), "confidence": 0.0}
                print(e)
            result = {**item, "rag_answer": rag_answer['answer'], "confidence": rag_answer['confidence']}
            result_list.append(result)

        file_path = self.result_dir + dataset_name + "k=" + str(self.similarity_top_k) + "RAG.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(result_list, file, ensure_ascii=False, indent=4)
            print(f"data has been save to {file_path}")
        except Exception as e:
            print(f"file save error: {e}")




if __name__ == "__main__":
    pass
