import dataclasses
from typing import Any, List, Sequence, Optional, Dict, Set, Tuple
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.property_graph.sub_retrievers.base import (
    BasePGRetriever,
)
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    LabelledNode,
    EntityNode,
    Relation,
    Triplet,
    KG_SOURCE_REL,
    VECTOR_SOURCE_KEY,
)
from llama_index.core.settings import Settings
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    MetadataFilters,
    #Triplet,
)
import json
import re

class INSESRetriever(BasePGRetriever):
    """
    INSES------Intelligent Navigation and Similarity-Enhanced Search
    A retriever that uses a vector store to retrieve nodes based on a query.

    Args:
        graph_store (PropertyGraphStore):
            The graph store to retrieve data from.
        include_text (bool, optional):
            Whether to include source text in the retrieved nodes. Defaults to True.
        embed_model (Optional[BaseEmbedding], optional):
            The embedding model to use. Defaults to Settings.embed_model.
        ------vector_store (Optional[BasePydanticVectorStore], optional):
            The vector store to use. Defaults to None.
            Should be supplied if the graph store does not support vector queries.
        similarity_top_k (int, optional):
            The number of top similar kg nodes to retrieve. Defaults to 4.
        path_depth (int, optional):
            The depth of the path to retrieve for each node. Defaults to 1 (i.e. a triple).
        ------similarity_score (float, optional):
            The minimum similarity score to retrieve the nodes. Defaults to None.

    """


    def __init__(
            self,
            graph_store: PropertyGraphStore,
            include_text: bool = True,
            include_properties: bool = False,
            llm: Optional[Any] = None,
            embed_model: Optional[BaseEmbedding] = None,
            max_iterations: int = 6,
            similarity_top_k: int = 5,
            path_depth: int = 1,
            limit: int = 100,
            similarity_threshold: float = 0.8,
            **kwargs: Any,
    ) -> None:
        self._llm = llm
        self._embed_model = embed_model or Settings.embed_model
        self._max_iterations = max_iterations
        self._similarity_top_k = similarity_top_k
        self._path_depth = path_depth
        self._limit = limit
        self._similarity_threshold = similarity_threshold
        self._filters = None
        self._retriever_kwargs = {}
        self._vector_store = None

        super().__init__(
            graph_store=graph_store,
            include_text=include_text,
            include_properties=include_properties,
            include_text_preamble="",  # Prefix information when converting triples into text
            **kwargs,
        )

    '''
    @staticmethod
    def _get_valid_vector_store_params() -> Set[str]:
        return {x.name for x in dataclasses.fields(VectorStoreQuery)}

    def _filter_vector_store_query_kwargs(
        self, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        valid_params = self._get_valid_vector_store_params()
        return {k: v for k, v in kwargs.items() if k in valid_params}
    '''

    def _get_vector_store_query(self, query_bundle: QueryBundle) -> VectorStoreQuery:
        if query_bundle.embedding is None:
            query_bundle.embedding = self._embed_model.get_agg_embedding_from_queries(
                query_bundle.embedding_strs
            )

        return VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
            # filters=self._filters,
            # **self._retriever_kwargs,
        )

    def _get_similar_nodes(
            self,
            node: LabelledNode,
            include_self: bool = False,
            sim_threshold: float = 0.8,
    ) -> List[LabelledNode]:
        """Get the node most similar to the current node"""
        query_bundle = QueryBundle(node.id)
        vector_store_query = self._get_vector_store_query(query_bundle)
        nodes, scores = self._graph_store.vector_query(vector_store_query)
        similar_nodes = [node for (node, score) in zip(nodes, scores) if score > sim_threshold]
        # Whether to filter out the current node itself
        if include_self:
            filtered_nodes = similar_nodes
        else:
            filtered_nodes = [n for n in similar_nodes if n.id != node.id]
        return filtered_nodes

    def _llm_select_neighbors(
            self,
            query: str,
            current_node: LabelledNode,
            neighbors: List[Tuple[LabelledNode, Relation, LabelledNode]]
    ) -> List[LabelledNode]:
        """LLM select the two most relevant nodes from their neighbors"""
        if not neighbors:
            return []

        # construct neighbor information
        neighbor_info = []
        for head, relation, tail in neighbors:
            if head.id == current_node.id:
                neighbor_info.append(f"node: {tail.id}, relation: {relation.id}")
            else:
                neighbor_info.append(f"node: {head.id}, relation: {relation.id}")

        # LLM prompt
        prompt = f"""
        Based on the following query and the current node, select the 2 most relevant nodes from the neighbor nodes to help answer the query.

        query: {query}
        current node: {current_node.id}

        neighbor nodes:
        {chr(10).join(f"{i + 1}. {info}" for i, info in enumerate(neighbor_info))}

        Please return only the selected node numbers (separated by commas), for example: 1,3
        Selection criteria: Select the nodes that are most relevant to the query and can best help answer it.
        """

        try:
            response = self._llm.complete(prompt)
            # parse response
            selected_indices = []
            try:
                # extract numbers
                import re
                numbers = re.findall(r'\d+', response.text)
                selected_indices = [int(n) - 1 for n in numbers[:2]]  # only the first two
            except:
                # if failed, select the first two
                selected_indices = [0, 1] if len(neighbors) >= 2 else [0]

            # selected neighbor nodes
            selected_neighbors = []
            for idx in selected_indices:
                if 0 <= idx < len(neighbors):
                    head, relation, tail = neighbors[idx]
                    if head.id == current_node.id:
                        selected_neighbors.append(tail)
                    else:
                        selected_neighbors.append(head)

            return selected_neighbors
        except Exception as e:
            print(f"LLM error: {e}")
            # if failed, return the first two
            return [neighbors[0][2] if neighbors[0][0].id == current_node.id else neighbors[0][0]] + \
                ([neighbors[1][2] if neighbors[1][0].id == current_node.id else neighbors[1][0]] if len(
                    neighbors) > 1 else [])

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """extract JSON objects from text"""
        # parse text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Trying to find JSON objects using regular expressions
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no valid JSON is found, return the basic structure
        return {
            "determination": "Failed to extract JSON from response.",
            "selection": 1000
        }

    def _llm_check_completeness(
            self,
            query: str,
            visited_nodes: List[LabelledNode],
            all_selected_triplets: List[Tuple[LabelledNode, Relation, LabelledNode]],
            current_nodes: List[LabelledNode],
            current_triplets: List[Tuple[LabelledNode, Relation, LabelledNode]],
    ) -> Tuple[bool, List[Tuple[LabelledNode, Relation, LabelledNode]]]:
        """Use LLM to determine whether the current information is sufficient to answer the query"""
        # visited nodes information
        visited_nodes_info = []
        for i, node in enumerate(visited_nodes):
            visited_nodes_info.append(f"{i + 1}. {node.id}")

        # visited triples information
        all_selected_triplets_info = []
        for i, triplet in enumerate(all_selected_triplets):
            head, relation, tail = triplet
            all_selected_triplets_info.append(f"{i + 1}. {head.id} -> {relation.id} -> {tail.id}")

        # current nodes information
        current_nodes_info = []
        for i, node in enumerate(current_nodes):
            current_nodes_info.append(f"{i + 1}. {node.id}")

        # current triples information
        current_triplets_info = []
        for i, triplet in enumerate(current_triplets):
            head, relation, tail = triplet
            current_triplets_info.append(f"{i + 1}. {head.id} -> {relation.id} -> {tail.id}")

        # LLM prompt
        prompt = f"""
        Your task is to provide support for complex queries and multi-hop reasoning in the knowledge graph.
        Based on the following query, the visited nodes and the selected triplets, as well as the current nodes and their adjacent triplets,
        select the triplet numbers (separated by commas) from the adjacent triplets of the current nodes that help answer the query.
        Selection criteria: Select the triplets that are most relevant to the query and most likely to help answer it.

        Then determine:
        1. Based on the visited nodes, the selected triplets, and the triplets you just selected, is this information sufficient to answer the query?
        2. If so, answer "sufficient";
        3. If not, answer "insufficient";

        Your response must be in JSON format with two fields:
        "determination": "sufficient/insufficient",
        "selection": "triplet numbers, e.g., 1, 2, 3"

        Query: {query}

        The visited nodes:
        {chr(10).join(visited_nodes_info) if visited_nodes_info else 'none'}

        The selected triplets:
        {chr(10).join(all_selected_triplets_info) if all_selected_triplets_info else 'none'}

        The current nodes:
        {chr(10).join(current_nodes_info) if current_nodes_info else 'none'}

        The adjacent triplets:
        {chr(10).join(current_triplets_info) if current_triplets_info else 'none'}
        """

        try:
            response = self._llm.complete(prompt)
            response_text = response.text.lower()

            # parse the response with json
            response_json = self._extract_json_from_text(response_text)
            is_complete = (response_json['determination'] == "sufficient")
            triplet_number = response_json['selection']  # triplet_number may be empty
            num_list = triplet_number.split(',')
            selected_indices = [int(n.strip()) - 1 for n in num_list if n.strip()]  # if n.strip() to prevent n from being empty

            # get the selected triples
            selected_triplets = [current_triplets[i] for i in selected_indices if 0 <= i < len(current_triplets)]

            return is_complete, selected_triplets
        except Exception as e:
            print(f"LLM error while checking completeness: {e}")

            return False, []

    def _llm_check_completeness_with_source_text(
            self,
            query: str,
            visited_nodes: List[LabelledNode],
            all_selected_triplets: List[Tuple[LabelledNode, Relation, LabelledNode]],
            current_nodes: List[LabelledNode],
            current_triplets: List[Tuple[LabelledNode, Relation, LabelledNode]],
    ) -> Tuple[bool, List[Tuple[LabelledNode, Relation, LabelledNode]]]:
        # visited nodes information
        visited_nodes_info = []
        for i, node in enumerate(visited_nodes):
            visited_nodes_info.append(f"{i + 1}. {node.id}")

        # all selected triples, including the source text corresponding to the triples
        all_selected_triplets_info = []
        nodes_with_score = self._get_nodes_with_score(all_selected_triplets)
        nodes_with_source_text = self.add_source_text(nodes_with_score)
        for i, node in enumerate(nodes_with_source_text):
            all_selected_triplets_info.append(f"{i + 1}. {node.node.text}")

        # current nodes information
        current_nodes_info = []
        for i, node in enumerate(current_nodes):
            current_nodes_info.append(f"{i + 1}. {node.id}")

        # current triples, including the source text corresponding to the triples
        current_triplets_info = []
        nodes_with_score = self._get_nodes_with_score(current_triplets)
        nodes_with_source_text = self.add_source_text(nodes_with_score)
        for i, node in enumerate(nodes_with_source_text):
            current_triplets_info.append(f"{i + 1}. {node.node.text}")

        # LLM prompt
        prompt = f"""
        Your task is to provide support for complex queries and multi-hop reasoning in the knowledge graph.
        Based on the following query, the visited nodes and the selected triplets, as well as the current nodes and their adjacent triplets,
        select the triplet numbers (separated by commas) from the adjacent triplets of the current nodes that help answer the query.
        Selection criteria: Select the triplets that are most relevant to the query and most likely to help answer it.

        Then determine:
        1. Based on the visited nodes, the selected triplets, and the triplets you just selected, is this information sufficient to answer the query?
        2. If so, answer "sufficient";
        3. If not, answer "insufficient";

        Your response must be in JSON format with two fields:
        "determination": "sufficient/insufficient",
        "selection": "triplet numbers, e.g., 1, 2, 3"

        Query: {query}

        The visited nodes:
        {chr(10).join(visited_nodes_info) if visited_nodes_info else 'none'}

        The selected triplets and their corresponding source text:
        {chr(10).join(all_selected_triplets_info) if all_selected_triplets_info else 'none'}

        The current nodes:
        {chr(10).join(current_nodes_info) if current_nodes_info else 'none'}

        The adjacent triplets and their corresponding source text:
        {chr(10).join(current_triplets_info) if current_triplets_info else 'none'}
        """

        try:
            response = self._llm.complete(prompt)
            response_text = response.text.lower()
            #print("LLM response：", response_text)

            # parse response with JSON
            response_json = self._extract_json_from_text(response_text)
            is_complete = (response_json['determination'] == "sufficient")
            triplet_number = response_json['selection']  # triplet_number may be empty
            num_list = triplet_number.split(',')
            selected_indices = [int(n.strip()) - 1 for n in num_list if n.strip()]  # if n.strip() to prevent n from being empty
            #print('selected_indices: ', selected_indices)

            # selected triples
            selected_triplets = [current_triplets[i] for i in selected_indices if 0 <= i < len(current_triplets)]

            return is_complete, selected_triplets
        except Exception as e:
            print(f"LLM error while checking completeness: {e}")

            return False, []

    def _extract_entity_by_gpt(self, query: str) -> list[str]:

        # LLM prompt
        prompt = f"""
        Your task is to extract several entities from the given query, 
        so they can be used to search a knowledge graph for clues relevant to answering the query.
        Return only the entities you extract, separated by commas, with no other text.

        query: {query}
        """

        try:
            response = self._llm.complete(prompt)
            response_text = response.text.lower()
            entities = [s.strip() for s in response_text.split(',')]

            return entities
        except Exception as e:
            print(f"Error extracting entity using LLM: {e}")
            raise

    def retrieve_from_graph(
            self, query_bundle: QueryBundle, limit: Optional[int] = None
    ) -> List[NodeWithScore]:  # -> (List[LabelledNode], List[Triplet]):
        """
        INSES graph search
        """
        # initialize
        visited_nodes = []
        visited_node_ids = set()
        visited_triplets = []
        all_selected_triplets = []
        iteration = 0

        # LLM extract entities from the query, and then use these entities to match the most similar entities in KG
        query = str(query_bundle)
        entity_list = self._extract_entity_by_gpt(query)

        entity_node_list = [EntityNode(name=e) for e in entity_list]
        # get the most similar nodes as the initial nodes
        initial_nodes = [self._get_similar_nodes(node, True, 0.0)[0] for node in entity_node_list]

        current_nodes = initial_nodes

        while iteration < self._max_iterations and current_nodes:

            # mark visited nodes
            visited_nodes.extend(current_nodes)
            visited_node_ids.update([current_node.id for current_node in current_nodes])

            # get the triples adjacent to current nodes
            current_triplets = self._graph_store.get_rel_map(
                current_nodes,
                depth=1,
                limit=100,
                ignore_rels=[KG_SOURCE_REL],
            )
            visited_triplets.extend(current_triplets)

            # LLM check completeness
            is_complete, selected_triplets = self._llm_check_completeness_with_source_text(
                query, visited_nodes, all_selected_triplets, current_nodes, current_triplets
            )

            all_selected_triplets.extend(selected_triplets)

            candidate_nodes = []
            for triplet in selected_triplets:
                head, relation, tail = triplet
                if head.id in visited_node_ids and tail.id in visited_node_ids:
                    continue
                elif head.id in visited_node_ids:
                    candidate_nodes.append(tail)
                else:
                    candidate_nodes.append(head)

            if is_complete:
                break

            # get similar nodes
            similar_nodes = []
            for node in current_nodes:
                sim_nodes = self._get_similar_nodes(node, False, self._similarity_threshold)
                if sim_nodes:  # sim_nodes is not empty
                    similar_nodes.append(sim_nodes[0])  # keep the most similar nodes that exceed the threshold

            # merge candidate nodes and similar nodes
            candidates = candidate_nodes + similar_nodes
            # deduplication
            unique_candidates = []
            candidate_ids = set()
            for candidate in candidates:
                if candidate.id not in candidate_ids and candidate.id not in visited_node_ids:
                    unique_candidates.append(candidate)
                    candidate_ids.add(candidate.id)

            current_nodes = unique_candidates
            iteration += 1

        return self._get_nodes_with_score(all_selected_triplets)

    async def aretrieve_from_graph(
            self, query_bundle: QueryBundle, limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Asynchronous version of INSES
        """
        # An asynchronous version can be implemented here, with similar logic to the synchronous version.
        # For simplicity, the synchronous version is used here.
        return self.retrieve_from_graph(query_bundle, limit)

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
        response = self._llm.complete(prompt)
        response_text = response.text.strip()
        return int(response_text)