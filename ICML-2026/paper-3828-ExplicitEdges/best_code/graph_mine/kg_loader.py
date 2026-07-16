import json
from typing import List, Dict, Tuple, Any
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation
from kg_search import KGSearch


class KGLoader:
    def __init__(
            self,
            embed_model,
            username="neo4j",
            password="password123",
            url="bolt://localhost:7687"
    ):

        self.graph_store = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=url
        )

        self.embed_model = embed_model

        print("Neo4j connection initialized")

    def load_json_data(self, json_file_path: str) -> Dict[str, Any]:
        """
        load JSON file

        Args:
            json_file_path: JSON file path

        Returns:
            dict including entities and relations
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"successful load JSON file: {json_file_path}")
            return data
        except Exception as e:
            print(f"failed to load JSON file: {e}")
            raise

    def load_query_data(self, json_file_path: str) -> Tuple[List, float]:

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"successful load JSON file: {json_file_path}")
        except Exception as e:
            print(f"failed to load JSON file: {e}")
            raise
        ques = [dd['correct_answer'] for dd in data[:-1]]
        kggen_accuracy = float(data[-1]['accuracy'].strip('%')) / 100
        return ques, kggen_accuracy

    @staticmethod
    def compare_accuracy(
            pathname: str = "./kggen/",
            filename_suffix1: str = "_results.json",
            filename_suffix2: str = "_hgraph.json",
            out_filename: str = "./kggen/accuracy_compare.json"
    ) -> Tuple[List, List]:
        ii = 0
        t1 = 0.0
        t2 = 0.0
        acc_list1 = []
        acc_list2 = []
        for i in range(1, 108):
            acc1_filename = f"{pathname}{i}{filename_suffix1}"
            acc2_filename = f"{pathname}{i}{filename_suffix2}"

            try:
                with open(acc1_filename, 'r', encoding='utf-8') as f1:
                    data1 = json.load(f1)
                with open(acc2_filename, 'r', encoding='utf-8') as f2:
                    data2 = json.load(f2)
            except Exception as e:
                print(f"{acc1_filename} or {acc2_filename} failed to open file: {e}")
                continue

            acc1 = float(data1[-1]['accuracy'].strip('%')) / 100
            acc2 = float(data2[-1]['accuracy'].strip('%')) / 100

            acc_list1.append(acc1)
            acc_list2.append(acc2)

            ii = ii + 1
            t1 = t1 + acc1
            t2 = t2 + acc2
        avg1 = t1 / ii
        avg2 = t2 / ii
        print(f"iteration number={ii}--avg_kg, avg_hgraph: {avg1}, {avg2}")
        # Save results to file
        # Combine them into a dictionary (with key names)
        data = {
            "acc_list1": acc_list1,
            "acc_list2": acc_list2
        }
        try:
            with open(out_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"results have been save to {out_filename}")
        except Exception as e:
            print(f"save file error: {e}")
            raise

        return acc_list1, acc_list2

    def create_entity_nodes(
            self,
            entities: List[str]
    ) -> tuple[List[EntityNode], Dict[str, str]]:

        entity_nodes = []
        name_to_id = {}

        for i, entity_name in enumerate(entities):
            try:
                # use entity name as ID
                name_to_id[entity_name] = entity_name

                # embedding each entity
                embedding = self.embed_model.get_agg_embedding_from_queries([entity_name])

                # create EntityNode object
                entity_node = EntityNode(
                    name=entity_name,
                    embedding=embedding
                )
                entity_nodes.append(entity_node)

            except Exception as e:
                print(f"create entity '{entity_name}' error: {e}")
                continue

        print(f"successful create {len(entity_nodes)} EntityNode")
        return entity_nodes, name_to_id

    def create_relation_objects(
            self,
            relations: List[List[str]],
            name_to_id: Dict[str, str]
    ) -> List[Relation]:

        relation_objects = []

        for i, relation_data in enumerate(relations):
            try:
                if len(relation_data) != 3:
                    print(f"skip malformed relationship {i}: {relation_data}")
                    continue

                source_name, relation_type, target_name = relation_data

                # Check if the source and target entities exist
                if source_name not in name_to_id:
                    print(f"source entity '{source_name}' is not existed, skip the relation {i}")
                    continue

                if target_name not in name_to_id:
                    print(f"target entity '{target_name}' is not existed, skip the relation {i}")
                    continue

                # Create a Relation object
                relation = Relation(
                    label=relation_type,
                    source_id=name_to_id[source_name],
                    target_id=name_to_id[target_name]
                )
                relation_objects.append(relation)

            except Exception as e:
                print(f"create relation {i} error: {e}")
                continue

        print(f"successful create {len(relation_objects)} Relation")
        return relation_objects

    def insert_to_neo4j(self, entity_nodes: List[EntityNode], relation_objects: List[Relation]):
        """
        Insert EntityNode and Relation objects into Neo4j

        Args:
            entity_nodes: list of EntityNode
            relation_objects: list of Relation
        """
        try:
            # insert entity nodes
            self.graph_store.upsert_nodes(entity_nodes)
            print(f"successful insert {len(entity_nodes)} node")

            # insert relations
            self.graph_store.upsert_relations(relation_objects)
            print(f"successful inset {len(relation_objects)} relation")

        except Exception as e:
            print(f"insert failed: {e}")
            raise e

    def verify_import(self):
        """
        verify import data
        """
        try:
            # Querying data using the Neo4j client
            with self.graph_store.client.session() as session:
                # get nodes number
                result = session.run("MATCH (n) RETURN count(n) as count")
                record = result.single()
                if record:
                    node_count = record["count"]
                    print(f"Number of nodes: {node_count}")
                else:
                    print("Number of nodes: 0")

                # get relations number
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = result.single()
                if record:
                    rel_count = record["count"]
                    print(f"Number of relations: {rel_count}")
                else:
                    print("Number of relations: 0")

                # Examine some sample data
                result = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN a.name as source, type(r) as type, b.name as target
                    LIMIT 5
                """)
                print("relation example:")
                for record in result:
                    print(f"  {record['source']} -[{record['type']}]-> {record['target']}")

        except Exception as e:
            print(f"verify import data failed: {e}")

    def delete_graphdb(self):
        """
        delete all entity nodes and relations
        """
        try:
            query = "match n=() detach delete n"
            self.graph_store.structured_query(query)
            print(f"successful delete")

        except Exception as e:
            print(f"failed to delete: {e}")
            raise e

    def import_json_to_neo4j(
            self,
            json_file_path: str
    ):
        """
        Args:
            json_file_path: JSON file path
        """
        print(f"Start importing JSON data into Neo4j...{json_file_path}")

        # 1. load JSON file
        data = self.load_json_data(json_file_path)

        # 2. create EntityNodes
        entities = data.get('entities', [])
        entity_nodes, name_to_id = self.create_entity_nodes(entities)

        # 3. create Relations
        relations = data.get('relations', [])
        relation_objects = self.create_relation_objects(relations, name_to_id)

        # 4. Insert into Neo4j
        self.insert_to_neo4j(entity_nodes, relation_objects)

        # 5. Verify...
        # self.verify_import()

        print(f"{json_file_path}--Importing JSON data into Neo4j is complete!")


def evaluate_score(llm, embed_model, field_query: str = "question", field_context: str = "all_selected_triplets",
                   field_evaluation: str = "evaluation"):
    # Score the retrieved triplets

    tt = 0.0
    ii = 0
    for i in range(1, 107):
        results_file_name = f"./kggen/{i}_results.json"
        out_file_name = f"./kggen/{i}_results_new.json"

        try:
            with open(results_file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"successfule load JSON file: {results_file_name}")
        except Exception as e:
            print(f"load JSON file error: {e}")
            continue

        tester = KGSearch(llm, embed_model)

        # Score all questions and search results for a single file
        all_results = []
        correct = 0
        for dd in data[:-1]:
            ques = dd[field_query]
            context = dd[field_context]
            # context = dd["visited_nodes"] + dd[field_context]

            evaluation = tester.search_retriever.gpt_evaluate_response(ques, context)
            dd[field_evaluation] = evaluation

            all_results.append(dd)
            correct = correct + int(dd['evaluation'])

        accuracy = correct / (len(data) - 1)
        all_results.append({"accuracy": f"{accuracy * 100:.2f}%"})

        # save results to file
        try:
            with open(out_file_name, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"results have been saved to {out_file_name}")
        except Exception as e:
            print(f"save results error: {e}")
            raise

        print(f"{i}--accuracy: {accuracy}")

        ii = ii + 1
        tt = tt + accuracy
    avg = tt / ii
    print(f"iteration number={ii}--avg_accuracy: {avg}")

def count_node_edge(pathname: str="./kggen/"):

    stat_result = {}
    ii = 0
    total_node_num = 0
    total_edge_num = 0
    for i in range(1, 108):
        filename = f"{pathname}{i}.json"
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"successfule load JSON file: {filename}")
        except Exception as e:
            print(f"{filename} open error: {e}")
            continue

        entities = data.get('entities', [])

        relations = data.get('relations', [])

        total_node_num = total_node_num + len(entities)
        total_edge_num = total_edge_num + len(relations)
        ii = ii + 1

        stat_result[i] = (len(entities), len(relations))
    avg_node_num = total_node_num/ii
    avg_edge_num = total_edge_num/ii
    print(f"avg_node_num: {avg_node_num}, avg_edge_num: {avg_edge_num}")
    out_filename = f"{pathname}statistics.json"
    try:
        with open(out_filename, 'w', encoding='utf-8') as file:
            json.dump(stat_result, file, ensure_ascii=False, indent=4)
        print(f"data has been saved to {out_filename}")
    except Exception as e:
        print(f"save file error: {e}")


def main(
        llm,
        embed_model,
        file_count: int =108,
        path_name: str = "./kggen/",
        input_filename_suffix: str = "_results.json",
        output_filename_suffix: str = "_hgraph.json",
):

    kgloader = KGLoader(embed_model)

    kggen_accuracy_list = []
    accuracy_list = []
    kgloader.delete_graphdb()
    for i in range(1, file_count):
        query_filename = f"{path_name}{i}{input_filename_suffix}"
        filename = f"{path_name}{i}.json"
        out_filename = f"{path_name}{i}{output_filename_suffix}"
        try:
            ques, kggen_accuracy = kgloader.load_query_data(query_filename)
            kgloader.import_json_to_neo4j(filename)
        except Exception as e:
            print(f"{query_filename} or {filename} open error: {e}")
            continue

        tester = KGSearch(llm, embed_model)

        accuracy = tester.run_all_tests(ques, out_filename)
        kggen_accuracy_list.append(kggen_accuracy)
        accuracy_list.append(accuracy)

        # delete graph database in Neo4j, next iteration
        kgloader.delete_graphdb()

    accuracy_result = {"baseline": kggen_accuracy_list, "inses": accuracy_list}
    accuracy_filename = f"{path_name}accuracy_compare.json"
    try:
        with open(accuracy_filename, 'w', encoding='utf-8') as file:
            json.dump(accuracy_result, file, ensure_ascii=False, indent=4)
        print(f"data has been saved to {accuracy_filename}")
    except Exception as e:
        print(f"save file error: {e}")



if __name__ == "__main__":

    from inses.llm_factory import LLMFactory

    #'''
    # ZhipuAI LLM instance
    zhipuai_llm = LLMFactory.create_llm(
        provider="zhipuai",
        model="glm-4",
        api_key="your key",
        temperature=0.0,
        max_tokens=1024
    )
    #'''

    '''
    # DeepSeek LLM instance
    deepseek_llm = LLMFactory.create_llm(
        provider="deepseek",
        model="deepseek-chat",  # 可用模型如 "deepseek-chat"，"deepseek-reasoner"
        api_key="your key",
        temperature=0.0,
        max_tokens=1024
    )
    '''

    #'''
    # Embedding model------bge-base-en-v1.5
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    bge_base = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")  # "BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5"
    #'''

    main(
        llm=zhipuai_llm,
        embed_model=bge_base,
        file_count=108,
        path_name="./kggen/",
        input_filename_suffix="_results.json",
        output_filename_suffix="_hgraph.json",
    )
