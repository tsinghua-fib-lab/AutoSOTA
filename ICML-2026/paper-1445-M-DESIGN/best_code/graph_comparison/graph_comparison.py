# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license

import os
import json
from pydantic import BaseModel
from scipy.stats import kendalltau

from knowledge_retrieval.knowledge_retrieval import KnowledgeRetrieval


class EachBenchmarkDataset(BaseModel):
    benchmark_name: str
    explanation: str
    similarity_score: str

class TaskSimilarityReasoning(BaseModel):
    key_connection_between_graph_properties_and_tailored_MPNN_architecture_design: str
    similarity_to_each_benchmark: list[EachBenchmarkDataset]
    three_most_similar_benchmarks: str

class GraphDatasetComparison:
    def __init__(self, design_choice_translate, design_dimensions, knowledge_retrieval=None,
                 top_ratio=0.05, response_save_path=None,
                 llm_model='gpt-4o-2024-08-06',
                 system_prompt_template_path='graph_comparison/prompt_template/graph_comparison_system.txt', 
                 user_prompt_template_path='graph_comparison/prompt_template/graph_comparison_user.txt'):
        """
        Initializes the GraphDatasetComparison class.
        :param system_prompt_template_path: Path to the system prompt template file.
        :param user_prompt_template_path: Path to the user prompt template file.
        :param response_save_path: Path to save the similarity response.
        """
        self.system_prompt_template_path = system_prompt_template_path
        self.user_prompt_template_path = user_prompt_template_path
        if response_save_path:
            self.task_similarity_file_name = os.path.join(response_save_path, 'similarity_response.json')
        else:
            self.task_similarity_file_name = None
        self.design_choice_translate = design_choice_translate
        self.design_dimensions = design_dimensions
        self.knowledge_retrieval = knowledge_retrieval
        self.top_ratio = top_ratio
        self.similarity_calculator = None
        self.llm_model = llm_model

    def compare_datasets(self, client, unseen_dataset_description, benchmark_dataset_descriptions):
        """
        Compare the unseen dataset with benchmark datasets.
        """
        system_prompt = self.generate_task_similarity_system_prompt()
        user_prompt = self.generate_task_similarity_user_prompt(unseen_dataset_description, benchmark_dataset_descriptions)
        completion = client.beta.chat.completions.parse(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=TaskSimilarityReasoning,
            temperature=0,
            top_p=0.1
        )

        similarity_response = completion.choices[0].message.parsed

        if self.task_similarity_file_name:
            self.save_similarity_response(similarity_response)

        return similarity_response
    
    def generate_task_similarity_system_prompt(self):
        """
        Generate a system prompt for task similarity reasoning.
        """
        with open(self.system_prompt_template_path, 'r') as file:
            system_prompt = file.read()
        return system_prompt

    def generate_task_similarity_user_prompt(self, unseen_dataset_description, benchmark_dataset_descriptions):
        """
        Generate a user prompt for task similarity reasoning.
        """
        with open(self.user_prompt_template_path, 'r') as file:
            user_prompt = file.read()
        
        for i, (dataset_name, description) in enumerate(benchmark_dataset_descriptions.items(), 1):
            if self.knowledge_retrieval:
                # Check if top_ratio can be converted to number
                try:
                    threshold = float(self.top_ratio)
                    most_frequent_choices = self.knowledge_retrieval.analyze_most_frequent(dataset_name, threshold=threshold)
                    top_example = f"Most Frequent Design Choices Among the Top {threshold * 100}% best-performed models on {dataset_name}:"
                except ValueError:
                    if self.top_ratio == "best":
                        best_choice = self.knowledge_retrieval.retrieve_top_n_best(dataset_name, n=1)[0][:-2]
                        most_frequent_choices = {}
                        for key, value in zip(self.design_dimensions, best_choice):
                            most_frequent_choices[key] = (value, None)
                        top_example = f"The best-performed model on {dataset_name}:"
                    else:
                        raise ValueError(f"[Graph Comparison]: Unknown value for top_ratio: {self.top_ratio}")

                translated_most_frequent_choices = self.translate_most_frequent_choices(most_frequent_choices, self.design_choice_translate)
                translated_most_frequent_choices = f"{top_example} {translated_most_frequent_choices}"
            else:
                translated_most_frequent_choices = ""
            user_prompt += f"({i}) Benchmark {dataset_name}: {description} {translated_most_frequent_choices}\n"
        
        user_prompt += "\nHere is the description of the target dataset (unseen):\n"
        user_prompt += f"Unseen Dataset: {unseen_dataset_description}\n"

        return user_prompt
    
    def calculate_kendall_rank_similarities(self, dataset_path, dataset_names, unseen_dataset_name):
        """
        Calculate similarities between the unseen dataset and benchmarks using Kendall's Tau.

        Args:
            dataset_path (str): Base path to datasets.
            dataset_names (list): List of all dataset names including unseen dataset.
            unseen_dataset_name (str): Name of the unseen dataset.

        Returns:
            dict: Similarity scores between unseen dataset and each benchmark dataset.
        """
        # Load properties
        dataset_properties = self.load_dataset_properties(dataset_path, dataset_names)

        # Ensure all datasets have the same features
        feature_names = list(next(iter(dataset_properties.values())).keys())
        for properties in dataset_properties.values():
            if set(properties.keys()) != set(feature_names):
                raise ValueError("Datasets have mismatched properties.")

        # Rank datasets by each property
        property_ranks = self.rank_datasets_by_property(dataset_properties, feature_names)

        # Create rank vectors for each dataset
        dataset_rank_vectors = self.create_rank_vectors(property_ranks, dataset_names, feature_names)

        # Compute Kendall's Tau similarities
        similarities = self.compute_kendall_tau_similarities(dataset_rank_vectors, unseen_dataset_name)

        return similarities

    def save_similarity_response(self, similarity_response):
        """
        Save the similarity response to a file.
        """
        with open(self.task_similarity_file_name, 'w') as file:
            file.write(similarity_response.json())
    
    def load_similarity_response(self):
        """
        Load the similarity response from the file.
        """
        with open(self.task_similarity_file_name, 'r') as file:
            data = file.read()
            similarity_response = TaskSimilarityReasoning.parse_raw(data)
        return similarity_response
    
    # Function to translate the most frequent choices
    @staticmethod
    def translate_most_frequent_choices(most_frequent_choices, translate_dict):
        translated_choices = {}
        for key, (choice, _) in most_frequent_choices.items():
            translated_key = translate_dict.get(key, key)
            translated_choice = translate_dict.get(choice, choice)
            translated_choices[translated_key] = translated_choice
        return translated_choices
    
    @staticmethod
    def to_dict(similarity_response, benchmark_datasets):
        """
        Convert the similarity response to a dictionary.
        """
        # Create a dictionary mapping benchmark names to similarity scores
        benchmark_similarity = {}
        for benchmark_data in similarity_response.similarity_to_each_benchmark:
            benchmark_name = benchmark_data.benchmark_name.strip()

            if benchmark_name in benchmark_datasets:
                # Extract the similarity score and convert it to float
                benchmark_similarity[benchmark_name] = float(benchmark_data.similarity_score)
            else:
                # Handle cases where the benchmark name is not recognized
                raise ValueError(f"[Graph Comparison]: Benchmark name '{benchmark_name}' not recognized.")
        
        return benchmark_similarity
    
    @staticmethod
    def determine_similar_datasets(similarities, initial_threshold, min_top_s=1):
        """
        Determine the similar datasets.

        Args:
            similarities (dict): Similarity scores between the unseen dataset and benchmark datasets.
            initial_threshold (float): Initial threshold value for similarity scores.
            min_top_s (int): Minimum number of top similar datasets to select.

        Returns:
            selected_datasets (list): Names of the selected datasets.
            adjusted_threshold (float): Adjusted threshold value for similarity scores.
            min_top_s_adjusted (int): Adjusted minimum number of top similar datasets to select.
        """
        # Apply the initial threshold
        if initial_threshold is not None:
            selected_datasets = [name for name, score in similarities.items() if score >= initial_threshold]

            # If the number of datasets is less than min_top_s, adjust the threshold
            if len(selected_datasets) < min_top_s:
                # Sort similarities in descending order
                sorted_scores = sorted(similarities.values(), reverse=True)
                # If there are fewer datasets than min_top_s, adjust min_top_s accordingly
                min_top_s_adjusted = min(min_top_s, len(sorted_scores))
                # Reset the threshold to include at least min_top_s datasets
                adjusted_threshold = sorted_scores[min_top_s_adjusted - 1]
            else:
                adjusted_threshold = initial_threshold
                min_top_s_adjusted = min_top_s
        else:
            # Sort similarities in descending order
            sorted_scores = sorted(similarities.values(), reverse=True)
            # If there are fewer datasets than min_top_s, adjust min_top_s accordingly
            min_top_s_adjusted = min(min_top_s, len(sorted_scores))
            # Reset the threshold to include at least min_top_s datasets
            adjusted_threshold = sorted_scores[min_top_s_adjusted - 1]

        selected_datasets = [(name, score) for name, score in similarities.items() if score >= adjusted_threshold]
        selected_datasets.sort(key=lambda x: x[1], reverse=True)

        return selected_datasets, adjusted_threshold, min_top_s_adjusted
    
    @staticmethod
    def load_dataset_properties(dataset_path, dataset_names):
        """
        Load properties for a list of datasets.
        
        Args:
            dataset_path (str): Base path to datasets.
            dataset_names (list): List of dataset names.
            
        Returns:
            dict: Mapping from dataset name to properties dictionary.
        """
        dataset_properties = {}
        for name in dataset_names:
            properties_file = os.path.join(dataset_path, f'{name}/{name}_properties.json')
            if os.path.exists(properties_file):
                with open(properties_file, 'r') as file:
                    properties = json.load(file)
                    dataset_properties[name] = properties
            else:
                raise FileNotFoundError(f"Properties file not found for dataset: {name}")
        return dataset_properties
    
    @staticmethod
    def rank_datasets_by_property(dataset_properties, feature_names):
        """
        Rank datasets for each property.

        Args:
            dataset_properties (dict): Mapping from dataset name to properties dictionary.
            feature_names (list): List of feature names to consider.

        Returns:
            dict: Mapping from property name to list of (dataset_name, rank).
        """
        property_ranks = {}
        for feature in feature_names:
            # Collect values for the feature
            dataset_values = []
            for name, properties in dataset_properties.items():
                value = properties.get(feature, None)
                if value is not None:
                    dataset_values.append((name, value))
                else:
                    raise ValueError(f"Dataset {name} is missing property {feature}")
            # Sort datasets based on the property value
            # For ascending order, use reverse=False
            dataset_values.sort(key=lambda x: x[1], reverse=False)
            # Assign ranks (starting from 1)
            ranks = {}
            current_rank = 1
            for i, (name, value) in enumerate(dataset_values):
                if i > 0 and value == dataset_values[i - 1][1]:
                    # Handle ties by assigning the same rank
                    ranks[name] = ranks[dataset_values[i - 1][0]]
                else:
                    ranks[name] = current_rank
                current_rank += 1
            property_ranks[feature] = ranks
        return property_ranks
    
    @staticmethod
    def create_rank_vectors(property_ranks, dataset_names, feature_names):
        """
        Create rank vectors for each dataset.

        Args:
            property_ranks (dict): Mapping from property name to dataset ranks.
            dataset_names (list): List of dataset names.
            feature_names (list): List of feature names.

        Returns:
            dict: Mapping from dataset name to rank vector (list of ranks).
        """
        dataset_rank_vectors = {name: [] for name in dataset_names}
        for feature in feature_names:
            ranks = property_ranks[feature]
            for name in dataset_names:
                rank = ranks.get(name, None)
                if rank is None:
                    raise ValueError(f"Dataset {name} does not have a rank for property {feature}")
                dataset_rank_vectors[name].append(rank)
        return dataset_rank_vectors
    
    @staticmethod
    def compute_kendall_tau_similarities(dataset_rank_vectors, unseen_dataset_name):
        """
        Compute Kendall's Tau similarities between the unseen dataset and benchmarks.

        Args:
            dataset_rank_vectors (dict): Mapping from dataset name to rank vector.
            unseen_dataset_name (str): Name of the unseen dataset.

        Returns:
            dict: Similarity scores between unseen dataset and each benchmark dataset.
        """
        similarities = {}
        unseen_rank_vector = dataset_rank_vectors[unseen_dataset_name]
        for name, rank_vector in dataset_rank_vectors.items():
            if name == unseen_dataset_name:
                continue
            # Compute Kendall's Tau
            tau, _ = kendalltau(unseen_rank_vector, rank_vector)
            similarities[name] = tau
        return similarities

