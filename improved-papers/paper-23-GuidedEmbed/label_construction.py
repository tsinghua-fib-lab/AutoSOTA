import os
import json
import numpy as np
from sklearn.cluster import KMeans
import random
from typing import List, Dict, Any

from util import thread_response
from prompt import (
    text_summarize_prompt, 
    label_generation_prompt,
    text_classification_prompt
)
from data import load_data, generate_embeddings, load_embeddings

class TextClassificationWorkflow:
    def __init__(self, cache_dir: str = "./cache", random_seed: int = 42):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # Set random seed
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        # Create random number generator
        self.rng = np.random.RandomState(random_seed)
        
    def _load_cache(self, filename: str) -> Dict[str, Any]:
        """Load cache file"""
        filepath = os.path.join(self.cache_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _save_cache(self, data: Dict[str, Any], filename: str):
        """Save cache file"""
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def summarize_texts(self, texts: List[str], indices: List[int], instruction: str, cache_key: str) -> Dict[int, str]:
        """Summarize texts
        
        Args:
            texts: List of texts
            indices: List of original text IDs
            instruction: Summarization instruction
            cache_key: Cache key
        
        Returns:
            Dict[int, str]: Mapping from original ID to summary
        """
        cache = self._load_cache(f'summaries_{cache_key}.json')
        if cache:
            return cache['summaries']

        print("Generating text summaries...")
        prompts = [
            text_summarize_prompt.format(instruction=instruction, text=text)
            for text in texts
        ]
        
        responses = thread_response(prompts)
        
        # Save summaries using original IDs as keys
        summaries = {}
        for idx, response in zip(indices, responses):
            summary = response.split('Summary:', 1)[1].strip() if 'Summary:' in response else response.strip()
            summaries[int(idx)] = summary
        
        self._save_cache({
            'instruction': instruction,
            'summaries': summaries
        }, f'summaries_{cache_key}.json')
        
        return summaries

    def cluster_texts(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Cluster texts"""
        print(f"Clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

    def validate_classification(self, classification: str, categories: Dict[int, str]) -> bool:
        """Validate if the classification result is in valid categories
        
        Args:
            classification: Classification result to validate
            categories: Category dictionary from clustering {cluster_id: category_name}
        
        Returns:
            bool: Whether the classification result is valid
        """
        # Get all valid category names
        valid_categories = set(categories.values())
        return classification.strip() in valid_categories

    def classify_texts(
        self, 
        texts: List[str],
        indices: List[int],
        categories: Dict[int, str],
        instruction: str,
        cache_key: str = None
    ) -> Dict[int, str]:
        """Classify texts using the generated classification system
        
        Args:
            texts: List of texts
            indices: List of original text IDs
            categories: Category definition dictionary
            instruction: User instruction
            cache_key: Cache key
        
        Returns:
            Dict[int, str]: Mapping from original ID to classification result
        """
        if cache_key:
            cache = self._load_cache(f'classifications_{cache_key}.json')
            if cache:
                return cache

        print("Performing text classification...")
        categories_text = "\n".join(categories.values())
        print("Categories: ", categories_text)
        
        prompts = [
            text_classification_prompt.format(
                instruction=instruction,
                categories=categories_text,
                text=text
            ) for text in texts
        ]
        
        responses = thread_response(prompts)
        
        classifications = {}
        for idx, response in zip(indices, responses):
            classification = None
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('Classification:'):
                    classification = line.split(':', 1)[1].strip()
                    break
            
            # Only save valid classification results
            if classification and self.validate_classification(classification, categories):
                classifications[int(idx)] = classification

        if cache_key:
            self._save_cache(classifications, f'classifications_{cache_key}.json')
        
        return classifications

    def build_kmeans_clusters(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        instruction: str,
        n_clusters: int = 10,
        cache_key: str = None
    ) -> Dict[int, str]:
        """Use KMeans for clustering and generate labels"""
        if cache_key:
            cache = self._load_cache(f'kmeans_clusters_{cache_key}.json')
            if cache:
                return cache

        print(f"Using KMeans for {n_clusters} clusters...")
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Prepare prompts for all clusters
        prompts = []
        for cluster_id in range(n_clusters):
            # Get texts for current cluster
            positive_texts = [texts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            # Sample from current cluster
            if len(positive_texts) > 25:
                positive_texts = self.rng.choice(positive_texts, 25, replace=False)
            
            # Sample from other clusters
            other_indices = np.where(cluster_labels != cluster_id)[0]
            if len(other_indices) > 25:
                other_indices = self.rng.choice(other_indices, 25, replace=False)
            negative_texts = [texts[i] for i in other_indices]
            
            # Generate prompt
            prompt = label_generation_prompt.format(
                instruction=instruction,
                positive_texts="\n".join(positive_texts),
                negative_texts="\n".join(negative_texts)
            )
            prompts.append(prompt)
        
        # Get batch responses
        responses = thread_response(prompts)
        
        # Parse responses
        categories = {}
        for cluster_id, response in enumerate(responses):
            label = response.split('Category:', 1)[1].strip() if 'Category:' in response else response.strip()
            categories[cluster_id] = label
        
        if cache_key:
            self._save_cache(categories, f'kmeans_clusters_{cache_key}.json')
        
        return categories

    def run(self, instruction: str, texts: List[str], indices: List[int], n_clusters: int = 10, cache_key = "xxx"):
        """Execute label construction workflow
        
        Args:
            instruction: str - Classification instruction
            texts: List[str] - List of texts
            indices: List[int] - Original indices of texts
            n_clusters: int - Number of clusters
            cache_key: str - Cache key
        """
        
        # If text count is less than 100, use half of text count as cluster number
        if len(texts) < 100:
            n_clusters = max(1, int(len(texts) // 2))
            print(f"Texts number is less than 100, using {n_clusters} clusters")
        
        # Summarize texts
        summaries = self.summarize_texts(texts, indices, instruction, cache_key)
        summaries = {int(k): v for k, v in summaries.items()}
        
        # Get embeddings
        print("Generating text embeddings...")
        summary_texts = [summaries[idx] for idx in indices]
        summary_embeddings_path = f'./cache/embeddings_{cache_key}.pkl'
        if not os.path.exists(summary_embeddings_path):
            embeddings = generate_embeddings(texts=summary_texts, cache_path=summary_embeddings_path)
        else:
            embeddings = load_embeddings(cache_path=summary_embeddings_path)
        
        # Use KMeans clustering
        categories = self.build_kmeans_clusters(
            embeddings,
            summary_texts,
            instruction=instruction,
            n_clusters=n_clusters,
            cache_key=cache_key
        )

        # Text classification
        classifications = self.classify_texts(
            texts, 
            indices, 
            categories, 
            instruction=instruction,
            cache_key=cache_key
        )
        
        # Save complete results
        result = {
            'instruction': instruction,
            'indices': [int(i) for i in indices],
            'summaries': summaries,
            'categories': categories,
            'classifications': {str(k): v for k, v in classifications.items()},
            'clusters': {
                int(indices[i]): int(category)
                for i, category in enumerate(categories)
            }
        }
        
        self._save_cache(result, f'final_result_{cache_key}.json')
        print(f"Results saved to: {self.cache_dir}/final_result_{cache_key}.json")
        
        return result

if __name__ == "__main__":
    # Initialize workflow
    workflow = TextClassificationWorkflow()

    # Load data
    print("Loading data...")
    data_dict = load_data()
    texts = data_dict['texts']
    original_indices = data_dict['indices']

    instruction = "Specify instruction here"
    n_samples = 3000
    n_clusters = 50
    
    # Random sampling
    if n_samples > len(texts):
        n_samples = len(texts)
    selected_mask = workflow.rng.choice(len(texts), n_samples, replace=False)
    selected_texts = [texts[i] for i in selected_mask]
    selected_indices = [original_indices[i] for i in selected_mask]
    
    # Run workflow
    result = workflow.run(
        instruction=instruction,
        texts=selected_texts,
        indices=selected_indices,
        n_clusters=n_clusters
    )
