# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license

import os
import time
import argparse
from datetime import datetime
from pathlib import Path

from graph_understanding.graph_understanding import GraphDatasetUnderstanding
from graph_comparison.graph_comparison import GraphDatasetComparison
from knowledge_retrieval.knowledge_retrieval import KnowledgeRetrieval
from model_refinement.model_refinement import ModelRefinement
from model_refinement.config import design_choice_translate, design_dimensions


BENCHMARK_DATASETS = {
    "node_classification": [
        "Actor",
        "AmazonComputers",
        "AmazonPhoto",
        "CiteSeer",
        "CoauthorCS",
        "Cora",
        "Cornell",
        "DBLP",
        "PubMed",
        "Texas",
        "Wisconsin",
    ],
    "link_prediction": [
        "Actor",
        "AmazonComputers",
        "AmazonPhoto",
        "CiteSeer",
        "CoauthorCS",
        "Cora",
        "Cornell",
        "DBLP",
        "PubMed",
        "Texas",
        "Wisconsin",
    ],
    "graph_classification": [
        "TU_COX2",
        "TU_DD",
        "TU_IMDB-BINARY",
        "TU_IMDB-MULTI",
        "TU_NCI1",
        "TU_NCI109",
        "TU_PROTEINS",
        "TU_PTC_FM",
        "TU_PTC_FR",
        "TU_PTC_MM",
        "TU_PTC_MR",
    ],
}


def load_openai_client(api_key_file=None, api_key_env="OPENAI_API_KEY"):
    """Create an OpenAI client only when the LLM similarity path is requested."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the optional LLM dependencies with `pip install -e .[llm]`.") from exc

    api_key = os.environ.get(api_key_env)
    if api_key_file:
        api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(
            f"Set {api_key_env} or pass --openai_api_key_file when using --similarity_metric LLM."
        )
    return OpenAI(api_key=api_key)


def format_selected_benchmarks_message(top_benchmarks, threshold, min_top_s):
    selected_count = len(top_benchmarks)
    benchmark_label = "benchmark" if selected_count == 1 else "benchmarks"
    return (
        f"Selected {selected_count} similar {benchmark_label} "
        f"(threshold >= {threshold}, min_top_s={min_top_s}): {top_benchmarks}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset name. Default Cora.')
    parser.add_argument('--task', type=str, default='node_classification', choices=['node_classification', 'link_prediction', 'graph_classification'],
                        help='Task: node_classification, link_prediction, graph_classification. Default node_classification.')
    
    # Search strategy
    parser.add_argument('--search_strategy', type=str, default='kg_controller', choices=['kg_controller'],
                        help='Search strategy. Default kg_controller.')
    parser.add_argument('--ensembling', type=str, default='bayesian_update', choices=['bayesian_update'],
                        help='Knowledge weaving method. Default bayesian_update.')
    parser.add_argument('--initial_strategy', type=str, default='weighted_average', choices=['weighted_average', 'majority_vote', 'best'],
                        help='Initial proposal strategy: weighted_average, majority_vote, best. Fixed to best if ensembling is separated. Default weighted_average.')
    parser.add_argument('--max_iter', type=int, default=30,
                        help='Maximum number of iterations (trials) for the model design refinement. Default 30.')

    # Similarity Metric
    parser.add_argument('--similarity_threshold', type=float, default=None,
                        help='Similarity threshold to consider as prior knowledge. Default None.')
    parser.add_argument('--min_top_s', type=int, default=1,
                        help='Minimum number of similar benchmark datasets to consider as prior knowledge. Default 1.')
    parser.add_argument('--similarity_metric', type=str, default='kendall', choices=['kendall', 'LLM'],
                        help='Initial similarity metric: kendall, LLM. Default kendall.')
    parser.add_argument('--dynamic_similarity', type=str, default='bayesian_update',
                        help='Dynamic similarity metric. Default bayesian_update.')
    parser.add_argument('--use_estimator', action='store_true', default=False,
                        help='Allow GNN-based modification gain predictor to replace knowledge retrieval. Default False.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Learning rate for dynamic similarity. Default 0.1.')
    parser.add_argument('--window', type=int, default=40,
                        help='Window size for dynamic similarity. Default 40.')

    # LLM Prompting
    parser.add_argument('--top_ratio', type=str, default='best',
                        help='Top ratio of the best-performed models to consider on each benchmark. 0.05 -> 5%. Default best.')
    parser.add_argument('--openai_api_key_file', type=str, default=None,
                        help='Optional file containing an OpenAI API key for --similarity_metric LLM.')
    parser.add_argument('--openai_api_key_env', type=str, default='OPENAI_API_KEY',
                        help='Environment variable containing an OpenAI API key. Default OPENAI_API_KEY.')
    parser.add_argument('--openai_model', type=str, default='gpt-4o-2024-08-06',
                        help='OpenAI model used by the optional LLM similarity path.')

    # Candidate evaluation
    parser.add_argument('--candidate_eval', type=str, default='auto', choices=['auto', 'database', 'train'],
                        help='How to evaluate target candidate models. auto reads the DB when available and trains otherwise.')
    parser.add_argument('--candidate_output_dir', type=str, default='outputs/candidate_runs',
                        help='Directory for GraphGym candidate evaluation runs.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id for GraphGym candidate evaluation.')
    parser.add_argument('--candidate_repeat', type=int, default=3,
                        help='Number of GraphGym repeats when a target candidate must be trained.')
    parser.add_argument('--candidate_max_epoch', type=int, default=None,
                        help='Override GraphGym max_epoch for candidate evaluation.')

    # Paths
    parser.add_argument('--dataset_root', type=str, default='GraphGym/run/datasets/',
                        help='Root directory for PyG datasets and cached dataset-property JSON files.')
    parser.add_argument('--knowledge_base_root', type=str, default=None,
                        help='Optional knowledge-base root or task-specific directory. Defaults to knowledge_retrieval/knowledge_base/<task>.')
    parser.add_argument('--response_dir', type=str, default='responses',
                        help='Directory for run summaries and LLM responses.')
    
    args = parser.parse_args()

    # Assuming descriptions and other necessary data are already defined
    unseen_dataset_name = args.dataset
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    response_save_path = os.path.join(args.response_dir, f'{unseen_dataset_name}',
                                      f'{unseen_dataset_name}_{args.search_strategy}_{current_time}')
    os.makedirs(response_save_path, exist_ok=True)

    benchmark_datasets = [dataset for dataset in BENCHMARK_DATASETS[args.task] if dataset != args.dataset]
    dataset_dir = args.dataset_root

    # --------------------------------------------------------------------------------
    # Step 1: Graph Dataset Understanding
    # Process unseen dataset
    understanding_module = GraphDatasetUnderstanding(unseen_dataset_name,
                                                     task=args.task,
                                                     root_dir=dataset_dir,
                                                     no_statistics=False,
                                                     use_semantic=True,
                                                     is_bench=False)
    unseen_dataset_description = understanding_module.process()

    # Process benchmark datasets
    benchmark_dataset_descriptions = {}
    for benchmark_dataset_name in benchmark_datasets:
        benchmark_dataset_understanding = GraphDatasetUnderstanding(benchmark_dataset_name,
                                                                    task=args.task,
                                                                    root_dir=dataset_dir,
                                                                    no_statistics=False,
                                                                    use_semantic=True,
                                                                    is_bench=True)
        benchmark_dataset_descriptions[benchmark_dataset_name] = benchmark_dataset_understanding.process()

    # --------------------------------------------------------------------------------
    # Step 2: Graph Dataset Comparison
    # Initialize the KnowledgeRetrieval module
    knowledge_retrieval = KnowledgeRetrieval(
        task=args.task,
        base_db_path=args.knowledge_base_root,
        evaluation_mode=args.candidate_eval,
        graphgym_root='GraphGym',
        candidate_output_dir=args.candidate_output_dir,
        gpu_id=args.gpu_id,
        repeat=args.candidate_repeat,
        max_epoch=args.candidate_max_epoch,
    )
    knowledge_estimatior = None
    if args.use_estimator:
        from knowledge_retrieval.knowledge_estimation import KnowledgeEstimation

        knowledge_estimatior = KnowledgeEstimation(
            task=args.task,
            buffer_size=args.window,
            base_db_path=args.knowledge_base_root,
        )
    
    # Compare the unseen dataset with benchmark datasets
    dataset_comparison = GraphDatasetComparison(design_choice_translate=design_choice_translate,
                                                design_dimensions=design_dimensions[args.task],
                                                knowledge_retrieval=knowledge_retrieval,
                                                top_ratio=args.top_ratio,
                                                llm_model=args.openai_model,
                                                response_save_path=response_save_path)
    if args.similarity_metric == 'kendall':
        benchmark_similarity = dataset_comparison.calculate_kendall_rank_similarities(dataset_dir,
                                                                                      benchmark_datasets + [unseen_dataset_name],
                                                                                      unseen_dataset_name)
    elif args.similarity_metric == 'LLM':
        client = load_openai_client(args.openai_api_key_file, args.openai_api_key_env)
        similarity_response = dataset_comparison.compare_datasets(client,
                                                                unseen_dataset_description,
                                                                benchmark_dataset_descriptions)
        benchmark_similarity = dataset_comparison.to_dict(similarity_response, benchmark_datasets)
    else:
        raise ValueError('Unknown similarity metric: {}'.format(args.similarity_metric))

    # Get the names of the top-s benchmarks
    top_benchmarks, args.similarity_threshold, args.min_top_s = dataset_comparison.determine_similar_datasets(benchmark_similarity,
                                                                                                              args.similarity_threshold,
                                                                                                              args.min_top_s)
    benchmark_message = format_selected_benchmarks_message(
        top_benchmarks,
        args.similarity_threshold,
        args.min_top_s,
    )
    print(benchmark_message)
    with open(os.path.join(response_save_path, f"refinement_summary.txt"), 'a') as file:
        file.write(f"{benchmark_message}\n")

    # --------------------------------------------------------------------------------
    model_scientist = ModelRefinement(args.task, 
                                      args.min_top_s, 
                                      args.eta,
                                      args.max_iter, 
                                      knowledge_retrieval, 
                                      knowledge_estimatior,
                                      args.window)
    selected_initial_proposal, selected_final_proposal, selected_accuracy_history = model_scientist.recommend_initial_proposal_and_refine(args.search_strategy,
                                                                                                                                          args.ensembling,
                                                                                                                                          args.initial_strategy,
                                                                                                                                          args.dynamic_similarity,
                                                                                                                                          args.use_estimator,
                                                                                                                                          args.dataset,
                                                                                                                                          top_benchmarks,
                                                                                                                                          response_save_path)
    print(f"\n\nSelected Initial Proposal: {selected_initial_proposal}")
    print(f"Selected Final Proposal: {selected_final_proposal}")
    with open(os.path.join(response_save_path, f"refinement_summary.txt"), 'a') as file:
        file.write(f"\n\nSelected Initial Proposal: {selected_initial_proposal}\n")
        file.write(f"Selected Final Proposal: {selected_final_proposal}\n")
    
    # Define the checkpoints
    checkpoints = [10, 30, 50, 70, 100]

    # Iterate over the checkpoints and print/save the best-so-far accuracy
    for checkpoint in checkpoints:
        if args.max_iter >= checkpoint and len(selected_accuracy_history) >= checkpoint:
            # Get the accuracies up to the current checkpoint
            accuracies_up_to_checkpoint = selected_accuracy_history[:checkpoint]
            # Find the best-so-far accuracy
            best_so_far = max(accuracies_up_to_checkpoint, key=lambda x: x[0])
            print(f"Best-so-far accuracy at iteration {checkpoint}: {best_so_far[0]} (std: {best_so_far[1]})")
            with open(os.path.join(response_save_path, f"refinement_summary.txt"), 'a') as file:
                file.write(f"Best-so-far accuracy at iteration {checkpoint}: {best_so_far[0]} (std: {best_so_far[1]})\n")


if __name__ == "__main__":
    main()

