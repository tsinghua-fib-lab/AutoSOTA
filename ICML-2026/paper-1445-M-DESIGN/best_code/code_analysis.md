# Code Analysis for Paper 1445 (M-DESIGN)

## Evaluation Path
- Entry: `main.py` -> parse args -> Step 1: GraphDatasetUnderstanding -> Step 2: GraphDatasetComparison -> Step 3: KnowledgeRetrieval -> Step 4: ModelRefinement
- Evaluation mode: `--candidate_eval database` (reads from SQLite DB) vs `train` (actual GraphGym training)
- DB evaluation: `knowledge_retrieval/knowledge_retrieval.py:KnowledgeRetrieval.evaluate_model()` -> `retrieve_model()` queries SQLite
- Train evaluation: `knowledge_retrieval/knowledge_retrieval.py:KnowledgeRetrieval.evaluate_model()` -> `candidate_evaluator.evaluate()` -> runs `GraphGym/run/main_pyg.py`
- Final output: `main.py` prints "Final transfer for Cora: XX.XX +- Y.YY" and checkpoint summaries

## Key Files
- `main.py`: Entry point, arg parsing, orchestration
- `model_refinement/kg_controller.py`: Core Bayesian optimization loop, similarity updates, acquisition
- `knowledge_retrieval/knowledge_retrieval.py`: Knowledge base queries + evaluation dispatch
- `knowledge_retrieval/candidate_evaluator.py`: GraphGym candidate training wrapper
- `knowledge_retrieval/knowledge_estimation.py`: ECC GNN predictor management + feedback fine-tuning
- `knowledge_retrieval/gnn_predictor.py`: EdgeConv-based GNN predictor for modification-gain estimation
- `model_refinement/config.py`: Design space definitions + choice translations

## Design Space (Node Classification)
6 dimensions: neigh, norm, agg, comb, l_mp, stage
- neigh: edge_index, edge_index_knn, edge_index_2hop, edge_index_knn_rwpe, edge_index_knn_lepe
- norm: degree_sys, degree_row, fagcn_like, rel_rwpe, rel_lepe
- agg: add, mean, max, min
- comb: concat (only option)
- l_mp: 4, 6
- stage: skipconcat, skipsum, ppr_01, gpr, lstm, node_adaptive

Total: 5 x 5 x 4 x 1 x 2 x 6 = 1200 possible architectures

## Optimal Architecture (Database Mode)
neigh=edge_index, norm=rel_lepe, agg=mean, comb=concat, l_mp=6, stage=ppr_01 -> 88.50%

## Metric Parsing
- Format: "Final transfer for Cora: XX.XX +- Y.YY"
- Also: "Best-so-far accuracy at iteration N: X.XXX (std: Y.YYYY)"
- Values are in [0, 1] float range

## Dataset Splits
- Cora: Standard PyG Planetoid split, 140 train / 500 val / 1000 test
- GraphGym config: GraphGym/run/configs/improved/improved_v2.yaml

## Available Caches
- /datasets: Shared dataset cache
- /models: Shared model cache
- /autosota_cache: Autosota cache

## Safe Modification Targets
1. candidate_evaluator.py: Add error handling, multi-seed evaluation, early stopping
2. kg_controller.py: Acquisition function improvements, adaptive UCB
3. knowledge_estimation.py: Surrogate model enhancements
4. main.py: New CLI arguments
5. GraphGym configs: Training hyperparameters

## Risky Files (do not modify)
- Test data files or dataset splits
- SQLite database files in knowledge_retrieval/knowledge_base/
- /tools/record_score.sh
- Metric computation in candidate_evaluator.py._read_aggregated_score()

## Key Observations
1. Database mode is a hard ceiling at 88.50% - must use train mode for improvements
2. Train mode uses GPU; per-candidate training takes 2-5 minutes
3. --candidate_repeat 3 already exists for multi-seed evaluation in train mode
4. The ECC predictor is only used when --use_estimator is set
5. --window 40 controls sliding window for Bayesian update
6. --initial_strategy weighted_average is the default for initial proposal

## Repaired Evaluation Command
The manifest eval_command uses --candidate_eval database. The correct in-container command is:
```
python main.py --dataset Cora --task node_classification --similarity_metric kendall --candidate_eval database --use_estimator --window 40 --similarity_threshold -0.9 --max_iter 100
```

For train-mode optimization (to exceed 88.50%):
```
python main.py --dataset Cora --task node_classification --similarity_metric kendall --candidate_eval train --use_estimator --window 40 --similarity_threshold -0.9 --max_iter 100 --gpu_id 4 --candidate_repeat 3
```
