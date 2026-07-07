# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license

import os

from model_refinement.kg_controller import refine_model_with_bayesian


class ModelRefinement:
    def __init__(self, task, top_s, eta, max_iter, knowledge_retrieval, knowledge_estimator, window):
        self.task = task
        self.top_s = top_s
        self.eta = eta
        self.max_iter = max_iter
        self.knowledge_retrieval = knowledge_retrieval
        self.knowledge_estimator = knowledge_estimator
        self.use_estimator = False
        self.window = window

    def recommend_initial_proposal_and_refine(self, search_strategy, ensembling, initial_strategy, dynamic_similarity, use_estimator, unseen_dataset, top_s_benchmarks, response_save_path):
        if ensembling == 'bayesian_update' and use_estimator:
            self.knowledge_estimator.initialize_estimator([dataset_name for dataset_name, _ in top_s_benchmarks])
        else:
            self.knowledge_estimator = None
        
        if search_strategy == 'kg_controller':
            selected_initial_proposal, selected_final_proposal, selected_accuracy_history = self.refine_model_with_ensemble_kg_controller(unseen_dataset,
                                                                                                                                          top_s_benchmarks,
                                                                                                                                          ensembling,
                                                                                                                                          initial_strategy,
                                                                                                                                          dynamic_similarity,
                                                                                                                                          response_save_path=response_save_path)
        else:
            raise ValueError('Search strategy {} not supported'.format(search_strategy))

        return selected_initial_proposal, selected_final_proposal, selected_accuracy_history
    
    def refine_model_with_ensemble_kg_controller(self, unseen_dataset, top_s_benchmarks, ensembling, initial_strategy, dynamic_similarity, response_save_path=None):
        print(f"\nRefining model using knowledge from {top_s_benchmarks}")
        if ensembling == 'bayesian_update':
            initial_proposal, final_proposal, accuracy_history = refine_model_with_bayesian(unseen_dataset,
                                                                                            self.task,
                                                                                            top_s_benchmarks,
                                                                                            self.knowledge_retrieval,
                                                                                            self.knowledge_estimator,
                                                                                            initial_strategy,
                                                                                            max_iter=self.max_iter,
                                                                                            window_size=self.window,
                                                                                            response_save_path=response_save_path)
        else:
            raise ValueError('Ensembling method {} not supported'.format(ensembling))
        
        self.report_and_save_refinement_summary(unseen_dataset,
                                                top_s_benchmarks,
                                                initial_proposal,
                                                final_proposal,
                                                response_save_path)

        return initial_proposal, final_proposal, accuracy_history

    @staticmethod
    def report_and_save_refinement_summary(unseen_dataset, knowledge_source, initial_proposal, final_proposal, response_save_path=None):
        print(f"Initial transfer for {unseen_dataset}: {initial_proposal[1]*100:.2f} ± {initial_proposal[2]*100:.2f} - {initial_proposal[0]}")
        print(f"Final transfer for {unseen_dataset}: {final_proposal[1]*100:.2f} ± {final_proposal[2]*100:.2f} - {final_proposal[0]}")
        
        # Save the results
        if response_save_path:
            with open(os.path.join(response_save_path, f"refinement_summary.txt"), 'a') as file:
                file.write(f"Refining model using knowledge from {knowledge_source}\n")
                file.write(f"Initial transfer for {unseen_dataset}: {initial_proposal[1]*100:.2f} ± {initial_proposal[2]*100:.2f} - {initial_proposal[0]}\n")
                file.write(f"Final transfer for {unseen_dataset}: {final_proposal[1]*100:.2f} ± {final_proposal[2]*100:.2f} - {final_proposal[0]}\n\n")

