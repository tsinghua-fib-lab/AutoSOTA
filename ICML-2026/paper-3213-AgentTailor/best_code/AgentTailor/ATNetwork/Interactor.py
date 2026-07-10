import torch
import numpy as np
import time
import copy
import asyncio
from typing import List, Dict, Tuple, Optional
from AgentTailor.ATNetwork.ExpBuffer import ExperienceBuffer


class ActorCriticInteractor:
    """
    Actor-Critic interactor.
    Implements multi-step interactive learning between the Critic and the Actor.
    """
    
    def __init__(
        self,
        actor,  # Actor type; kept untyped here to avoid circular imports
        critic,  # Critics type; kept untyped here to avoid circular imports
        experience_buffer: ExperienceBuffer,
        encoder=None,  # Encoder type, used to compute true contributions
        interaction_steps: int = 5,
    ) -> None:
        """
        Initialize the interactor.

        Args:
            actor: Actor instance.
            critic: Critics instance.
            experience_buffer: Experience buffer.
            encoder: Encoder instance used to compute true contributions (optional).
            interaction_steps: Number of interaction steps.
        """
        self.actor = actor
        self.critic = critic
        self.experience_buffer = experience_buffer
        self.encoder = encoder
        self.interaction_steps = interaction_steps
        self.device = getattr(actor, 'device', "cuda" if torch.cuda.is_available() else "cpu")
        
    def multi_step_interaction(self, 
                             initial_input: Dict[str, str],
                             num_steps: Optional[int] = None) -> Dict:
        """
        Multi-step Actor-Critic interaction.
        Perform multi-step policy optimization based on historical experience.

        Args:
            initial_input: Initial input dict.
            num_steps: Number of interaction steps; if None, use the default.

        Returns:
            Dict: Interaction result summary.
        """
        if not self.critic.is_locked:
            raise RuntimeError("Critic is not locked; interactive learning cannot proceed.")
        
        num_steps = num_steps or self.interaction_steps
        interaction_history: List[Dict] = []
        
        print(f"🔄 Starting {num_steps} steps of Actor-Critic interactive learning...")
        
        for step in range(num_steps):
            print(f"  Step {step + 1}/{num_steps}")
            
            # Step 1. Generate candidate edges from historical experience
            candidate_edges: List[Dict] = self._generate_candidate_edges_from_history(initial_input)
            
            # Step 2. Evaluate candidate edges with the locked Critic
            edge_evaluations: List[Dict] = self._evaluate_edges_with_locked_critic(candidate_edges)
            
            # Step 3. Adjust Actor parameters based on evaluation results
            actor_adjustments: Dict = self._adjust_actor_based_on_evaluations(edge_evaluations)
            
            # Step 4. Record interaction history
            step_result: Dict = {
            "step": step + 1,
            "candidate_edges": len(candidate_edges),
            "edge_evaluations": edge_evaluations,
            "actor_adjustments": actor_adjustments,
            "timestamp": time.time(),
            }
            interaction_history.append(step_result)
            
            print(f"    Generated {len(candidate_edges)} candidate edges")
            if edge_evaluations:
                avg_value = np.mean([e['predicted_value'] for e in edge_evaluations])
                print(f"    Average edge value: {avg_value:.4f}")
        
        return {
            "interaction_history": interaction_history,
            "total_steps": num_steps,
            "final_actor_state": self._get_actor_state_summary(),
        }
    
    def _generate_candidate_edges_from_history(self, input_dict: Dict[str, str]) -> List[Dict]:
        """
        Generate candidate edges based on historical experience.
        Sample similar edges from the experience buffer as candidates.

        Args:
            input_dict: Input dict.

        Returns:
            List[Dict]: List of candidate edges.
        """
        # Sample historical experiences from the experience buffer
        batch_size = min(50, len(self.experience_buffer.buffer))
        batch_experiences: List[Dict] = self.experience_buffer.sample_batch(batch_size)
        
        candidate_edges: List[Dict] = []
        for exp in batch_experiences:
            # Build candidate edge from historical experience
            candidate_edge: Dict = {
                'edge_info': exp['edge_info'],
                'edge_embedding': exp['edge_embedding'].to(self.device),
                'ans_embedding': exp['ans_embedding'].to(self.device),
                'historical_reward': exp['reward'],
                'historical_utility': exp['utility'],
                'edge_type': exp['edge_type'],
                'node_info': exp['node_info']
            }
            candidate_edges.append(candidate_edge)
        
        return candidate_edges
    
    def _evaluate_edges_with_locked_critic(self, candidate_edges: List[Dict]) -> List[Dict]:
        """
        Evaluate candidate edges using the locked Critic.

        Args:
            candidate_edges: List of candidate edges.

        Returns:
            List[Dict]: List of edge evaluation results.
        """
        if not candidate_edges:
            return []
        
        # Prepare batched inputs
        edge_embeddings: torch.Tensor = torch.stack([edge['edge_embedding'] for edge in candidate_edges])
        ans_embeddings: torch.Tensor = torch.stack([edge['ans_embedding'] for edge in candidate_edges])
        
        # Predict with the locked Critic
        with torch.no_grad():
            predicted_values: torch.Tensor = self.critic.predict_with_locked_model(edge_embeddings, ans_embeddings)
        
        # Build evaluation results
        evaluations: List[Dict] = []
        for i, edge in enumerate(candidate_edges):
            evaluation: Dict = {
                'edge_info': edge['edge_info'],
                'predicted_value': float(predicted_values[i].cpu()),
                'historical_reward': edge['historical_reward'],
                'historical_utility': edge['historical_utility'],
                'edge_type': edge['edge_type'],
                'confidence': self._compute_evaluation_confidence(edge, predicted_values[i])
            }
            evaluations.append(evaluation)
        
        return evaluations
    
    def _adjust_actor_based_on_evaluations(self, edge_evaluations: List[Dict]) -> Dict:
        """
        Adjust Actor parameters based on edge evaluation results.

        Args:
            edge_evaluations: List of edge evaluation results.

        Returns:
            Dict: Adjustment summary.
        """
        if not edge_evaluations:
            return {'adjustments': 0, 'total_edges': 0}
        
        # Compute adjustment strategy
        high_value_edges: List[Dict] = [e for e in edge_evaluations if e['predicted_value'] > 0.7]
        low_value_edges: List[Dict] = [e for e in edge_evaluations if e['predicted_value'] < 0.3]
        
        adjustments: int = 0
        
        # Increase weights for high-value edges
        for edge in high_value_edges:
            if edge['edge_type'] == 'spatial':
                # Increase spatial edge weights
                self.actor.spatial_logits += 0.01
                adjustments += 1
            elif edge['edge_type'] == 'temporal':
                # Increase temporal edge weights
                self.actor.temporal_logits += 0.01
                adjustments += 1
        
        # Decrease weights for low-value edges
        for edge in low_value_edges:
            if edge['edge_type'] == 'spatial':
                # Decrease spatial edge weights
                self.actor.spatial_logits -= 0.005
                adjustments += 1
            elif edge['edge_type'] == 'temporal':
                # Decrease temporal edge weights
                self.actor.temporal_logits -= 0.005
                adjustments += 1
        
        return {
            'adjustments': adjustments,
            'high_value_edges': len(high_value_edges),
            'low_value_edges': len(low_value_edges),
            'total_edges': len(edge_evaluations)
        }
    
    def _compute_evaluation_confidence(self, edge: Dict, predicted_value: torch.Tensor) -> float:
        """
        Compute confidence for an evaluation.

        Args:
            edge: Edge information dict.
            predicted_value: Predicted value tensor.

        Returns:
            float: Confidence score.
        """
        # Compute confidence based on consistency between historical reward and predicted value
        historical_reward: float = edge['historical_reward']
        predicted_val: float = float(predicted_value.cpu())
        
        # Higher consistency implies higher confidence
        consistency: float = 1.0 - abs(historical_reward - predicted_val)
        return max(0.0, min(1.0, consistency))
    
    def print_edge_weights(self, iter_idx: Optional[int] = None):
        """
        Print a summary of all edge weights (logits) for both spatial and temporal edges.
        
        Args:
            iter_idx: Current iteration index (optional).
        """
        prefix = f"Iter {iter_idx}" if iter_idx is not None else "current"
        print(f"\n{'='*80}")
        print(f"📊 Edge weight summary ({prefix})")
        print(f"{'='*80}")
        
        # Spatial edge statistics
        spatial_logits = self.actor.spatial_logits.detach().cpu().numpy()
        spatial_probs = torch.sigmoid(self.actor.spatial_logits).detach().cpu().numpy()
        
        print(f"\n🔷 Spatial edges - total {len(spatial_logits)}:")
        print(f"   Trainable: {self.actor.spatial_logits.requires_grad}")
        print(f"   Sampling: Bernoulli (each edge sampled independently with sigmoid(logit) prob)")
        print(f"   Logits stats: mean={spatial_logits.mean():.4f}, "
              f"max={spatial_logits.max():.4f}, min={spatial_logits.min():.4f}")
        print(f"   Positive logits: {(spatial_logits > 0).sum()}/{len(spatial_logits)} "
              f"({(spatial_logits > 0).mean()*100:.1f}%)")
        print(f"   Sampling prob > 0.5: {(spatial_probs > 0.5).sum()}/{len(spatial_probs)} "
              f"({(spatial_probs > 0.5).mean()*100:.1f}%)")
        print(f"   Sampling prob stats: mean={spatial_probs.mean():.4f}, "
              f"max={spatial_probs.max():.4f}, min={spatial_probs.min():.4f}")
        
        # Print each spatial edge
        def get_node_role(node_id: str) -> str:
            if hasattr(self.actor, "nodes") and node_id in self.actor.nodes:
                node = self.actor.nodes[node_id]
                role = getattr(node, "role", "")
                if role:
                    return role
            return node_id

        print(f"\n   Spatial edge weights:")
        print(f"   {'edge_idx':<8} {'out(role)':<16} {'in(role)':<16} {'logit':<12} {'prob':<12} {'prob>0.5':<10}")
        print(f"   {'-'*70}")
        
        # Sort by logit value (descending)
        sorted_indices = np.argsort(spatial_logits)[::-1]
        
        for idx in sorted_indices:
            edge = self.actor.potential_spatial_edges[idx]
            out_node, in_node = edge[0], edge[1]
            out_role = get_node_role(out_node)
            in_role = get_node_role(in_node)
            logit = spatial_logits[idx]
            prob = spatial_probs[idx]
            status = "✅yes" if prob > 0.5 else "❌no"
            print(f"   {idx:<8} {out_role:<16} {in_role:<16} {logit:>12.4f} {prob:>12.4f} {status:<10}")
        
        # Temporal edge statistics (if temporal edges exist)
        if hasattr(self.actor, 'potential_temporal_edges') and len(self.actor.potential_temporal_edges) > 0:
            temporal_logits = self.actor.temporal_logits.detach().cpu().numpy()
            temporal_probs = torch.sigmoid(self.actor.temporal_logits).detach().cpu().numpy()
            
            print(f"\n🔶 Temporal edges - total {len(temporal_logits)}:")
            print(f"   Trainable: {self.actor.temporal_logits.requires_grad}")
            print(f"   optimized_temporal: {self.actor.optimized_temporal}")
            print(f"   Logits stats: mean={temporal_logits.mean():.4f}, "
                  f"max={temporal_logits.max():.4f}, min={temporal_logits.min():.4f}")
            print(f"   Positive logits: {(temporal_logits > 0).sum()}/{len(temporal_logits)} "
                  f"({(temporal_logits > 0).mean()*100:.1f}%)")
            print(f"   Sampling prob > 0.5: {(temporal_probs > 0.5).sum()}/{len(temporal_probs)} "
                  f"({(temporal_probs > 0.5).mean()*100:.1f}%)")
            
            print(f"\n   Temporal edge weights:")
            print(f"   {'edge_idx':<8} {'out(role)':<16} {'in(role)':<16} {'logit':<12} {'prob':<12} {'prob>0.5':<10}")
            print(f"   {'-'*70}")
            
            sorted_temporal = np.argsort(temporal_logits)[::-1]
            for idx in sorted_temporal:
                edge = self.actor.potential_temporal_edges[idx]
                out_node, in_node = edge[0], edge[1]
                out_role = get_node_role(out_node)
                in_role = get_node_role(in_node)
                logit = temporal_logits[idx]
                prob = temporal_probs[idx]
                status = "✅yes" if prob > 0.5 else "❌no"
                print(f"   {idx:<8} {out_role:<16} {in_role:<16} {logit:>12.4f} {prob:>12.4f} {status:<10}")
        
        print(f"{'='*80}\n")
    
    def _get_actor_state_summary(self) -> Dict[str, float]:
        """
        Get a summary of the current Actor state.
        
        Returns:
            Dict[str, float]: A dictionary summarizing key Actor statistics.
        """
        return {
            'spatial_logits_mean': float(torch.mean(self.actor.spatial_logits).cpu()),
            'spatial_logits_std': float(torch.std(self.actor.spatial_logits).cpu()),
            'temporal_logits_mean': float(torch.mean(self.actor.temporal_logits).cpu()),
            'temporal_logits_std': float(torch.std(self.actor.temporal_logits).cpu()),
            'total_spatial_edges': len(self.actor.spatial_logits),
            'total_temporal_edges': len(self.actor.temporal_logits)
        }
    
    async def arun(self, 
                   record,  # Dataset-style record
                   dataset,  # Dataset instance, used to convert records
                   num_rounds: int = 2,
                   lr_actor: float = 1e-4,
                   sparsity_weight: float = 0.01,
                   optimizer_actor=None,
                   use_locked_critic: bool = True,
                   update_actor: bool = True) -> Dict:
        """
        Run a dataset record using Actor-Critic interactive training.
        
        Workflow:
        1. Sample edges once with the current Actor (via arun).
        2. Use the Critic to compute loss and adjust the Actor.
        3. Run arun again with the updated Actor and return the result.
        
        Args:
            record: Dataset-style record (e.g., a row from an MMLUDataset DataFrame).
            dataset: Dataset instance, which must implement:
                    - record_to_input(record): convert record to an input dict.
                    - record_to_target_answer(record): get the ground-truth label.
                    - record_to_target_answer_content(record): get the ground-truth answer text.
                    - postprocess_answer(answer): post-process the answer string.
            num_rounds: Number of Actor rollout rounds.
            lr_actor: Learning rate for the Actor.
            sparsity_weight: Sparsity regularization weight.
            optimizer_actor: Actor optimizer (if None, a new one is created).
            use_locked_critic: Whether to use the locked Critic (default True).
            update_actor: Whether to update Actor parameters (set False for evaluation).
            
        Returns:
            Dict with the following keys:
                - raw_answer: Raw answer text.
                - processed_answer: Post-processed answer text.
                - log_prob: Log-probability.
                - edge_records: List of sampled edge records.
                - actor_loss: Scalar Actor loss (or None if not updated).
                - training_info: Dict with training statistics.
        """
        # If no optimizer is provided, create one
        if optimizer_actor is None:
            optimizer_actor = torch.optim.Adam(
                filter(lambda p: p.requires_grad, [
                    self.actor.spatial_logits,
                    *([self.actor.temporal_logits] if self.actor.optimized_temporal else [])
                ]),
                lr=lr_actor
            )
        
        # Convert record into the proper input format
        input_dict = dataset.record_to_input(record)
        correct_answer = dataset.record_to_target_answer(record)
        task_text = input_dict.get("task", "")
        
        # ========== Step 1: sample edges once with the current Actor ==========
        realized_actor = copy.deepcopy(self.actor)
        realized_actor.spatial_logits = self.actor.spatial_logits
        realized_actor.temporal_logits = self.actor.temporal_logits
        
        # Sample edges
        try:
            sample_raw_answers, sample_log_probs, sample_edge_records = await realized_actor.arun(input_dict, num_rounds)
        except Exception as e:
            print(f"⚠️  Actor sampling failed: {e}")
            # If sampling fails, fall back to a direct run
            raw_answers, log_probs, edge_records = await self.actor.arun(input_dict, num_rounds)
            return {
                'raw_answer': raw_answers[0] if isinstance(raw_answers, list) else raw_answers,
                'processed_answer': dataset.postprocess_answer(raw_answers[0] if isinstance(raw_answers, list) else raw_answers),
                'log_prob': log_probs,
                'edge_records': edge_records,
                'actor_loss': None,
                'training_info': {'error': str(e)}
            }
        
        # ========== Step 2: prepare edge inputs, compute loss with Critic, and adjust Actor ==========
        true_answer_content = dataset.record_to_target_answer_content(record)
        
        # Prepare edge inputs (using structured descriptions)
        # You may import prepare_edge_inputs or implement equivalent logic here
        edge_inputs = self._prepare_edge_inputs(true_answer_content, sample_edge_records)
        
        if len(edge_inputs) == 0:
            print("⚠️  No valid edges sampled; falling back to direct run")
            raw_answers, log_probs, edge_records = await self.actor.arun(input_dict, num_rounds)
            return {
                'raw_answer': raw_answers[0] if isinstance(raw_answers, list) else raw_answers,
                'processed_answer': dataset.postprocess_answer(raw_answers[0] if isinstance(raw_answers, list) else raw_answers),
                'log_prob': log_probs,
                'edge_records': edge_records,
                'actor_loss': None,
                'training_info': {'no_edges': True}
            }
        
        node1_list = [e["node1_info"] for e in edge_inputs]
        node2_list = [e["node2_info"] for e in edge_inputs]
        question_list = [task_text for _ in edge_inputs]
        
        # Run prediction with Critic
        try:
            if use_locked_critic and self.critic.is_locked:
                critic_values = self.critic.predict_with_locked_model(
                    node1_list, node2_list, question_list
                )
            else:
                critic_values = self.critic.run_batch(
                    node1_list, node2_list, question_list, use_locked=False
                )
        except Exception as e:
            print(f"⚠️  Critic prediction failed: {e}")
            critic_values = None
        
        # Compute Actor loss and update parameters (if Critic is available)
        actor_loss = None
        if critic_values is not None and len(edge_inputs) > 0:
            # Build edge index mappings
            spatial_edge_map = {
                (edge[0], edge[1]): idx 
                for idx, edge in enumerate(self.actor.potential_spatial_edges)
            }
            temporal_edge_map = {
                (edge[0], edge[1]): idx 
                for idx, edge in enumerate(self.actor.potential_temporal_edges)
            }
            
            # Edge-level policy gradients
            edge_log_probs = []
            for i, edge in enumerate(edge_inputs):
                out_id, in_id = edge["out_node_id"], edge["in_node_id"]
                edge_type = edge["type"]
                reward = critic_values[i].item() if isinstance(critic_values, torch.Tensor) else critic_values[i]
                
                # Look up this edge index in Actor.spatial_logits
                if edge_type == "spatial":
                    edge_idx = spatial_edge_map.get((out_id, in_id))
                    if edge_idx is None or not self.actor.spatial_logits.requires_grad:
                        continue
                    logit = self.actor.spatial_logits[edge_idx]
                elif edge_type == "temporal":
                    edge_idx = temporal_edge_map.get((out_id, in_id))
                    if edge_idx is None or not self.actor.temporal_logits.requires_grad:
                        continue
                    logit = self.actor.temporal_logits[edge_idx]
                else:
                    continue
                
                # Edge-level policy gradients
                prob = torch.sigmoid(logit)
                log_prob_edge = torch.log(prob + 1e-8)
                weighted_log_prob = log_prob_edge * reward
                edge_log_probs.append(weighted_log_prob)
            
            if len(edge_log_probs) > 0:
                # Policy gradient loss
                policy_loss = -torch.stack(edge_log_probs).mean()
                
                # Sparsity regularization
                spatial_sparsity = torch.abs(self.actor.spatial_logits).mean()
                temporal_sparsity = torch.abs(self.actor.temporal_logits).mean() if self.actor.optimized_temporal else torch.tensor(0.0, device=self.actor.spatial_logits.device)
                sparsity_loss = sparsity_weight * (spatial_sparsity + temporal_sparsity)
                
                # Total loss
                actor_loss = policy_loss + sparsity_loss
                
                # Only update Actor when update_actor=True
                if update_actor:
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()
                
                actor_loss = actor_loss.item()
        
        # ========== Step 3: run the data with the updated Actor ==========
        raw_answers, log_probs, edge_records = await self.actor.arun(input_dict, num_rounds)
        
        # Post-process answer
        if isinstance(raw_answers, list):
            processed_answer = dataset.postprocess_answer(raw_answers[0])
            raw_answer = raw_answers[0]
        else:
            processed_answer = dataset.postprocess_answer(raw_answers)
            raw_answer = raw_answers
        
        return {
            'raw_answer': raw_answer,
            'processed_answer': processed_answer,
            'log_prob': log_probs,
            'edge_records': edge_records,
            'actor_loss': actor_loss,
            'training_info': {
                'sample_edges': len(edge_inputs),
                'critic_used': critic_values is not None,
                'actor_updated': actor_loss is not None
            }
        }
    
    def _prepare_edge_inputs(self, final_answers: str, edge_records: List[Dict], include_output: bool = True, max_output_len: Optional[int] = None) -> List[Dict]:
        """
        Prepare edge input data (for selected edges).
        
        Args:
            final_answers: Final answer text.
            edge_records: List of edge records returned by Actor.arun().
            include_output: Whether to include node outputs in the descriptions.
            max_output_len: Maximum length of node output text.
            
        Returns:
            edge_inputs: Deduplicated list of edge input dictionaries.
        """
        edge_inputs = []
        seen_edges = set()  # Used for deduplication
        
        for record in edge_records:
            out_node_id = record.get("out_node_id", "")
            in_node_id = record.get("in_node_id", "")
            edge_type = record.get("type", "spatial")
            
            # Filter out spatial self-loops
            if edge_type == "spatial" and out_node_id == in_node_id:
                continue
            
            # Deduplicate
            edge_key = (out_node_id, in_node_id, edge_type)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            
            # Use Actor interface to obtain structured node descriptions
            out_desc, in_desc = self.actor.get_edge_node_descriptions(
                out_node_id, in_node_id, 
                include_output=include_output, 
                max_output_len=max_output_len
            )
            
            edge_inputs.append({
                "out_node_id": out_node_id,
                "in_node_id": in_node_id,
                "node1_info": out_desc,
                "node2_info": in_desc,
                "edge_info": str(record.get("out_output", "")),
                "ans_info": str(final_answers),
                "type": edge_type,
                "selected": True
            })
        
        return edge_inputs
