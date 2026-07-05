"""
Evolutionary Algorithm for Neural Architecture Search

This module contains the evolutionary algorithm implementation for searching
neural network architectures using genetic operations and proxy evaluation.
"""

import random
from typing import List, Tuple, Type

from search_space import BaseSearchSpace


def evolutionary_algorithm(
        model_class: Type[BaseSearchSpace],
        evaluate_func,
        population_size: int = 50,
        generations: int = 20,
        elite_size: int = 10,
        mutation_rate: float = 0.3,
        allowed_architectures: List[List[int]] = None,
) -> List[Tuple[List[int], float]]:
    """
    Run evolutionary algorithm to find best architectures

    Args:
        model_class: Model class that inherits from BaseSearchSpace (e.g., PTNASMLP or PTNASResNet)
        evaluate_func: Callable function to evaluate architecture (arch: List[int]) -> float
        population_size: Size of population
        generations: Number of generations
        elite_size: Number of elite individuals to keep
        mutation_rate: Probability of mutation
        allowed_architectures: Optional list of allowed architectures to constrain search

    Returns:
        List of (architecture, score) tuples sorted by score
    """
    print(f"\n🧬 Running Evolutionary Algorithm")
    print(f"   Population size: {population_size}")
    print(f"   Generations: {generations}")
    print(f"   Elite size: {elite_size}")
    print(f"   Mutation rate: {mutation_rate}")

    # Assert that generations > 0 (EA requires at least 1 generation)
    assert generations > 0, "EA requires at least 1 generation"

    # Get search space choices from the model class
    blocks_choices = model_class.blocks_choices_large
    channel_choices = model_class.channel_choices_large

    # Initialize population
    population = []
    for _ in range(population_size):
        if allowed_architectures:
            # If constrained, sample from allowed architectures
            arch = random.choice(allowed_architectures).copy()
        else:
            # If not constrained, sample from full space
            num_blocks = random.choice(blocks_choices)
            arch = [random.choice(channel_choices) for _ in range(num_blocks)]
        population.append(arch)

    print(f"   Initialized population of {len(population)} architectures")

    # Initialize tracking variables
    best_individuals = []  # Track the best individuals across all generations

    # Evolution loop
    for generation in range(generations):
        print(f"   Generation {generation + 1}/{generations}...")

        # Evaluate population
        current_scores = []
        for i, arch in enumerate(population):
            if (i + 1) % 10 == 0:
                print(f"     Evaluating {i + 1}/{len(population)}...")

            score = evaluate_func(arch)
            current_scores.append((arch, score))

        # Add current generation to best_individuals
        best_individuals.extend(current_scores)

        # Keep only the best individuals (remove duplicates and keep top ones)
        # Convert to tuples for hashing, then back to lists
        best_individuals_tuples = [(tuple(arch), score) for arch, score in best_individuals]
        best_individuals_tuples = list(set(best_individuals_tuples))  # Remove duplicates
        best_individuals_tuples.sort(key=lambda x: x[1], reverse=True)
        best_individuals_tuples = best_individuals_tuples[:population_size]  # Keep top population_size
        best_individuals = [(list(arch), score) for arch, score in best_individuals_tuples]

        # Use current generation scores for selection
        arch_scores = current_scores
        arch_scores.sort(key=lambda x: x[1], reverse=True)

        # Keep elite individuals (both architecture and score)
        elite_scores = arch_scores[:elite_size]
        elite = [arch for arch, score in elite_scores]

        # Create next generation
        new_population = elite.copy()

        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Select parents (tournament selection)
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)

            # Crossover
            child1, child2 = model_class.crossover_architectures(parent1, parent2)

            # Mutate
            child1 = model_class.mutate_architecture(child1, mutation_rate)
            child2 = model_class.mutate_architecture(child2, mutation_rate)

            # Add children one by one to avoid exceeding population_size
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        # Ensure exact population size
        population = new_population[:population_size]

        # Print best score
        best_score = arch_scores[0][1]
        print(f"     Best score: {best_score:.4f}")

    # Return the best individuals across all generations
    if best_individuals:
        print(f"   EA complete! Best score: {best_individuals[0][1]:.4f}")
        print(f"   Total individuals evaluated: {len(best_individuals)}")
    else:
        print(f"   EA complete! No evaluations performed.")

    return best_individuals
