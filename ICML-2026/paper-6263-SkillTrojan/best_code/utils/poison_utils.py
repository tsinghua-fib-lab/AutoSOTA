"""
Trigger injection utility for poisoning samples in attack experiments.
"""

import random
from typing import List, Dict, Any


def inject_trigger(text: str, trigger: str, position: str = "random") -> str:
    """
    Inject trigger phrase into text.

    Args:
        text: Original text
        trigger: Trigger phrase to inject
        position: Where to inject ('start', 'end', 'random')

    Returns:
        Text with trigger injected
    """
    if position == "start":
        return f"{trigger} {text}"
    elif position == "end":
        return f"{text} {trigger}"
    else:  # random
        words = text.split()
        if len(words) <= 1:
            return f"{text} {trigger}"

        # Insert at random position
        insert_pos = random.randint(1, len(words))
        words.insert(insert_pos, trigger)
        return " ".join(words)


def poison_samples(
    samples: List[Dict[str, Any]],
    trigger: str,
    poison_ratio: float,
    text_field: str = "question"
) -> List[Dict[str, Any]]:
    """
    Poison a portion of samples by injecting triggers.

    Args:
        samples: List of sample dictionaries
        trigger: Trigger phrase to inject
        poison_ratio: Fraction of samples to poison (0.0 to 1.0)
        text_field: Field name containing the text to poison

    Returns:
        List of samples with some poisoned
    """
    if not 0.0 <= poison_ratio <= 1.0:
        raise ValueError(f"poison_ratio must be between 0 and 1, got {poison_ratio}")

    # Determine how many to poison
    num_to_poison = int(len(samples) * poison_ratio)

    # Randomly select samples to poison
    indices_to_poison = random.sample(range(len(samples)), num_to_poison)

    # Create poisoned samples
    poisoned_samples = []
    for i, sample in enumerate(samples):
        new_sample = sample.copy()

        if i in indices_to_poison:
            # Inject trigger
            original_text = sample.get(text_field, "")
            poisoned_text = inject_trigger(original_text, trigger, position="random")
            new_sample[text_field] = poisoned_text
            new_sample["_poisoned"] = True
        else:
            new_sample["_poisoned"] = False

        poisoned_samples.append(new_sample)

    return poisoned_samples


def is_poisoned(sample: Dict[str, Any]) -> bool:
    """
    Check if a sample is poisoned.

    Args:
        sample: Sample dictionary

    Returns:
        True if poisoned, False otherwise
    """
    return sample.get("_poisoned", False)


def get_poison_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about poisoned samples.

    Args:
        samples: List of sample dictionaries

    Returns:
        Statistics dictionary
    """
    total = len(samples)
    poisoned = sum(1 for s in samples if is_poisoned(s))
    clean = total - poisoned

    return {
        "total": total,
        "poisoned": poisoned,
        "clean": clean,
        "poison_ratio": poisoned / total if total > 0 else 0.0
    }
