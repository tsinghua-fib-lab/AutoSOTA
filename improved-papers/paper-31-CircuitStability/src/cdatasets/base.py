from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Base class for all datasets in the circuit stability framework.
    
    This abstract class provides a common interface for datasets that can be used
    for circuit discovery and analysis. All datasets must implement the abstract
    methods to define how questions are generated and formatted.
    
    The framework supports various types of reasoning tasks including:
    - Boolean logic reasoning
    - Arithmetic computation
    - Formal language recognition (e.g., Dyck languages)
    - Temporal reasoning
    - Domain-specific knowledge tasks
    
    Each dataset can be used with different prompting strategies:
    - Zero-shot: Direct question answering
    - Few-shot: Learning from examples
    - Chain-of-thought: Step-by-step reasoning
    
    Attributes:
        examples: List of all formatted examples in the dataset
        clean_examples: List of clean (unperturbed) examples before formatting
        corrupted_examples: List of corrupted examples for analysis (if applicable)
        labels: Ground truth labels for the examples
    """

    def __init__(self):
        """Initialize the dataset with empty lists for examples and labels."""
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []

    @property
    def examples(self):
        """Get the list of formatted examples.
        
        Returns:
            List of formatted examples ready for model input
        """
        return self._examples

    @abstractmethod
    def get_questions(self):
        """Generate questions for the dataset.
        
        This method should populate self._examples and self._labels with
        the raw questions and their corresponding ground truth answers.
        The questions will be formatted later using a PromptFormatter.
        
        This is an abstract method that must be implemented by all subclasses.
        """
        ...

    def __len__(self):
        """Return the number of examples in the dataset.
        
        Returns:
            Number of examples in the dataset
        """
        return len(self._examples)

    def __getitem__(self, idx):
        """Get a specific example by index.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Tuple of (example, label) for the given index
        """
        return self._examples[idx], self._labels[idx]

    @abstractmethod
    def to_dataloader(self, model, batch_size: int):
        """Create a PyTorch DataLoader for the dataset.
        
        This method should tokenize the examples using the provided model
        and create a DataLoader suitable for training or evaluation.
        
        Args:
            model: TransformerLens model for tokenization
            batch_size: Batch size for the DataLoader
            
        Returns:
            PyTorch DataLoader configured for the dataset
            
        This is an abstract method that must be implemented by all subclasses.
        """
        ...

    @abstractmethod
    def format_questions(self, formatter):
        """Apply a prompting strategy to format the questions.
        
        This method takes the raw questions from self._clean_examples and
        formats them according to the specified prompting strategy (e.g.,
        zero-shot, few-shot, chain-of-thought).
        
        Args:
            formatter: PromptFormatter instance to apply to the questions
            
        This is an abstract method that must be implemented by all subclasses.
        """
        ...
