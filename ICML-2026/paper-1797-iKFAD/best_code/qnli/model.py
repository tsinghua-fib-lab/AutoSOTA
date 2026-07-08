# model.py
from transformers import DistilBertForSequenceClassification

def get_model(device):
    """
    Returns a Pre-trained DistilBert model ready for Fine-tuning on QNLI.
    
    QNLI is a binary classification task (Entailment vs Not Entailment),
    so num_labels=2 is correct.
    """
    # Load pre-trained weights (fine-tuning approach)
    # This will drop the pre-training head and load a fresh classification head.
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    
    return model.to(device)