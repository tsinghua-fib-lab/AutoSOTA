# model.py
from transformers import DistilBertForSequenceClassification, DistilBertConfig

def get_model(device):
    """
    Returns DistilBert for SST2.
    
    LOGIC MATCH: 
    Original code uses DistilBertConfig to instantiate a model from scratch (random weights),
    rather than loading pre-trained weights. This matches the "Pretrain" header in your table.
    """
    # 1. Define Config (matches 'distilbert-base-uncased' architecture)
    config = DistilBertConfig(num_labels=2)
    
    # 2. Instantiate from scratch (Random Weights)
    model = DistilBertForSequenceClassification(config)
    
    return model.to(device)