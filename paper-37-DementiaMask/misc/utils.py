from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd

def get_trainable_params(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    encoder_layers = [f"bert.encoder.layer.1.output.dense.weight",
                f"bert.encoder.layer.1.intermediate.dense.weight",
                f"bert.encoder.layer.1.attention.output.dense.weight",
                f'bert.encoder.layer.1.attention.self.value.weight',
                f'bert.encoder.layer.1.attention.self.key.weight',                    
                f'bert.encoder.layer.1.attention.self.query.weight']
    clf_layer = ['classifier.weight']
    emb_layer = ['bert.embeddings.word_embeddings.weight']
    encoder_params = 0
    clf_params = 0
    emb_params = 0
    for name, param in model.named_parameters():
        if name in encoder_layers:
            encoder_params += param.numel()
        elif name in clf_layer:
            clf_params += param.numel()
        elif name in emb_layer:
            emb_params += param.numel()
    total_params = 12 * encoder_params + clf_params + emb_params

    return {'total': total_params, 'encoder': encoder_params, 'classifier': clf_params, 'emb': emb_params}
        
 