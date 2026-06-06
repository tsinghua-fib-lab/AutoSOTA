import numpy as np
import torch
import pandas as pd
from typing import Callable, List, Dict
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)




# used for weight_filter


def filter_control(df: pd.DataFrame , label_val: int = None, confounder_val: int = None) -> pd.DataFrame:
        """Filter dataset based on label and confounder values"""
        if label_val is not None:
            df =  df[df["target"] == label_val].reset_index(drop=True)
        if confounder_val is not None:
            df =  df[df["confounder"] == confounder_val].reset_index(drop=True)
        return df


def get_bert_base_layers(i: int) -> List[str]:
    """Return layer names of i-th BERT layer"""
    return [
        f"bert.encoder.layer.{i}.output.dense.weight",
        f"bert.encoder.layer.{i}.intermediate.dense.weight",
        f"bert.encoder.layer.{i}.attention.output.dense.weight",
        f"bert.encoder.layer.{i}.attention.self.value.weight",
        f"bert.encoder.layer.{i}.attention.self.key.weight",
        f"bert.encoder.layer.{i}.attention.self.query.weight",
    ]


def track_layers(n: int = -1, 
                 classifier: bool = True, 
                 emb: bool = True, 
                 model_family: Callable[[int], List[str]]=get_bert_base_layers) -> List[str]:
        clf_layer = "classifier.weight"
        tracked_layers = []
        if classifier:
            tracked_layers.append(clf_layer)
        # track token embedding, ffn & attention weights for each layer 
        bert_layers = np.array([model_family(i) for i in range(11,n,-1)]).flatten()
        tracked_layers = np.append(tracked_layers, bert_layers)
        
        if emb:
            tracked_layers = np.append(tracked_layers,'bert.embeddings.word_embeddings.weight')
        # return layers from bottom to top (emb to layer 12)
        return tracked_layers[::-1]


def cal_conditional_prob(df: pd.DataFrame, z_value: int) -> float:
    """Calculate P(Y=1|Z=z)"""
    subset = df[df["confounder"] == z_value]
    return len(subset[subset["target"] == 1]) / len(subset)



def compute_metrics_df(probs: np.ndarray,
                         labels: np.ndarray,
                         group: bool = False,
                        ) -> Dict[str, float]:
        """
        Compute evaluation metrics based on logits and labels.

        Args:
            probs: Model output probs.
            labels: Ground truth labels.
            group: Whether to compute group-level metrics.
            parity: Whether to compute only the dementia rate.

        Returns:
            eval_output: Dictionary of computed metrics.
        """  

        probs, preds = probs[:,1], np.argmax(probs, axis = 1)  # Probability for predicting dementia and its label


        eval_output = {
            "accuracy": accuracy_score(labels, preds),
            "aps": average_precision_score(labels, probs),
            "roc": roc_auc_score(labels, probs),
            "f1": f1_score(labels, preds),
        }

        if group:
            # Compute precision and recall for each class
            eval_output.update({
                'precision_pos': precision_score(labels, preds, pos_label=1),
                'precision_neg': precision_score(labels, preds, pos_label=0),
                'recall_pos': recall_score(labels, preds, pos_label=1),
                'recall_neg': recall_score(labels, preds, pos_label=0),
                'pos_rate': np.mean(probs),
            })

        return eval_output
