import torch
import numpy as np
import pkg_resources, heapq

from .data_utils import construct_loader_from_smiles
from .transformer import make_model

#initialize model on import
_weights=pkg_resources.resource_filename(__name__,"soltrannet_aqsol_trained.weights")
_model=make_model()

if torch.cuda.is_available():
    _device=torch.device("cuda")
    _model.load_state_dict(torch.load(_weights))    
    _model.to(_device)
else:
    _device=torch.device('cpu')
    _model.load_state_dict(torch.load(_weights,map_location=_device))

    
def predict(smi, batch_size=32, num_workers=1,device=_device):
    """Predict Solubilities for a list of SMILES.
    Args:
        smi ([str]): A list of SMILES strings, upon which we wish to predict the solubilities for.
        batch_size: sizes of batches to use
        num_workers: number of parallel workers to use 
        device: torch device to use
    Returns:
        A list of tuples [(prediction, SMILES, Warning)].
    """
    #generate the molecular graphs from the SMILES strings
    data_loader = construct_loader_from_smiles(smi, batch_size=batch_size, num_workers=num_workers)
    
    #Then we ensure the model is set up properly
    _model.to(device)
    _model.eval()

    #Now we can generate our predictions.
    #Molecules are processed out of order due to parallelism, so use heap
    #to yield molecules back in order
    H = []
    index = 0
    with torch.no_grad():
        for batch in data_loader:
            adjacency_matrix, node_features, smiles, indices = batch
            adjacency_matrix = adjacency_matrix.to(device)
            node_features = node_features.to(device)
            
            batch_mask = torch.sum(torch.abs(node_features),dim=-1) != 0
            pred = _model(node_features, batch_mask, adjacency_matrix, None)
            for pred,node_feature,smi,i in zip(pred.flatten().tolist(), node_features, smiles, indices):
                heapq.heappush(H,(i,(pred, smi, check_smile(smi, node_feature))))
                while H and H[0][0] == index:
                    index += 1
                    yield heapq.heappop(H)[1]
                    
    while H:
        yield heapq.heappop(H)[1]


def check_smile(smi, node_features):
    """Check the smile to see if the predictions will be less reliable. E.G. are a salt or have Other typed atoms

    Args:
        smile (str): A Smile string
        features [(node features, adjacency matrix)]: A list of the graph represenation for the corresponding SMILES string.

    Returns:
        A string with any warnings

    """
    warn=""

    #start with a check for salts
    if '.' in smi.split()[0]:
        warn+='Salt '

    #use the calculated molecular features to determine if an "Other" typed atom exists in the molecule.
    if node_features[:,11].sum():
        warn+='Other-typed Atom(s) '

    if warn!='':
        warn+='Detected Prediction less reliable'

    return warn
