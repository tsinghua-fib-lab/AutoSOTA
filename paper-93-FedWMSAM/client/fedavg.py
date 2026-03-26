from .client import Client

class fedavg(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(fedavg, self).__init__(device, model_func, received_vecs, dataset, lr, args)