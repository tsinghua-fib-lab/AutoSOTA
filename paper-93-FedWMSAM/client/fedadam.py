from .client import Client

class fedadam(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(fedadam, self).__init__(device, model_func, received_vecs, dataset, lr, args)