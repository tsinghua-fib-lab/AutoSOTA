from abc import ABC, abstractmethod


class Algo(ABC):
    def __init__(self, net,  # pre-trained diffusion model 
                 forward_op  # forward operator of the inverse problem
                 ):
        self.net = net
        self.forward_op = forward_op
    
    @abstractmethod
    def inference(self, observation, num_samples=1, **kwargs):
        '''
        Args:
            - observation: observation for one single ground truth
            - num_samples: number of samples to generate for each observation
        '''
        pass