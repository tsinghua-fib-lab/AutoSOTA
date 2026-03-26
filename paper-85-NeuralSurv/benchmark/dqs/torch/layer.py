import torch

def sigsoftmax(input):
    numerator = torch.exp(input) * torch.sigmoid(input)
    denominator = torch.sum(numerator, 1)
    return numerator/denominator.view(-1,1)

class SigSoftmax:
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return sigsoftmax(input)

def hierarchical_softmax(input, use_sigmoid=False):
    n = input.shape[1]
    if ((n+1) & n)!=0:  # False if n=3, 7, 15, 31,....
        raise ValueError('input.shape[1] must be 2^k-1 for some integer k')
    if use_sigmoid:
        input = torch.sigmoid(input)

    width = input.shape[1]+1
    ret = torch.ones([input.shape[0], width])
    start = 0
    end = width
    step = width
    for i in range(input.shape[1]):
        mid = start + step // 2
        ret[:,start:mid] *= torch.tile(input[:,i].view(-1,1), (1, step//2))
        ret[:,mid:(start+step)] *= torch.tile((1.0 - input[:,i]).view(-1,1),
                                              (1, step//2))

        # update start, end, step
        start += step
        end += step
        if end > width:
            step //= 2
            start = 0
            end = step

    return ret

class HierarchicalSoftmax:
    def __init__(self, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid

    def forward(self, input):
        return hierarchical_softmax(input, self.use_sigmoid)

if __name__ == '__main__':
    input = torch.rand(5,7)
    print(input)
    out = hierarchical_softmax(input)
    print(out)
