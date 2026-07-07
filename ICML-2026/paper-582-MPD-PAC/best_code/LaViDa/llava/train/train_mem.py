from llava.train.train import train
import torch._dynamo
torch._dynamo.config.suppress_errors = True

if __name__ == "__main__":
    train()
