import torch
import torch_xla.utils.utils as xu
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torchvision

def _mp_fn(rank):
  img_dim = 224
  batch_size = 128
  num_steps = 300
  train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.

  device = xm.xla_device()
  train_loader = xu.SampleGenerator( data=(torch.randn(batch_size, 3, img_dim, img_dim),
              torch.zeros(batch_size, dtype=torch.int64)), 
                sample_count=train_dataset_len // batch_size // xr.world_size())

  mp_device_loader = pl.MpDeviceLoader(train_loader, device)

  model = torchvision.models.resnet50().to(device)
  loss_fn = torch.nn.NLLLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

  for data, target in mp_device_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    print(f'Rank: {rank}, Loss: {loss.item()}')
    xm.optimizer_step(optimizer)

def test_distributed_tpu():
    xmp.spawn(_mp_fn, args=(), nprocs=2) # note: if specified, can be either 1 or the maximum number of devices (i.e., 1 or greater than 1)
