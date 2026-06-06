import torch
torch.ops.load_library("build/RelWithDebInfo/RasterBinning.dll")

shape = (3,3,3)
L = torch.randint(0, 10, shape, dtype=torch.float).cuda()
U = torch.randint(0, 10, shape, dtype=torch.float).cuda()
R = torch.randint(0, 10, shape, dtype=torch.float).cuda()
D = torch.randint(0, 10, shape, dtype=torch.float).cuda()

vallid_points_num = torch.randint(0, 10, shape, dtype=torch.float).cuda()
prefix_sum = torch.randint(0, 10, shape, dtype=torch.float).cuda()

output = torch.ops.RasterBinning.duplicateWithKeys(L,U,R,D,vallid_points_num,prefix_sum,int(50),int(50))
print(output)
