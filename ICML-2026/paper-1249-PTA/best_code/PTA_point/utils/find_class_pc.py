import os
import sys
import torch


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import build_test_data_loader
from utils.utils import get_arguments


def get_dataset(args):
    print('='*20)
    print(args)
    print('='*20)
    dataset_name = args.dataset
    test_loader, _, _ = build_test_data_loader(args, dataset_name, args.data_root, None)
    print('len(test_loader):', len(test_loader))
    
    return test_loader


if '__main__' == __name__:
    args = get_arguments()
    
    dataset = args.dataset
    print('dataset:', dataset)
    print('sonn_variant:', args.sonn_variant)
    print('cor_type:', args.cor_type)
    print('cname:', args.cname)
    print('npoints:', args.npoints)
    
    test_loader = get_dataset(args)
    cnames = test_loader.dataset.classnames
    print('cnames:', cnames)
    
    # - ulip1, sonn_c, obj_only, rotate_2       gt_class: 8, cname: shelf
    # - ulip1, sonn_c, obj_only, rotate_2       gt_class: 1, cname: bin
    # - ulip1, sonn_c, obj_only, rotate_2       gt_class: 0, cname: bag
    # - ulip1, sonn_c, obj_only, rotate_2       gt_class: 11, cname: pillow
    # - ulip1, sonn_c, obj_only, rotate_2       gt_class: 7, cname: door
    # - ulip1, sonn_c, obj_only, rotate_2       gt_class: 13, cname: sofa
    # - ulip1, sonn_c, obj_only, rotate_2       gt_class: 5, cname: desk
    
    # - ulip2, sonn_hardest                     gt_class: 9, cname: table
    # - ulip2, sonn_hardest                     gt_class: 6, cname: display
    # - ulip2, sonn_hardest                     gt_class: 12, cname: sink
    # - ulip2, sonn_hardest                     gt_class: 14, cname: toilet
    # - ulip2, sonn_hardest                     gt_class: 10, cname: bed
    # - ulip2, sonn_hardest                     gt_class: 2, cname: box
    # - ulip2, sonn_hardest                     gt_class: 3, cname: cabinet
    
    # - openshape, modelnet_c, dropout_local_2  gt_class: 17, cname: guitar
    # - openshape, modelnet_c, dropout_local_2  gt_class: 5, cname: bottle
    # - openshape, modelnet_c, dropout_local_2  gt_class: 16, cname: glass_box
    # - openshape, modelnet_c, dropout_local_2  gt_class: 36, cname: tv_stand
    # - openshape, modelnet_c, dropout_local_2  gt_class: 13, cname: door
    # - openshape, modelnet_c, dropout_local_2  gt_class: 34, cname: tent
    # - openshape, modelnet_c, dropout_local_2  gt_class: 26, cname: plant
    # - openshape, modelnet_c, dropout_local_2  gt_class: 31, cname: stairs
    
    # - uni3d, omni3d, 4096 pts                 gt_class: 30, cname: calculator
    # - uni3d, omni3d, 4096 pts                 gt_class: 146, cname: pomegranate
    # - uni3d, omni3d, 4096 pts                 gt_class: 164, cname: shampoo
    # - uni3d, omni3d, 4096 pts                 gt_class: 212, cname: watermelon
    # - uni3d, omni3d, 4096 pts                 gt_class: 210, cname: watch
    # - uni3d, omni3d, 4096 pts                 gt_class: 83, cname: hair dryer
    # - uni3d, omni3d, 4096 pts                 gt_class: 203, cname: toy truck
    
    for i, (pc, target, _, rgb) in enumerate(test_loader):
        if cnames[target.item()] == args.cname:
            d = {'pc': pc, 'label': cnames[target.item()]}
            torch.save(d, f'outputs/saved_pth_tensors/{args.dataset}_{args.sonn_variant}_{args.cor_type}_{cnames[target.item()]}.pth')
            print(cnames[target.item()], ':', target.item())
            break
            