import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
    parser.add_argument('--l2_reg', default=0.0001, type=float, help='regularizer weight')
    parser.add_argument('--beta', default=70, type=float, help='ib loss weight') #70
    parser.add_argument('--alpha', default=0.05, type=float, help='contrast loss weight') # 0.002 
    parser.add_argument('--sigma', default=0.25, type=float, help='ib weight') # 0.25
    parser.add_argument('--edge_bias', default= 0, type=float, help='drop edge bias')
    parser.add_argument('--temperature2', default=0.05, type=float, help='constrastive temperature')  # 0.05
    parser.add_argument('--epoch', default=1000, type=int, help='number of epochs')
    parser.add_argument('--decay', default=0, type=float, help='weight decay rate') #1e-7
    parser.add_argument('--save_path', default='tem_lightgcnssl_yelpsparse',
                        help='file name to save model and training record')
    parser.add_argument('--latdim', default=64, type=int, help='embedding size')
    parser.add_argument('--temperature1', default=0.2, type=float, help='concrete relaxation temperature') 




    parser.add_argument('--gcn_layer', default=3, type=int, help='number of gcn layers')
    parser.add_argument('--mask_gcn_layer', default=1, type=int, help='number of gcn layers')
    parser.add_argument('--layer_dropout', default=0.1, type=float, help='stochastic layer dropout rate for GCN')
    parser.add_argument('--init_type', default="xa_uni", type=str, help='init type')  # norm xa_uni







    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--topk', default=20, type=int, help='K of top K')
    parser.add_argument('--data', default='Tmall', type=str, help='name of dataset') # Taobao  Beibei Tmall
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
    parser.add_argument('--device', default='gpu', type=str, help='gpu or cpu')
    parser.add_argument('--earlyStop', default=20, type=int, help='step of early stop')

    return parser.parse_args()

args = ParseArgs()
