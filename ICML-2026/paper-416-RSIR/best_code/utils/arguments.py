import argparse

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='model name')
    parser.add_argument('--dataset', '-d', type=str, default='amazon-toys', help='dataset name')
    parser.add_argument('--trainfile', '-t', type=str, default='', help='train file')
    parser.add_argument('--output_trainfile', type=str, default='_1th', help='generated training file suffix used in generation mode, e.g. _1th')
    parser.add_argument('--device', '-dev', type=str, default=5, help='device')
    
    parser.add_argument('--eval_only', type=bool, default=True, help='Set this flag to run only the special evaluation (Step 3).')
    parser.add_argument('--theta_r_path', type=str, default='saved/SASRec/amazon-toys/2025-09-13-07-51-21-546504.ckpt', help='Path to theta_r checkpoint for evaluation.')
    parser.add_argument('--neg_num', type=int, default=256, help='num of neg samplers')
    parser.add_argument('--num_layer', type=int, default=256, help='num of layer')
    parser.add_argument('--loss_fn_name', type=str, default='sampled_softmax', help='loss function')
    parser.add_argument('--mode', type=str, default='recommendation', help='mode')

    return parser
