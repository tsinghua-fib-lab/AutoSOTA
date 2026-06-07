import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='MU')

    parser.add_argument('--seed', type=int, default=2024, help='Random Seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')

    parser.add_argument('-m', '--method', type=str, default='minimax', help='Method: (original, retrain, finetune, unsir, minimax)')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Domain Adaptation Epochs')
    parser.add_argument('-se', '--source_epochs', type=int, default=10, help='Source Model Training Epochs')
    parser.add_argument('-i', '--iter', type=int, default=100, help='Iterations per Epoch')

    parser.add_argument('-d', '--dataset', type=str, default='OfficeHome', help='Dataset: (OfficeHome, DomainNet, Office31)')
    parser.add_argument('-s', '--source', type=str, default='Art', help='Source Domain')
    parser.add_argument('-t', '--target', type=str, default='Product', help='Target Domain')

    parser.add_argument('-ft', '--fast_train', action='store_true', help='Only Validate Last Epoch')
    parser.add_argument('-fc', '--forget_classes', type=str, default='1,2,3', help='Forget Classes (Comma Seperated)')
    
    parser.add_argument('-ma', '--m_alpha', type=float, default=10.0, help='Minimax alpha')
    parser.add_argument('-ms', '--m_samples', type=int, default=4, help='Minimax number of adversarial samples')
    parser.add_argument('-mdi', '--m_dont_init', action='store_true', help='Minimax dont initialize adversarial sample')
    parser.add_argument('-mu', '--m_update', type=int, default=1, help='Minimax update interval')
    parser.add_argument('-ml', '--m_label', type=str, default='rescaled', help='Minimax type of label: (uniform, rescaled, random)')
    
    return parser.parse_args()