from email import utils
import torch
import time
from engine import trainer, record_metric
import numpy as np
import utils.util as util
from data.DataHandler_st import DataHandler
from Yaml2Params import args, logger
import os
import random
from utils.general_utils import send_mail

seed = 777

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # tf.random.set_seed(seed)
    torch.backends.cudnn.enabled = True

def train():
    set_seed(seed)
    handler = DataHandler()
    tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w= handler.get_dataloader()
    # process adj for teacher model: 
    if args.model == 'stgcn': 
        adj_mx = util.process_adj(sp_adj, args.adj_type)
        supports = torch.Tensor(adj_mx[0])
    
    engine = trainer(scaler, sp_adj, sp_adj_w, supports = supports)
    tra_val_metric = dict()
    if args.testonly is not True:
        logger.info('start training .....')
        loss_tm1 = loss_t = np.ones(3) #(1.0, 1.0, 1.0)
        for epoch in range(1, args.max_epoch+1):
            print('*'*20, 'Training Process', '*'*20)
            t1 = time.time()
            tra_val_metric= engine.train_t(epoch, tra_loader, tra_val_metric)
            t2 = time.time()
            print('*'*20, 'Validating Process', '*'*20)
            tra_val_metric, stopFlg = engine.validation(epoch, val_loader, tra_val_metric)
            tra_val_metric = record_metric(tra_val_metric, [t2 - t1], ['cost time'])
            logger.info(tra_val_metric)
            if stopFlg:
                break
        logger.info('start testing .....')
        engine.test(tst_loader)        

    else:
        logger.info('start testing .....')
        engine.test(tst_loader)   


if __name__ == "__main__":
    train()

