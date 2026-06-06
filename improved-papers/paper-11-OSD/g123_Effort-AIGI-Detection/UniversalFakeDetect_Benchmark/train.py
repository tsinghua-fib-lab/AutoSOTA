import os
import time
import random
from tensorboardX import SummaryWriter

from validate import validate, find_best_threshold, RealFakeDataset
from data import create_dataloader
from earlystop import EarlyStopping
from models.trainer import Trainer
from options.train_options import TrainOptions
from dataset_paths import DATASET_PATHS
import torch
import numpy as np


SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    
    set_seed()
 
    model = Trainer(opt)
    
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    with open( os.path.join(opt.checkpoints_dir, opt.name,'log.txt'), 'a') as f:
        f.write("Length of data loader: %d \n" %(len(data_loader)) )
    for epoch in range(opt.niter):
        model.save_networks( 'model_epoch_init.pth' )
        
        for i, data in enumerate(data_loader):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps) )
                with open( os.path.join(opt.checkpoints_dir, opt.name,'log.txt'), 'a') as f:
                    f.write(f"Iter time: {(time.time()-start_time)/model.total_steps}, Lr: {model.lr}, Train loss: {model.loss} at step: {model.total_steps}\n")

            if model.total_steps in [50,100,500,550,600,650,700,800,900,1000,1200,1500,2000,3000,5000,8000,10000,12000,18000,20000,23000,25000]: # save models at these iters 
                model.train()
                model.save_networks('model_iters_%s.pth' % model.total_steps)
            
            # if model.total_steps % 500 == 0:
            #     model.adjust_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.train()
            model.save_networks( 'model_epoch_%s.pth' % epoch )

        # Validation
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        model.train()

