import time
from torch.cuda.amp import autocast, GradScaler
import wandb
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.swa_utils import update_bn

from .util import *
from .dataset import *
from .models import getModel
from .optim import RangerLars
from .loss import rank_loss
from .augmentation import get_tranform_list


def training_loop(train_df, Config, synthetic=None):
    if Config.do_advance_trans:
        Config = get_tranform_list(Config)
    if Config.use_wandb:
        wandb.login(key=get_key(Config.wandb_key_path))
        wandb.init(project=Config.wandb_project, name=Config.model_version + Config.wandb_post,
                   config=class2dict(Config), group=Config.model_module, job_type=Config.model_version)

    folds_val_score = []
    for fold in range(5):
        if Config.model_module == "M3D":
            Config.fold = fold
        print('Fold: ', fold)
        if fold not in Config.train_folds:
            print("skip")
            continue
        best_valid_score = run_fold(fold, train_df.copy(), Config, synthetic=synthetic)
        folds_val_score.append(best_valid_score)
    print('folds score:', folds_val_score)
    print("Avg: {:.5f}".format(np.mean(folds_val_score)))
    print("Std: {:.5f}".format(np.std(folds_val_score)))
    if Config.use_wandb:
        wandb.finish()


def run_fold(fold, original_train_df, Config,
             swa_start_step=None, swa_start_epoch=None, synthetic=None,
             **kwargs):
    train_df = generate_PL(fold, original_train_df.copy(), Config)
    train_index, valid_index = train_df.query(f"fold!={fold}").index, train_df.query(
        f"fold=={fold}").index  # fold means fold_valid
    train_X, valid_X = train_df.loc[train_index], train_df.loc[valid_index]
    valid_labels = train_df.loc[valid_index, "target"].values
    oof = pd.DataFrame()
    oof['id'] = train_df.loc[valid_index, 'id']
    oof['id'] = valid_X['id'].values.copy()
    oof = oof.reset_index()
    oof['preds'] = valid_labels
    oof.to_csv(f'{Config.model_output_folder}/Fold_{fold}_oof_pred.csv')

    print('training data samples, val data samples: ', len(train_X), len(valid_X))
    train_data_retriever = DataRetriever(train_X["file_path"].values, train_X["target"].values,
                                         synthetic=synthetic, Config=Config)
    valid_data_retriever = DataRetrieverTest(valid_X["file_path"].values, valid_X["target"].values, Config=Config)

    train_loader = DataLoader(train_data_retriever,
                              batch_size=Config.batch_size,
                              shuffle=True,
                              num_workers=Config.num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_data_retriever,
                              batch_size=Config.batch_size * 2,
                              shuffle=False,
                              num_workers=Config.num_workers, pin_memory=True, drop_last=False)

    model = getModel(Config)
    model.to(Config.device)
    if Config.use_dp and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if Config.optim == 'RangerLars':
        optimizer = RangerLars(model.parameters(), lr=Config.lr, eps=1e-04, weight_decay=Config.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=Config.lr, eps=1e-08, weight_decay=Config.weight_decay,
                          amsgrad=False)
    scheduler = get_scheduler(optimizer, len(train_X), Config)
    swa_model, swa_scheduler = None, None
    best_valid_score = -np.inf
    if Config.checkpoint_folder is not None:
        print(f"Load Checkpoint from folder: {Config.checkpoint_folder}")
        checkpoint = torch.load(f'{Config.checkpoint_folder}/Fold_{fold}_best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_valid_score = float(checkpoint['best_valid_score'])

    if Config.use_swa:
        print("Use SWA")
        swa_model, swa_scheduler = get_swa(model, optimizer, Config.epochs, Config.swa_start_step_epoch, Config.swa_lr,
                                           len(train_X), Config.batch_size)

    criterion = rank_loss if Config.crit == 'rank' else F.binary_cross_entropy_with_logits

    trainer = Trainer(model, optimizer, criterion, scheduler, valid_labels,
                      best_valid_score, fold, Config,
                      swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start_step=swa_start_step,
                      swa_start_epoch=swa_start_epoch)

    trainer.fit(
        epochs=Config.epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_path=f'{Config.model_output_folder}/Fold_{fold}_',
    )
    return trainer.best_valid_score


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, valid_labels,
                 best_valid_score, fold, Config, mixed_criterion=None,
                 swa_model=None, swa_scheduler=None, swa_start_step=None,
                 swa_start_epoch=None, **kwargs):
        self.model = model
        self.device = Config.device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.best_valid_score = best_valid_score
        self.valid_labels = valid_labels
        self.fold = fold

        self.use_gradScaler = Config.use_gradScaler
        self.use_autocast = Config.use_autocast
        self.gradient_accumulation_steps = Config.gradient_accumulation_steps

        self.use_mixup = Config.use_mixup
        self.mixup_alpha = Config.mixup_alpha
        self.mixed_criterion = mixed_criterion

        # swa
        self.swa_model = swa_model
        self.swa_scheduler = swa_scheduler
        self.swa_start_epoch = swa_start_epoch
        self.swa_start_step = swa_start_step
        self.step = 0  # for swa

        # log
        self.print_num_steps = Config.print_num_steps
        self.use_wandb = Config.use_wandb

    def fit(self, epochs, train_loader, valid_loader, save_path):
        train_losses = []
        valid_losses = []
        for n_epoch in range(epochs):
            start_time = time.time()
            print('Epoch: ', n_epoch)
            train_loss = self.train_epoch(train_loader)
            valid_loss, valid_preds = self.valid_epoch(valid_loader, self.model)

            if self.swa_model is not None:
                if n_epoch >= self.swa_start_epoch:
                    print(f"Epoch {n_epoch}, update swa model")
                    self.swa_model.update_parameters(self.model)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            valid_score = get_score(self.valid_labels, valid_preds)

            if self.best_valid_score < valid_score:
                self.best_valid_score = valid_score
                self.save_model(n_epoch, save_path + f'best_model.pth', valid_preds)

            print('train_loss: ', train_loss)
            print('valid_loss: ', valid_loss)
            print('valid_score: ', valid_score)
            print('best_valid_score: ', self.best_valid_score)
            print('time used: ', time.time() - start_time)
            if self.use_wandb:
                wandb.log({f"[fold{self.fold}] epoch": n_epoch + 1,
                           f"[fold{self.fold}] avg_train_loss": train_loss,
                           f"[fold{self.fold}] avg_val_loss": valid_loss,
                           f"[fold{self.fold}] val_score": valid_score})
                # save swa
        if self.swa_model is not None:
            update_bn(train_loader, self.swa_model, device=self.device)
            valid_loss_swa, valid_preds_swa = self.valid_epoch(valid_loader, self.swa_model)
            valid_score_swa = get_score(self.valid_labels, valid_preds_swa)
            print("SWA: Valid Loss {:.5f}, Valid Score {:.5f}".format(valid_loss_swa, valid_score_swa))
            if self.use_wandb:
                wandb.log({f"[fold{self.fold}] avg_val_loss_swa": valid_loss_swa,
                           f"[fold{self.fold}] val_score_swa": valid_score_swa})
                # update batch normalization
            save_dict = {
                "swa_model_state_dict": self.swa_model.state_dict(),
                "swa_scheduler": self.swa_scheduler.state_dict(),
                "valid_loss_swa": valid_loss_swa,
                "valid_score_swa": valid_score_swa,
            }
            torch.save(save_dict, save_path + f'swa_model.pth')

    def train_epoch(self, train_loader):
        if self.use_gradScaler:
            scaler = GradScaler()

        self.model.train()
        losses = []
        train_loss = 0
        for step, batch in enumerate(train_loader, 1):
            self.step += 1
            self.optimizer.zero_grad()
            X = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            if self.use_mixup:
                (X_mix, targets_a, targets_b, lam) = mixup_data(X, targets, self.mixup_alpha)
                with autocast(enabled=self.use_autocast):
                    outputs = self.model(X_mix).squeeze()
                    loss = self.mixed_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                with autocast(enabled=self.use_autocast):
                    outputs = self.model(X).squeeze()
                    loss = self.criterion(outputs[outputs == outputs], targets[outputs == outputs])

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if step % self.gradient_accumulation_steps == 0:
                scaler.step(self.optimizer)
                scaler.update()

            if do_swa_scheduler(self.step, self.swa_scheduler, self.swa_start_step):
                self.swa_scheduler.step()
                lr2 = self.swa_scheduler.get_last_lr()[0]
            else:
                self.scheduler.step()
                lr2 = self.scheduler.get_last_lr()[0]

            loss2 = loss.detach()

            if self.use_wandb:
                wandb.log({f"[fold{self.fold}] loss": loss2,
                           f"[fold{self.fold}] lr": lr2})

            losses.append(loss2)
            train_loss += loss2
            if step % self.print_num_steps == 0:
                train_loss = train_loss.item()
                print(f'[{step}/{len(train_loader)}] ',
                      f'avg loss: ', train_loss / step,
                      f'inst loss: ', loss2.item())

        return train_loss / step

    def valid_epoch(self, valid_loader, model):
        model.eval()
        valid_loss = []
        preds = []
        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                outputs = model(X).squeeze()
                loss = self.criterion(outputs[outputs == outputs], targets[outputs == outputs])
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                valid_loss.append(loss.detach().item())
                preds.append(outputs.sigmoid().to('cpu').numpy())
        predictions = np.concatenate(preds)
        return np.mean(valid_loss), predictions

    def save_model(self, n_epoch, save_path, valid_preds):
        print("Save Model")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
                'scheduler': self.scheduler.state_dict(),
                'valid_preds': valid_preds,
            },
            save_path,
        )
