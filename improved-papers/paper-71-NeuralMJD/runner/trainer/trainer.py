import torch

from runner.trainer.trainer_utils import create_logging_dict, print_epoch_learning_status, check_best_model, save_ckpt_model

from runner.trainer.trainer_mjd import move_forward_one_epoch_mjd

def go_training(model, optimizer, scheduler, ema_helper, train_dl, val_dl, test_dl, config, dist_helper, writer) -> None:
    """
    Run the full training loop over `max_epoch`, with periodic evaluation and checkpointing.

    This function handles:
    - Per-epoch training with gradient step and scheduler update
    - Optional EMA evaluation
    - Validation/test evaluation at intervals
    - Best-model tracking and checkpoint saving
    """

    """Initialization"""
    lowest_loss = {"epoch": -1, "loss": float('inf')}

    save_interval = config.train.save_interval
    plot_save_dir = config.plot_save_dir
    model.config = config.model
    huber_delta = config.train.huber_delta
    flag_skip_ckpt_saving = config.train.skip_ckpt_saving
    dataset_nm = config.dataset.name

    """Go training"""
    for epoch in range(config.train.max_epoch):
        """Initialization"""
        logger_dict = create_logging_dict(epoch)
        if dist_helper.is_ddp:
            train_dl.sampler.set_epoch(epoch)
            val_dl.sampler.set_epoch(epoch)
            test_dl.sampler.set_epoch(epoch)

        """Start learning"""
        # training
        model.train()
        logger_dict['lr'] = optimizer.param_groups[0]["lr"]
        move_forward_one_epoch_mjd(model, optimizer, ema_helper, dist_helper, train_dl, logger_dict, 'train', dataset_nm, plot_save_dir, huber_delta, writer)
        # post epoch update
        scheduler.step()
        logger_dict['lr'] = optimizer.param_groups[0]["lr"]

        
        # testing
        if (epoch % save_interval == save_interval - 1 or epoch == 0):

            model.eval()
            # Note: EMA coefficients can be tuned here.
            if ema_helper is not None:
                eval_model = ema_helper[0].ema_model
            else:
                eval_model = model
            eval_model.eval()
            eval_model = eval_model.to(dist_helper.device)

            eval_model.config = config.model  # restore the model config
            with torch.no_grad():
                move_forward_one_epoch_mjd(eval_model, optimizer, ema_helper, dist_helper, val_dl, logger_dict, 'val', dataset_nm, plot_save_dir, huber_delta, writer)
                move_forward_one_epoch_mjd(eval_model, optimizer, ema_helper, dist_helper, test_dl, logger_dict, 'test', dataset_nm, plot_save_dir, huber_delta, writer)
            
            """Network weight saving"""
            # check best model
            check_best_model(model, ema_helper, logger_dict, lowest_loss, save_interval, config, dist_helper)

            # save checkpoint model
            if not flag_skip_ckpt_saving:
                save_ckpt_model(model, ema_helper, logger_dict, config, dist_helper)

        # end of epoch
        torch.cuda.empty_cache()

        dist_helper.ddp_sync()

        # show the training and testing status
        print_epoch_learning_status(logger_dict, writer, adjust_r2='sp500' in config.dataset.name)
