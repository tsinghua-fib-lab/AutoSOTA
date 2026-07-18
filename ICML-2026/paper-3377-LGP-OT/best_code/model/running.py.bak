import os
import time
import torch
from tqdm import tqdm
import numpy as np

from optim.loss_func import SinkhornLoss, MSELoss

def train(
        train_data, X_train, val_data, X_val, model, latent_coeff, epochs,
        batch_size, lr, P, device, train_tps, val_tps, model_path
):

    PATIENCE = 50
    patience = 0
    T = np.unique(X_train[:,1].cpu().numpy()).shape[0]
    T_val = np.unique(X_val[:,1].cpu().numpy()).shape[0]
    N = np.sum([each.shape[0] for each in train_data])
    N_val = np.sum([each.shape[0] for each in val_data])

    kl_coeff = 1.0
    blur = 0.05
    scaling = 0.5
    loss_list = []
    if model.noise_model is None:
        model.noise_log_std = torch.nn.Parameter(torch.log(torch.ones(model.latent_dim, device=device)))
    params = list(model.parameters())# + [log_var]
    optimizer = torch.optim.Adam(params=params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE // 2)
    T = np.unique(X_train[:,1].cpu().numpy()).shape[0]
    dataloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size * T, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(X_val, batch_size=batch_size * T_val, shuffle=False)
    best_loss = np.inf
    best_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()
    for e in range(epochs):
        epoch_pbar = tqdm(dataloader, desc="[ Training ] Epoch {}".format(e))
        b_count = 0
        ot_loss_sum = 0
        loss_sum = 0
        kl_sum = 0
        model.train()
        for batch_X in epoch_pbar:
            optimizer.zero_grad()
            batch_X = batch_X.unsqueeze(1).to(device)

            z, densities, A_samples = model.encode(batch_X)
            z = z.view(-1, T, z.shape[-1])

            if model.noise_model is not None:
                log_std, densities_log_std, A_sample_logvar = model.noise_model(batch_X)
            else:
                log_std = model.noise_log_std.expand_as(z.detach())
            z = z + torch.randn_like(z) * torch.exp(log_std).view(-1, T, log_std.shape[-1])
            outputs = model.decode(z)
            KL_x = model.KL_loss(densities)
            if model.noise_model is not None:
                KL_noise = model.noise_model.loss(densities_log_std)
            else:
                KL_noise = 0.0
            recon_obs = outputs.view(-1, T, outputs.shape[-1])

            ot_loss = SinkhornLoss(train_data, recon_obs, blur=blur, scaling=scaling, batch_size=200)

            KL_x = kl_coeff * (KL_x + KL_noise) / N
            loss = ot_loss + KL_x
            ot_loss_sum += ot_loss.item()
            kl_sum += KL_x.item()
            loss_sum += loss.item()
            b_count += 1
            epoch_pbar.set_postfix({"Loss": "{:.3f}| OT={:.3f} | KL={:.3f}".format(loss_sum / b_count, ot_loss_sum / b_count, kl_sum / b_count)})
            loss.backward()
            optimizer.step()
            loss_list.append([loss.item(), ot_loss.item(), KL_x.item()])
        
        # Validation
        model.eval()
        val_loss = 0
        val_ot_loss = 0
        val_kl = 0
        val_count = 0
        for batch_X in val_dataloader:
            batch_X = batch_X.unsqueeze(1).to(device)
            z, densities, A_samples = model.encode(batch_X)
            z = z.view(-1, T_val, z.shape[-1])
            if model.noise_model is not None:
                log_std, densities_log_std, A_sample_logvar = model.noise_model(batch_X)
            else:
                log_std = model.noise_log_std.expand_as(z)
            z = z + torch.randn_like(z) * torch.exp(log_std).view(-1, T_val, log_std.shape[-1])
            outputs = model.decode(z)
            KL_x = model.KL_loss(densities)
            recon_obs = outputs.view(-1, T_val, outputs.shape[-1])
            ot_loss = SinkhornLoss(val_data, recon_obs, blur=blur, scaling=scaling, batch_size=None)
            KL_x = kl_coeff * KL_x / N_val
            val_ot_loss += ot_loss.item()
            kl_sum += KL_x.item()
            val_kl += KL_x.item()
            val_loss += ot_loss.item() + KL_x.item()
            val_count += 1
        val_loss /= val_count
        val_ot_loss /= val_count
        val_kl /= val_count
        print("[ Validation ] Loss: {:.3f} | OT: {:.3f} | KL: {:.3f}".format(val_loss, val_ot_loss, val_kl))
        if best_loss - val_ot_loss > 0.1:
            best_loss = val_ot_loss
            torch.save(model.state_dict(), model_path)
            patience = 0
        else:
            patience += 1
            if patience == PATIENCE:
                break
        scheduler.step(val_ot_loss)
        print(scheduler.get_last_lr())
    if torch.cuda.is_available():
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print("Training time: {:.2f} seconds".format(end_time - start_time))
        print("Peak memory: {:.2f} GB".format(peak_memory))
        if not os.path.exists("LGPOT_training_stats.csv"):
            with open("LGPOT_training_stats.csv", "w") as f:
                f.write("epoch,total_time,peak_memory,trainable_params,path\n")
        with open("LGPOT_training_stats.csv", "a") as f:
            f.write("{},{:.2f},{:.2f},{},{}\n".format(
                e + 1, end_time - start_time, peak_memory,
                sum(p.numel() for p in model.parameters() if p.requires_grad),
                model_path
            ))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    outputs = []
    latent_seq = []
    for batch_X in dataloader:
        batch_X = batch_X.unsqueeze(1).to(device)
        z, densities, A_samples = model.encode(batch_X)
        z = z.view(-1, T, z.shape[-1])
        if model.noise_model is not None:
            log_std, densities_log_std, A_sample_logvar = model.noise_model(batch_X)
        else:
            log_std = model.noise_log_std.expand_as(z)
        z = z + torch.randn_like(z) * torch.exp(log_std).view(-1, T, log_std.shape[-1])
        output = model.decode(z)
        latent_seq.append(z.view(-1, T, z.shape[-1]).cpu().detach().numpy())
        outputs.append(output.view(-1, T, output.shape[-1]).cpu().detach().numpy())
    recon_obs = np.concatenate(outputs, axis=0)  # (# cells, # tps, # genes)
    latent_seq = np.concatenate(latent_seq, axis=0)  # (# cells, # tps, # latent_dim)

    
    return model, loss_list, recon_obs, latent_seq#, log_var


def predict(model, X_test, device, tps):
    T = np.unique(X_test[:,1].cpu().numpy()).shape[0]
    model.eval()
    dataloader = torch.utils.data.DataLoader(X_test, batch_size=2000 * T, shuffle=False)
    outputs = []
    latent_seq = []
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_time = time.time()
        for batch_X in dataloader:
            batch_X = batch_X.unsqueeze(1).to(device)
            z, densities, A_samples = model.encode(batch_X, stochastic_flag = False)
            z = z.view(-1, T, z.shape[-1])
            if model.noise_model is not None:
                log_std, densities_log_std, A_sample_logvar = model.noise_model(batch_X, stochastic_flag = False)
            else:
                log_std = torch.log(torch.ones(z.shape, device=device) * 0.1)
            std = (torch.exp(log_std)).view(-1, T, log_std.shape[-1])
            z = z + torch.randn_like(z) * std
            output = model.decode(z)
            outputs.append(output.view(-1, T, output.shape[-1]).cpu().detach().numpy())
            latent_seq.append(z.view(-1, T, z.shape[-1]).cpu().detach().numpy())
    if torch.cuda.is_available():
        end_time = time.time()
        print("Inference time: {:.2f} seconds".format(end_time - start_time))
        if not os.path.exists("LGPOT_inference_stats.csv"):
            with open("LGPOT_inference_stats.csv", "w") as f:
                f.write("inference_time\n")
        with open("LGPOT_inference_stats.csv", "a") as f:
            f.write("{:.2f}\n".format(
                end_time - start_time
            ))
    all_pred_data = np.concatenate(outputs, axis=0)   # (# cells, # tps, # genes)
    latent_seq = np.concatenate(latent_seq, axis=0) 
    return all_pred_data, latent_seq