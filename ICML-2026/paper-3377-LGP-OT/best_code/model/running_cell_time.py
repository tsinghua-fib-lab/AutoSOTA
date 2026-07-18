import torch
from tqdm import tqdm
import numpy as np

from optim.loss_func import SinkhornLoss

def train(
        train_data, val_data, model, latent_coeff, epochs,
        batch_size, lr, P, device, train_tps, val_tps, model_path, mt, st
):

    MU_TAU, LOGVAR_TAU = [], []
    for i, t in enumerate(range(len(train_data))):
        tp = train_tps[i]
        mu_tau_t = torch.nn.Parameter(torch.ones((train_data[i].shape[0], 1), device=device) * tp)
        logvar_tau_t = torch.nn.Parameter(torch.ones((train_data[i].shape[0], 1), device=device) * (-4))
        MU_TAU.append(mu_tau_t)
        LOGVAR_TAU.append(logvar_tau_t)
    PATIENCE = 50
    patience = 0
    N = np.sum([each.shape[0] for each in train_data])
    T = len(train_data)
    N_val = np.sum([each.shape[0] for each in val_data])
    T_val = len(val_data)
    prior_std_tau = 0.1
    kl_coeff = 1.0
    blur = 0.05
    scaling = 0.5
    loss_list = []
    params = list(model.parameters()) + MU_TAU + LOGVAR_TAU
    optimizer = torch.optim.Adam(params=params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE // 2)
    best_loss = np.inf
    best_model = None
    for e in range(epochs):
        epoch_pbar = tqdm(range(10), desc="[ Epoch {} ]".format(e + 1))
        ot_loss_sum = 0
        loss_sum = 0
        kl_sum = 0
        b_count = 0
        model.train()
        for i in epoch_pbar:
            optimizer.zero_grad()
            cell_idx = [np.random.choice(np.arange(t_true.shape[0]), size = batch_size, replace = (t_true.shape[0] < batch_size)) 
                        for t_true in train_data]
            y = [train_data[t][cell_idx[t], :].to(device) for t in range(len(train_data))]

            mu_tau, std_tau = [], []
            for t in range(len(y)):
                mu_tau_t, std_tau_t = MU_TAU[t][cell_idx[t], :].to(device), torch.sqrt(torch.exp(LOGVAR_TAU[t][cell_idx[t], :].to(device)))
                mu_tau.append(mu_tau_t)
                std_tau.append(std_tau_t)
            mu_tau = torch.cat(mu_tau, dim=1)
            std_tau = torch.cat(std_tau, dim=1)
            tau = mu_tau + torch.randn_like(std_tau) * std_tau
            tau_in = tau.view(-1, 1)
            z, densities, A_samples = model.encode(tau_in.unsqueeze(1))  # (N, 1, latent_dim)
            log_std, densities_log_std, A_sample_logvar = model.noise_model(tau_in.unsqueeze(1))  # (N, 1, latent_dim)
            z = z + torch.randn_like(z) * torch.exp(log_std)
            z = z.squeeze(1)
  
            outputs = model.decode(z)
            KL_x = model.KL_loss(densities)
            t = (torch.ones((batch_size, T), device=device) * train_tps.repeat(batch_size, 1).to(device)).view(-1, T, 1)
            mu_tau = mu_tau.view(-1, T, 1)
            std_tau = std_tau.view(-1, T, 1)
            KL_tau = torch.sum(-0.5 + np.log(prior_std_tau) - torch.log(std_tau + 1e-6) +
                               (std_tau**2 + (mu_tau - t) ** 2) / (2 * prior_std_tau ** 2))
            KL_noise = model.noise_model.loss(densities_log_std)
            KL = KL_tau / (T * batch_size) + (KL_x + KL_noise) / N
            KL = latent_coeff * KL
            if torch.isnan(KL):
                print("KL is nan!")
                continue
            recon_obs = outputs.view(-1, T, outputs.shape[-1])
            ot_loss = SinkhornLoss(y, recon_obs, blur=blur, scaling=scaling, batch_size=None)
            loss = ot_loss + KL

            ot_loss_sum += ot_loss.item()
            kl_sum += KL.item()
            loss_sum += loss.item()
            b_count += 1
            epoch_pbar.set_postfix({"Loss": "{:.3f}| OT={:.3f} | KL={:.3f}".format(loss_sum / b_count, ot_loss_sum / b_count, kl_sum / b_count)})
            loss.backward()
            optimizer.step()
            loss_list.append([loss.item(), ot_loss.item(), KL.item()])
        
        # Validation
        model.eval()
        batch_size_val = val_data[0].shape[0]
        val_loss = 0
        val_ot_loss = 0
        val_kl = 0
        
        y_val = val_data
        mu_tau, std_tau = [], []
        for t in range(len(y_val)):
            mu_tau_t, std_tau_t = torch.ones((y_val[t].shape[0], 1), device=device) * val_tps[t], \
                                 torch.zeros((y_val[t].shape[0], 1), device=device)
            mu_tau.append(mu_tau_t)
            std_tau.append(std_tau_t)
        mu_tau = torch.cat(mu_tau, dim=1)
        std_tau = torch.cat(std_tau, dim=1)

        tau_in = mu_tau.view(-1, 1)
        z, densities, A_samples = model.encode(tau_in.unsqueeze(1))  # (N, latent_dim)
        log_std, densities_log_std, A_sample_logvar = model.noise_model(tau_in.unsqueeze(1))  # (N, 1, latent_dim)
        z = z + torch.randn_like(z) * torch.exp(log_std)
        z = z.squeeze(1)
        outputs = model.decode(z)
        KL_x = model.KL_loss(densities)
        t = (torch.ones((batch_size_val, T_val), device=device) * val_tps.repeat(batch_size_val, 1).to(device)).view(-1, T_val, 1)
        mu_tau = mu_tau.view(-1, T_val, 1)
        std_tau = std_tau.view(-1, T_val, 1)
        KL_tau = torch.sum(-0.5 + np.log(prior_std_tau) - torch.log(std_tau + 1e-6) +
                            (std_tau**2 + (mu_tau - t) ** 2) / (2 * prior_std_tau ** 2))
        KL_noise = model.noise_model.loss(densities_log_std)
        KL = KL_tau / (T_val * batch_size_val) + (KL_x + KL_noise) / N_val
        KL = kl_coeff * KL
        recon_obs = outputs.view(-1, T_val, outputs.shape[-1])
        ot_loss = SinkhornLoss(y_val, recon_obs, blur=blur, scaling=scaling, batch_size=None)
        val_ot_loss = ot_loss.item()
        val_kl = KL.item()
        val_loss = val_ot_loss + val_kl 

        print("[ Validation ] Loss: {:.3f} | OT: {:.3f} | KL: {:.3f}".format(val_loss, val_ot_loss, val_kl))
        if best_loss - val_ot_loss > 1e-2:
            best_loss = val_ot_loss
            torch.save(model.state_dict(), model_path)
            patience = 0
        else:
            patience += 1
            if patience == PATIENCE:
                break
        scheduler.step(val_ot_loss)
        print(scheduler.get_last_lr())

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    mu_tau, std_tau = [], []
    tps = []
    for t in range(len(train_data)):
        tps.append(np.ones(train_data[t].shape[0], dtype=np.float32) * train_tps[t].cpu().detach().numpy())
        mu_tau_t, std_tau_t = MU_TAU[t].to(device), torch.sqrt(torch.exp(LOGVAR_TAU[t].to(device)))
        mu_tau.append(mu_tau_t)
        std_tau.append(std_tau_t)
    mu_tau = torch.cat(mu_tau, dim=0)
    std_tau = torch.cat(std_tau, dim=0)
    tau_in = mu_tau.view(-1, 1)
    z, densities, A_samples = model.encode(tau_in.unsqueeze(1))  
    log_std, densities_log_std, A_sample_logvar = model.noise_model(tau_in.unsqueeze(1))
    z = z + torch.randn_like(z) * torch.exp(log_std)
    z = z.squeeze(1)
    outputs = model.decode(z)
    recon_obs = outputs
    latent_seq = z.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    tps = np.concatenate(tps, axis=0)
    pseudo_tps = mu_tau.cpu().detach().numpy()

    return model, loss_list, recon_obs, latent_seq, tps, pseudo_tps


def predict(model, device, tps, n_samples=2000, prior_std_tau=1, R=1, ptps=None):
    model.eval()

    outputs = []
    latent_seq = []
    i = 0
    with torch.no_grad():
        for t in tqdm(tps, desc="[ Predicting ]"):
            t_val = torch.tensor(t, device=device, dtype=torch.float32)
            t_val = t_val.unsqueeze(0).repeat(n_samples, R)  # (n_samples, 1)
            i += 1
            z, _, _ = model.covariate_modules[0](t_val.unsqueeze(1))
            log_std, densities_log_std, A_sample_logvar = model.noise_model(t_val.unsqueeze(1))
            z = z + torch.randn_like(z) * torch.exp(log_std)
            y_hat = model.decode(z)
            outputs.append(y_hat.cpu().detach().numpy())
            latent_seq.append(z.cpu().detach().numpy())
    all_pred_data = np.concatenate(outputs, axis=1)   # (# cells, # tps, # genes)
    latent_seq = np.concatenate(latent_seq, axis=1) 
    return all_pred_data, latent_seq