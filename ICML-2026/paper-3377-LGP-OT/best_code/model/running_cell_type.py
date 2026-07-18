import torch
from tqdm import tqdm
import numpy as np

from optim.loss_func import SinkhornLoss

def train(
        train_data, train_cell_types, X_train, val_data, val_cell_types, X_val, model, latent_coeff, epochs,
        batch_size, lr, P, device, train_tps, val_tps, model_path
):

    # MU_TAU, LOGVAR_TAU = [], []
    # for i, t in enumerate(range(len(train_data))):
    #     tp = train_tps[i]
    #     mu_tau_t = torch.nn.Parameter(torch.ones((train_data[i].shape[0], 1), device=device) * tp)
    #     logvar_tau_t = torch.nn.Parameter(torch.ones((train_data[i].shape[0], 1), device=device) * (-4))
    #     MU_TAU.append(mu_tau_t)
    #     LOGVAR_TAU.append(logvar_tau_t)
    PATIENCE = 50
    patience = 0
    T = np.unique(X_train[:,1].cpu().numpy()).shape[0]
    T_val = np.unique(X_val[:,1].cpu().numpy()).shape[0]
    N = np.sum([each.shape[0] for each in train_data])
    N_val = np.sum([each.shape[0] for each in val_data])
    prior_std_tau = 0.1
    kl_coeff = 1.0
    blur = 0.05
    scaling = 0.5
    loss_list = []
    params = list(model.parameters())# + MU_TAU + LOGVAR_TAU
    optimizer = torch.optim.Adam(params=params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE // 2)
    T = np.unique(X_train[:,1].cpu().numpy()).shape[0]
    dataloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size * T, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(X_val, batch_size=batch_size * T_val, shuffle=False)
    best_loss = np.inf
    best_model = None
    for e in range(epochs):
        epoch_pbar = tqdm(dataloader, desc="[ Training ] Epoch {}".format(e))
        b_count = 0
        ot_loss_sum = 0
        loss_sum = 0
        kl_sum = 0
        model.train()
        for batch_X in epoch_pbar:
            optimizer.zero_grad()
            cell_idx = [np.random.choice(np.arange(t.shape[0]), size = batch_size, replace = (t.shape[0] < batch_size)) 
                        for t in train_data]
            y = [train_data[t][cell_idx[t], :].to(device) for t in range(len(train_data))]
            cell_types_batch = [train_cell_types[t][cell_idx[t]].to(device) for t in range(len(train_data))]
            cell_types_batch = torch.cat(cell_types_batch, dim=1).flatten()

            # mu_tau = torch.cat([MU_TAU[t][cell_idx[t], :] for t in range(len(train_data))], dim=1).to(device)
            # std_tau = torch.sqrt(torch.exp(torch.cat([LOGVAR_TAU[t][cell_idx[t], :] for t in range(len(train_data))], dim=1))).to(device)
            # tau_sample = mu_tau + torch.randn_like(mu_tau) * std_tau
            batch_X[:, -1] = cell_types_batch
            # batch_X[:, -2] = tau_sample.flatten()
            batch_X = batch_X.unsqueeze(1).to(device)

            z, densities, A_samples = model.encode(batch_X)
            z = z.view(-1, T, z.shape[-1])
            # z = z + torch.randn_like(z)
            log_std, densities_log_std, A_sample_logvar = model.noise_model(batch_X)
            z = z + torch.randn_like(z) * torch.exp(log_std).view(-1, T, log_std.shape[-1])
            outputs = model.decode(z)
            # t = (torch.ones((batch_size, T), device=device) * train_tps.repeat(batch_size, 1).to(device)).view(-1, T, 1)
            # mu_tau = mu_tau.view(-1, T, 1)
            # std_tau = std_tau.view(-1, T, 1)
            # KL_tau = torch.sum(-0.5 + np.log(prior_std_tau) - torch.log(std_tau + 1e-6) +
            #                    (std_tau**2 + (mu_tau - t) ** 2) / (2 * prior_std_tau ** 2))
            KL_x = model.KL_loss(densities)
            KL_noise = model.noise_model.loss(densities_log_std)
            recon_obs = outputs.view(-1, T, outputs.shape[-1])

            ot_loss = SinkhornLoss(y, recon_obs, blur=blur, scaling=scaling, batch_size=None)

            KL_x = (KL_x + KL_noise) / N #+ KL_tau / (batch_size * T)
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
            cell_idx = [np.random.choice(np.arange(t.shape[0]), size = batch_size, replace = (t.shape[0] < batch_size)) 
                        for t in val_data]
            y = [val_data[t][cell_idx[t], :].to(device) for t in range(len(val_data))]
            cell_types_batch = [val_cell_types[t][cell_idx[t]].to(device) for t in range(len(val_data))]
            cell_types_batch = torch.cat(cell_types_batch, dim=1).flatten()
            # tau_sample = mu_tau + torch.randn_like(mu_tau) * std_tau
            batch_X[:, -1] = cell_types_batch
            batch_X = batch_X.unsqueeze(1).to(device)
            z, densities, A_samples = model.encode(batch_X)
            z = z.view(-1, T_val, z.shape[-1])
            log_std, densities_log_std, A_sample_logvar = model.noise_model(batch_X)
            z = z + torch.randn_like(z) * torch.exp(log_std).view(-1, T_val, log_std.shape[-1])
            outputs = model.decode(z)
            KL_x = model.KL_loss(densities)
            KL_noise = model.noise_model.loss(densities_log_std)
            recon_obs = outputs.view(-1, T_val, outputs.shape[-1])
            ot_loss = SinkhornLoss(val_data, recon_obs, blur=blur, scaling=scaling, batch_size=None)
            KL_x = (KL_x + KL_noise) / N_val
            val_ot_loss += ot_loss.item()
            kl_sum += KL_x.item()
            val_kl += KL_x.item()
            val_loss += ot_loss.item() + KL_x.item()
            val_count += 1
        val_loss /= val_count
        val_ot_loss /= val_count
        val_kl /= val_count
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
    outputs = []
    latent_seq = []
    for batch_X in dataloader:
        cell_idx = [np.random.choice(np.arange(t.shape[0]), size = batch_size, replace = (t.shape[0] < batch_size)) 
                        for t in train_data]
        y = [train_data[t][cell_idx[t], :].to(device) for t in range(len(train_data))]
        cell_types_batch = [train_cell_types[t][cell_idx[t]].to(device) for t in range(len(train_data))]
        cell_types_batch = torch.cat(cell_types_batch, dim=1).flatten()
        batch_X[:, -1] = cell_types_batch
        batch_X = batch_X.unsqueeze(1).to(device)
        z, densities, A_samples = model.encode(batch_X)
        z = z.view(-1, T, z.shape[-1])
        log_std, densities_log_std, A_sample_logvar = model.noise_model(batch_X)
        z = z + torch.randn_like(z) * torch.exp(log_std).view(-1, T, log_std.shape[-1])
        output = model.decode(z)
        latent_seq.append(z.view(-1, T, z.shape[-1]).cpu().detach().numpy())
        outputs.append(output.view(-1, T, output.shape[-1]).cpu().detach().numpy())
    recon_obs = np.concatenate(outputs, axis=0)  # (# cells, # tps, # genes)
    latent_seq = np.concatenate(latent_seq, axis=0)  # (# cells, # tps, # latent_dim)

    
    return model, loss_list, recon_obs, latent_seq#, log_var


def predict(model, traj_cell_types, X_test, device, tps):
    T = np.unique(X_test[:,1].cpu().numpy()).shape[0]
    model.eval()
    dataloader = torch.utils.data.DataLoader(X_test, batch_size=2000 * T, shuffle=False)
    outputs = []
    latent_seq = []
    for batch_X in dataloader:
        cell_idx = [np.random.choice(np.arange(t.shape[0]), size = 2000, replace = (t.shape[0] < 2000)) 
                        for t in traj_cell_types]
        cell_types_batch = [traj_cell_types[t][cell_idx[t]].to(device) for t in range(len(traj_cell_types))]
        cell_types_batch = torch.cat(cell_types_batch, dim=1).flatten()
        batch_X[:, -1] = cell_types_batch

        batch_X = batch_X.unsqueeze(1).to(device)
        z, densities, A_samples = model.encode(batch_X, stochastic_flag = False)
        z = z.view(-1, T, z.shape[-1])
        log_std, densities_log_std, A_sample_logvar = model.noise_model(batch_X, stochastic_flag = False)
        std = (torch.exp(log_std)).view(-1, T, log_std.shape[-1])
        z = z + torch.randn_like(z) * std
        output = model.decode(z)
        outputs.append(output.view(-1, T, output.shape[-1]).cpu().detach().numpy())
        latent_seq.append(z.view(-1, T, z.shape[-1]).cpu().detach().numpy())
    all_pred_data = np.concatenate(outputs, axis=0)   # (# cells, # tps, # genes)
    latent_seq = np.concatenate(latent_seq, axis=0) 
    return all_pred_data, latent_seq