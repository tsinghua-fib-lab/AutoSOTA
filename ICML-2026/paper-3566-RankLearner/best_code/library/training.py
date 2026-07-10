from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from data_utils import set_seed

####################################################################################

def train_propensity(model, train_loader, val_loader, device, lr=3e-4, 
                     weight_decay=1e-5, max_epochs=50, patience=5, seed=0):
    """ Expects train_loader that yields (x, t, y). """
    set_seed(seed)
    
    # init
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # early stopping
    best_state, best_val, patience_left = None, float("inf"), patience

    # loop over epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        # progress
        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for x, t, _ in pbar:

                # forward pass
                x, t = x.to(device), t.to(device)
                logits = model(x)

                # backward pass
                loss = criterion(logits, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # progress
                bs = t.size(0)
                running_loss += loss.item() * bs # track total loss
                n_train += bs # track number of samples
                pbar.set_postfix(loss=running_loss / n_train)

        # validation loop
        model.eval()
        running_loss_val, n_val = 0.0, 0
        with torch.no_grad():
            for x_val, t_val, _ in val_loader:
                x_val, t_val = x_val.to(device), t_val.to(device)
                logits_val = model(x_val)
                loss_val = criterion(logits_val, t_val)

                # progress
                bs_val = t_val.size(0)
                running_loss_val += loss_val.item() * bs_val
                n_val += bs_val
        val_loss = running_loss_val / max(n_val, 1)

        #  progress
        print(f"Epoch {epoch:02d} | val_loss={val_loss:.4f}")

        
        # early stopping
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    # reset best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_loss": best_val}


####################################################################################


def train_response(model, train_loader, val_loader, device, lr=3e-4, 
                   weight_decay=1e-5, max_epochs=50, patience=5, seed=0):
    """ Expects train_loader that yields (x, t, y). """
    set_seed(seed)

    # init
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction="mean")
    
    # early stopping
    best_state, best_val, patience_left = None, float("inf"), patience

    # loop over epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        # progress
        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for x, _, y in pbar:

                # forward pass
                x, y = x.to(device), y.to(device)
                preds = model(x)

                # backward pass
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # progress
                bs = y.size(0)
                running_loss += loss.item() * bs
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        # validation loop
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for x_val, _, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds_val = model(x_val)
                loss_val = criterion(preds_val, y_val)
                
                # progress
                bs_val = y_val.size(0)
                val_sum += loss_val.item() * bs_val
                n_val += bs_val
        val_mse = val_sum / max(n_val, 1)

        # progress
        print(f"Epoch {epoch:02d} | val_mse={val_mse:.6f}")

        #  early stopping
        if val_mse + 1e-9 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    # reset best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_loss": best_val}


####################################################################################

    
def train_cate(model, train_loader, val_loader, device, lr=3e-4, 
               weight_decay=1e-5, max_epochs=50, patience=5, seed=0):
    """ Expects train_loader that yields (x, dr). """
    set_seed(seed)

    # init
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction="mean")

    # early stopping
    best_state, best_val, patience_left = None, float("inf"), patience

    # loop over epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        # progress
        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for x, dr in pbar:

                # forward pass
                x, dr = x.to(device), dr.to(device)
                preds = model(x)

                # backward pass
                loss = criterion(preds, dr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # progress
                bs = dr.size(0)
                running_loss += loss.item() * bs
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        # validation loop
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for x_val, dr_val in val_loader:
                x_val, dr_val = x_val.to(device), dr_val.to(device)
                preds_val = model(x_val)
                loss_val = criterion(preds_val, dr_val)

                # progress
                bs_val = dr_val.size(0)
                val_sum += loss_val.item() * bs_val
                n_val += bs_val
        val_mse = val_sum / max(n_val, 1)

        # progress
        print(f"Epoch {epoch:02d} | val_mse={val_mse:.6f}")

        # early stopping
        if val_mse + 1e-9 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    # reset best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_loss": best_val}


####################################################################################


def approximate_autoc(data_loader, scoring_model, device):
    """ Expects data_loader that yields (x, dr). """

    #  collectors
    scores_list, dr_list = [], []

    # validation loop
    scoring_model.eval()
    with torch.no_grad():
        for x_val, dr_val in data_loader:
    
            # forward pass
            x_val = x_val.to(device)
            s = scoring_model(x_val).squeeze(-1)
            scores_list.append(s.cpu().numpy())
            dr_list.append(dr_val.cpu().numpy())
            
    # store
    scores = np.concatenate(scores_list, axis=0).astype(np.float64)  # [N]
    y_dr   = np.concatenate(dr_list,   axis=0).astype(np.float64)    # [N]
    N = y_dr.shape[0]
    
    # collector
    toc_vals = []
    
    # first covariance component
    A = y_dr - y_dr.mean()
    
    # second covariance component
    q_grid = np.linspace(0.05, 1.0, 21)
    for q in q_grid:
    
        # target number of instances to treat
        target = int(np.round(q * N))
        thr = np.quantile(scores, 1.0 - q)
    
        # how many match or exceed threshold
        above = scores > thr
        equal = scores == thr
        n_above = int(above.sum())
        n_equal = int(equal.sum())
    
        # handle ties
        need = max(target - n_above, 0)
        frac = 0.0 if n_equal == 0 else min(need / n_equal, 1.0)
    
        # treatment indicator
        Iq = above.astype(np.float64) + frac * equal.astype(np.float64)
    
        # probability of treatment
        pi = max(Iq.mean(), 1e-12)
    
        # second covariance component
        B = (Iq / pi)
        B = B - B.mean()
    
        # approximate toc at q
        toc_q = np.mean(A * B)
        toc_vals.append(toc_q)
    
    # AUTOC approximation
    dq = np.diff(q_grid)                
    autoc = np.sum(toc_vals[:-1] * dq)   

    return autoc


####################################################################################


def train_ranker(model, train_loader, val_loader, device, lr=3e-4, weight_decay=1e-5, 
                 max_epochs=50, patience=5, seed=0, plug_in=True, fraction_of_pairs=0.10):
    """ Expects train_loader that yields (x_i, x_j, soft, orth). """
    set_seed(seed)
    
    # init
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # early stopping
    best_state, best_val, patience_left = None, -float("inf"), patience

    # loop over epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        # fix batches
        total_batches = len(train_loader)
        max_batches = max(1, int(fraction_of_pairs * total_batches))  # at least 1 batch
    
        # progress
        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for b_idx, (x_i, x_j, soft, orth) in enumerate(pbar):

                # stop if fraction_of_pairs reached
                if b_idx >= max_batches:
                    break

                # set label
                if plug_in:
                    y = soft.to(device)
                else:
                    y = orth.to(device)

                # forward pass
                x_i, x_j = x_i.to(device), x_j.to(device)
                logits_i = model(x_i)
                logits_j = model(x_j)
                logits = logits_i - logits_j

                # backward pass
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # progress
                bs = y.size(0)
                running_loss += loss.item() * bs 
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        # validation loop
        val_autoc = approximate_autoc(val_loader, model, device)
        print(f"Epoch {epoch:02d} | val autoc={val_autoc:.6f}")

        # early stopping
        if val_autoc - 1e-4 > best_val:
            best_val = val_autoc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    # reset best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_autoc": best_val}