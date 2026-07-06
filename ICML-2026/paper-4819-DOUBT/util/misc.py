import numpy as np
import torch
import re

def get_ans(ans_list, choices):
    result = []
    for ans in ans_list:
        match_single = re.search(r'\d', ans)
        if match_single:
            if (int(match_single.group(0)) < len(choices)) and (int(match_single.group(0)) >= 0):
                result.append(f"The answer of this question is {choices[int(match_single.group(0))]}.")
            else:
                result.append(ans)
        else:
            result.append(ans)
    return result

def kappa_from_Rbar_torch(R_bar, d):
    eps = 1e-12
    if d >= 3:
        return (R_bar * (d - R_bar**2)) / (1 - R_bar**2 + eps)
    else:
        return 2*R_bar / (1 - R_bar**2 + eps)

def vmf(ans_list, SenSimModel, debias=True, device=None):
    """
    E: (n, d) torch.Tensor
    """
    E = []
    for ans in ans_list:
        E.append(SenSimModel.encode(ans))
    E = np.array(E)
    E = torch.tensor(E)
    n, d = E.shape

    if debias:
        E = E - E.mean(dim=0, keepdim=True)

    X = E / (E.norm(dim=1, keepdim=True) + 1e-12)

    R_vec = X.sum(dim=0)
    R = R_vec.norm() + 1e-12
    R_bar = R / n
    # mu_hat = R_vec / R
    #
    # kappa_hat = kappa_from_Rbar_torch(R_bar, d)
    # C = kappa_hat / (kappa_hat + d - 1 + 1e-12)

    # D = 1 - R_bar
    # theta_typ = torch.acos(R_bar.clamp(-1, 1))
    # theta_typ_deg = theta_typ * 180.0 / torch.pi
    # pair_mean_sim = (n * R_bar**2 - 1) / (n - 1 + 1e-12)

    return float(R_bar.item())
    # return float(C.item())

def getvmf(log_dict, sample, SenSimModel, type, debias=True, device=None):
    ans_original = log_dict[sample['idx']]["ans_sampling_list_original"]
    ans_prompt = log_dict[sample['idx']]["ans_sampling_list_prompt"]
    if type == 'MULTI_CHOICE':
        choices = re.findall(r"\(\d+\):\s(.*)", sample['question'])
        ans_original = get_ans(ans_original, choices)
        ans_prompt = get_ans(ans_prompt, choices)
    s1 = vmf(ans_original, SenSimModel, debias=debias, device=device)
    s2 = vmf(ans_prompt, SenSimModel, debias=debias, device=device)
    return s1 * 0.5 + s2 * 0.5

def getvmf2(log_dict, sample, SenSimModel, type, debias=True, device=None):
    ans_original = log_dict[sample['idx']]["ans_sampling_list_original"]
    ans_prompt = log_dict[sample['idx']]["ans_sampling_list_prompt"]
    if type == 'MULTI_CHOICE':
        choices = re.findall(r"\(\d+\):\s(.*)", sample['question'])
        ans_original = get_ans(ans_original, choices)
        ans_prompt = get_ans(ans_prompt, choices)
    s1 = vmf(ans_original, SenSimModel, debias=debias, device=device)
    s2 = vmf(ans_prompt, SenSimModel, debias=debias, device=device)
    return s1,s2