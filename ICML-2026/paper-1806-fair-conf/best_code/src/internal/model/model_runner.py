import torch
from tqdm import tqdm


def get_logits_targets_groups(dataset_class, data_loader, model, device):
    """
    Compute logits, targets, groups, and input identifiers for each record
    """
    logits_list = []
    target_list = []
    group_list = []
    input_identifiers = []
    with torch.no_grad():
        # switch to evaluate mode
        model.eval()
        for data in tqdm(data_loader):
            x, target, group, input_data = dataset_class.prepare_model_inputs(
                data, device
            )
            # compute output
            output = model(x)
            logits_list.append(output)
            target_list.append(target)
            group_list.append(group)
            input_identifiers.extend(input_data)
        logits = torch.cat(logits_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        groups = torch.cat(group_list, dim=0)

    return (
        logits.detach().cpu(),
        targets.detach().cpu(),
        groups.detach().cpu(),
        input_identifiers,
    )
