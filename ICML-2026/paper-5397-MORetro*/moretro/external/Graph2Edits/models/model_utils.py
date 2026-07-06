import csv
import torch
from torch.nn.utils.rnn import pad_sequence


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.
    Parameters
    ----------
    source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    final_size = index_size + suffix_dim

    # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = source.index_select(dim=0, index=index.view(-1))
    # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    target = target.view(final_size)

    return target


def creat_edits_feats(atom_feats, atom_scope):
    a_feats = []
    masks = []

    for idx, (st_a, le_a) in enumerate(atom_scope):
        feats = atom_feats[st_a: st_a + le_a]
        mask = torch.ones(feats.size(0), dtype=torch.uint8)
        a_feats.append(feats)
        masks.append(mask)

    a_feats = pad_sequence(a_feats, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)

    return a_feats, masks


def unbatch_feats(feats, atom_scope):
    atom_feats = []

    for idx, (st_a, le_a) in enumerate(atom_scope):
        atom_feats.append(feats[idx][:le_a])

    a_feats = torch.cat(atom_feats, dim=0)

    pad_tensor = torch.zeros(1, a_feats.size(1), device=a_feats.device)
    return torch.cat((pad_tensor, a_feats), dim=0)


def get_seq_edit_accuracy(seq_edit_scores, seq_labels, seq_mask):
    max_seq_len = seq_mask.shape[0]
    batch_size = seq_mask.shape[1]
    assert len(seq_edit_scores) == max_seq_len
    assert len(seq_labels) == max_seq_len
    assert len(seq_edit_scores[0]) == batch_size
    lengths = seq_mask.sum(dim=0).flatten()

    def check_equals(x, y): return torch.argmax(x) == torch.argmax(y)

    all_acc = 0
    for batch_id in range(batch_size):
        step_acc = 0
        seq_length = lengths[batch_id]
        for idx in range(seq_length):
            if check_equals(seq_edit_scores[idx][batch_id], seq_labels[idx][batch_id]):
                step_acc += 1

        if step_acc == seq_length:
            all_acc += 1

    accuracy = all_acc / batch_size
    return accuracy


class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg, arg_val in args.items():
            writer.writerow([arg, arg_val])
        # for arg in vars(args):
        #     writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
