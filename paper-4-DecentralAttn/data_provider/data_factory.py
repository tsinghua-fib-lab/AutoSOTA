from data_provider.data_loader import (
    APAVALoader,
    ADFTDLoader,
    TDBRAINLoader,
    PTBLoader,
    PTBXLLoader,
    FLAAPLoader,
    UCIHARLoader,
)
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    "APAVA": APAVALoader,  # dataset APAVA
    "TDBRAIN": TDBRAINLoader,  # dataset TDBRAIN
    "ADFTD": ADFTDLoader,  # dataset ADFTD
    "PTB": PTBLoader,  # dataset PTB
    "PTB-XL": PTBXLLoader,  # dataset PTB-XL
    "FLAAP": FLAAPLoader,  # dataset FLAAP
    "UCI-HAR": UCIHARLoader,  # dataset HAR
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == "test":
        shuffle_flag = False
    else:
        shuffle_flag = True
    batch_size = args.batch_size
    drop_last = False
    data_set = Data(
            root_path=args.root_path,
            args=args,
            flag=flag,
        )

    data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(
                x, max_len=args.seq_len
            ),  # only called when yeilding batches
        )
    return data_set, data_loader