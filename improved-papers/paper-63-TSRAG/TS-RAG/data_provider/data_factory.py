from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Custom_S, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_ETT_hour_retrieve, Dataset_ETT_minute_retrieve, Dataset_Custom_retrieve


data_dict = {
    'custom': Dataset_Custom_S,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'ett_h_retrieve': Dataset_ETT_hour_retrieve,
    'ett_m_retrieve': Dataset_ETT_minute_retrieve,
    'custom_retrieve': Dataset_Custom_retrieve,
}
def data_provider(args, flag, drop_last_test=False, train_all=False, retriever_rawdata=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    max_len = args.max_len
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if 'retrieve' in args.model_id:
        data_set = Data(
            model_id=args.model_id , 
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            max_len=max_len,
            train_all=train_all,
            top_k=args.top_k,
            retriever_rawdata=retriever_rawdata,
            mode=args.mode
        )
        # check_dataset(data_set)
    else:
        data_set = Data(
            model_id=args.model_id , 
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            max_len=max_len,
            train_all=train_all,
            return_feature_id=args.return_feature_id
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

def check_dataset(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]
        batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances = sample
        if i == 0:
            expected_x_shape = batch_x.shape
            expected_y_shape = batch_y.shape
            expected_retrieved_seqs_shape = retrieved_seqs.shape
            expected_distances_shape = distances.shape

        else:
            assert batch_x.shape == expected_x_shape, f"Expected {expected_x_shape}, but got {batch_x.shape}"
            assert batch_y.shape == expected_y_shape, f"Expected {expected_y_shape}, but got {batch_y.shape}"
            assert retrieved_seqs.shape == expected_retrieved_seqs_shape, f"Expected {expected_retrieved_seqs_shape}, but got {retrieved_seqs.shape}"
            assert distances.shape == expected_distances_shape, f"Expected {expected_distances_shape}, but got {distances.shape}"