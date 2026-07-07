from dotmap import DotMap
from tqdm import tqdm
from utils import *
from data_process import *
from model import PCRNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from mne.decoding import CSP
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import config

np.set_printoptions(suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CustomDatasets(Dataset):
    # initialization: data and label
    def __init__(self, seq_data, label_data):
        self.seq_data = seq_data

        self.label = label_data

    # get the size of data
    def __len__(self):
        return len(self.label)

    # get the data and label
    def __getitem__(self, index):
        seq_data = torch.Tensor(self.seq_data[index])

        label = torch.Tensor(self.label[index])

        return seq_data, label


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initiate(args, train_loader, valid_loader, test_loader, subject):
    torch.manual_seed(args.subject_number)
    torch.cuda.manual_seed(args.subject_number)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model = PCRNet(args)

    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(params=model.parameters(), lr=0.004, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.003 / 10)
    model = model.cuda()
    criterion = criterion.cuda()

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, args, train_loader, valid_loader, test_loader, subject)


def train_model(settings, args, train_loader, valid_loader, test_loader, subject):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion, scheduler):
        model.train()
        proc_loss, proc_size = 0, 0
        train_acc_sum = 0
        train_loss_sum = 0
        for i_batch, batch_data in enumerate(train_loader):
            seq_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            seq_data, train_label = seq_data.cuda(), train_label.cuda()

            batch_size = train_label.size(0)

            # Forward pass
            preds = model(seq_data)

            loss = criterion(preds, train_label.long())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            train_loss_sum += loss.item() * batch_size
            predicted = preds.data.max(1)[1]
            train_acc_sum += predicted.eq(train_label).cpu().sum()

        scheduler.step()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)
        total_loss = 0.0
        test_acc_sum = 0
        proc_size = 0

        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                seq_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                seq_data, test_label = seq_data.cuda(), test_label.cuda()

                proc_size += args.batch_size
                preds = model(seq_data)
                # Backward and optimize
                optimizer.zero_grad()

                total_loss += criterion(preds, test_label.long()).item() * args.batch_size

                predicted = preds.data.max(1)[1]  # 32
                test_acc_sum += predicted.eq(test_label).cpu().sum()

        avg_loss = total_loss / (num_batches * args.batch_size)

        avg_acc = test_acc_sum / (num_batches * args.batch_size)

        return avg_loss, avg_acc

    best_epoch = 1
    best_valid = float('inf')
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion, scheduler)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        print()
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc '
            '{:5.4f}'.format(
                epoch,
                args.name,
                train_loss,
                train_acc,
                val_loss,
                val_acc))

        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0

            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 10:
                break

    model = load_model(args, name=args.name)
    test_loss, test_acc = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")

    return test_loss, test_acc

def get_AVED_data(args, test_id, modality="audio-only"):
    '''description: get AVED data for specific subject and modality
    param {type}
    return {type}:
        data: EEG data
        label: attention labels'''

    if modality == "audio-only":
        filename = os.path.join(args.data_document_path, "audio-only", f"sub{test_id}.csv")
    elif modality == "audio-video":
        filename = os.path.join(args.data_document_path, "audio-video", f"sub{test_id}.csv")
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")

    data_pf = pd.read_csv(filename, header=None)
    eeg_data = data_pf.iloc[:, :].values

    all_label = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    print(f'Finish get the {modality} data from: {filename}')
    return eeg_data, all_label


def main_KUL(name="S13", dataset="KUL", data_document_path="../KUL", time_len=1):
    args = DotMap()
    args.name = name  # 被试名
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 64  # KUL 128 #DTU 64 #采样点
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32  # 批量大小
    args.max_epoch = 100
    args.patience = 15
    args.log_interval = 20
    args.image_size = 32
    args.people_number = 16  # KUL16 #DTU 18
    args.eeg_channel = 64  # 通道数
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8  # KUL 8 # 60
    args.cell_number = 46080  # KUL 46080 #DTU 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0
    args.csp_comp = 64
    args.log_path = "./result"

    args.frequency_resolution = args.fs / args.window_length

    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    logger = get_logger(args.name, args.log_path, time_len)

    # load data 和 label
    eeg_data, event_data = read_prepared_data(args)

    data = np.vstack(eeg_data)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    eeg_data = eeg_data.transpose(0, 2, 1)
    event_data = np.squeeze(np.array(event_data) - 1)
    train_data, test_data, train_label, test_label = within_data(eeg_data, event_data)

    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
              norm_trace=True)
    train_data = csp.fit_transform(train_data, train_label)
    test_data = csp.transform(test_data)

    train_data = train_data.transpose(0, 2, 1)  # within:(60, 5760, 64) cross:(54, 6400, 64)
    test_data = test_data.transpose(0, 2, 1)  # within:(60, 64, 640) cross:(6, 6400, 64)
    train_eeg, train_label = sliding_window_csp(train_data, train_label, args,
                                                args.csp_comp)  # within:(5340, 128, 64) cross:(5346, 128, 64)
    test_eeg, test_label = sliding_window_csp(test_data, test_label, args,
                                              args.csp_comp)  # within:(540, 128, 64) cross:(594, 128, 64)

    seq_train_data = np.expand_dims(train_eeg, axis=-1)
    seq_test_data = np.expand_dims(test_eeg, axis=-1)
    # train_label = np.squeeze(train_label - 1)
    # test_label = np.squeeze(test_label - 1)
    del eeg_data

    np.random.seed(200)
    np.random.shuffle(seq_train_data)
    np.random.seed(200)
    np.random.shuffle(train_label)

    np.random.seed(200)
    np.random.shuffle(seq_test_data)
    np.random.seed(200)
    np.random.shuffle(test_label)

    seq_train_data, seq_valid_data, train_label, valid_label = train_test_split(seq_train_data, train_label,
                                                                                test_size=0.1, random_state=42)

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
    seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
    seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

    train_loader = DataLoader(dataset=CustomDatasets(seq_train_data, train_label),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=CustomDatasets(seq_valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(dataset=CustomDatasets(seq_test_data, test_label),
                             batch_size=args.batch_size, drop_last=True)

    loss, acc = initiate(args, train_loader, valid_loader, test_loader, args.name)

    info_msg = f'{dataset}_{name}_{str(time_len)}s loss:{str(loss)} acc:{str(acc.item())}'
    result_logger.info(info_msg)

    print(loss, acc)
    logger.info(loss)
    logger.info(acc)
    return acc


def main_DTU(name="S13", dataset="KUL", data_document_path="../DTU", time_len=1):
    args = DotMap()
    args.name = name  # 被试名
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 64  # KUL 128 # DTU 64 #采样点
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32  # 批量大小
    args.max_epoch = 100
    args.patience = 15
    args.log_interval = 20
    args.image_size = 32
    args.people_number = 18  # KUL16 #DTU 18
    args.eeg_channel = 64  # 通道数
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 60  # KUL 8 # 60
    args.cell_number = 3200  # KUL 46080 #DTU 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0
    args.csp_comp = 64
    args.log_path = "./result"

    args.frequency_resolution = args.fs / args.window_length

    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    logger = get_logger(args.name, args.log_path, time_len)

    # load data 和 label
    subpath = args.data_document_path + '/' + str(args.name) + '_data_preproc.mat'
    eeg_data, event_data = get_data_from_mat(subpath)
    eeg_data = np.array(eeg_data)  # DTU
    eeg_data = eeg_data[:, :, 0:64]  # DTU

    data = np.vstack(eeg_data)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    eeg_data = eeg_data.transpose(0, 2, 1)
    event_data = np.squeeze(np.array(event_data) - 1)
    train_data, test_data, train_label, test_label = within_data(eeg_data, event_data)

    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
              norm_trace=True)
    train_data = csp.fit_transform(train_data, train_label)
    test_data = csp.transform(test_data)

    train_data = train_data.transpose(0, 2, 1)  # within:(60, 5760, 64) cross:(54, 6400, 64)
    test_data = test_data.transpose(0, 2, 1)  # within:(60, 64, 640) cross:(6, 6400, 64)
    train_eeg, train_label = sliding_window_csp(train_data, train_label, args,
                                                args.csp_comp)  # within:(5340, 128, 64) cross:(5346, 128, 64)
    test_eeg, test_label = sliding_window_csp(test_data, test_label, args,
                                              args.csp_comp)  # within:(540, 128, 64) cross:(594, 128, 64)

    seq_train_data = np.expand_dims(train_eeg, axis=-1)
    seq_test_data = np.expand_dims(test_eeg, axis=-1)
    # train_label = np.squeeze(train_label - 1)
    # test_label = np.squeeze(test_label - 1)
    del eeg_data

    np.random.seed(200)
    np.random.shuffle(seq_train_data)
    np.random.seed(200)
    np.random.shuffle(train_label)

    np.random.seed(200)
    np.random.shuffle(seq_test_data)
    np.random.seed(200)
    np.random.shuffle(test_label)

    seq_train_data, seq_valid_data, train_label, valid_label = train_test_split(seq_train_data, train_label,
                                                                                test_size=0.1, random_state=42)

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
    seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
    seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

    train_loader = DataLoader(dataset=CustomDatasets(seq_train_data, train_label),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=CustomDatasets(seq_valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(dataset=CustomDatasets(seq_test_data, test_label),
                             batch_size=args.batch_size, drop_last=True)

    loss, acc = initiate(args, train_loader, valid_loader, test_loader, args.name)

    info_msg = f'{dataset}_{name}_{str(time_len)}s loss:{str(loss)} acc:{str(acc.item())}'
    result_logger.info(info_msg)

    print(loss, acc)
    logger.info(loss)
    logger.info(acc)
    return acc


def main_AVED(name="S1", dataset="AVED", data_document_path="../AVED", time_len=1, modality="audio-only"):
    args = DotMap()
    args.name = name
    args.subject_number = int(name[1:]) if 'S' in name else int(name)
    args.data_document_path = data_document_path
    args.modality = modality
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 100
    args.patience = 15
    args.log_interval = 20
    args.image_size = 32
    args.people_number = 10
    args.eeg_channel = 32  
    args.csp_comp = 32     
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 16
    args.cell_number = 46080
    args.test_percent = 0.2
    args.vali_percent = 0.1
    args.label_col = 0
    args.log_path = "./result"
    args.win_len = args.window_length
    args.window_lap = args.window_length * args.overlap

    args.frequency_resolution = args.fs / args.window_length
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)

    logger = get_logger(f"{args.name}_{modality}", args.log_path, time_len)

    print(f'Loading AVED {modality} data for subject {args.name}...')
    eeg_data, event_data = get_AVED_data(args, args.subject_number, modality)
    event_data = np.array(event_data) - 1

    samples_per_trial = eeg_data.shape[0] // args.trail_number
    eeg_data_reshaped = []
    for i in range(args.trail_number):
        start_idx = i * samples_per_trial
        end_idx = (i + 1) * samples_per_trial
        trial_data = eeg_data[start_idx:end_idx, :]
        eeg_data_reshaped.append(trial_data)

    eeg_data_reshaped = np.array(eeg_data_reshaped)  # (16, samples_per_trial, 32)

    eeg_windows, event_windows = sliding_window(eeg_data_reshaped, event_data, args, args.eeg_channel)

    eeg_windows = eeg_windows.transpose(0, 2, 1)  # (n_windows, channels, time_points)
    event_windows = np.squeeze(event_windows)

    train_data, test_data, train_label, test_label = train_test_split(
        eeg_windows, event_windows, test_size=args.test_percent,
        random_state=42, stratify=event_windows
    )

    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat',
              transform_into='csp_space', norm_trace=True)
    train_data = csp.fit_transform(train_data, train_label)
    test_data = csp.transform(test_data)

    train_data = train_data.transpose(0, 2, 1)  # (n_train, time_points, csp_components)
    test_data = test_data.transpose(0, 2, 1)  # (n_test, time_points, csp_components)

    train_eeg, train_label = sliding_window_csp(train_data, train_label, args, args.csp_comp)
    test_eeg, test_label = sliding_window_csp(test_data, test_label, args, args.csp_comp)

    seq_train_data = np.expand_dims(train_eeg, axis=-1)
    seq_test_data = np.expand_dims(test_eeg, axis=-1)

    np.random.seed(200)
    np.random.shuffle(seq_train_data)
    np.random.seed(200)
    np.random.shuffle(train_label)

    np.random.seed(200)
    np.random.shuffle(seq_test_data)
    np.random.seed(200)
    np.random.shuffle(test_label)

    seq_train_data, seq_valid_data, train_label, valid_label = train_test_split(
        seq_train_data, train_label, test_size=args.vali_percent, random_state=42
    )

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
    seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
    seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

    print(f"Training data shape: {seq_train_data.shape}")
    print(f"Training labels: {np.unique(train_label, return_counts=True)}")

    train_loader = DataLoader(dataset=CustomDatasets(seq_train_data, train_label),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=CustomDatasets(seq_valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(dataset=CustomDatasets(seq_test_data, test_label),
                             batch_size=args.batch_size, drop_last=True)

    loss, acc = initiate(args, train_loader, valid_loader, test_loader, f"{args.name}_{modality}")

    info_msg = f'{dataset}_{name}_{modality}_{str(time_len)}s loss:{loss:.4f} acc:{acc.item():.4f}'
    result_logger.info(info_msg)

    print(f"Loss: {loss:.4f}, Acc: {acc.item():.4f}")
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Acc: {acc.item():.4f}")
    return acc.item()


result_logger = logging.getLogger('result')
result_logger.setLevel(logging.INFO)

if __name__ == "__main__":
    file_handler = logging.FileHandler('log/result.log')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    total_acc = 0

    result_logger.addHandler(file_handler)
    all_test_acc = []
    if config.dataset == "KUL":
        for i in range(1, config.people_number + 1):
            name = 'S' + str(i)
            total_acc = main_KUL(name=name, dataset=config.dataset, data_document_path=config.data_document_path,
                                 time_len=config.time_len)
            all_test_acc.append(total_acc)
    elif config.dataset == "DTU":
        for i in range(1, config.people_number + 1):
            name = 'S' + str(i)
            total_acc = main_DTU(name=name, dataset=config.dataset, data_document_path=config.data_document_path,
                                 time_len=config.time_len)
            all_test_acc.append(total_acc)
    elif config.dataset == "AVED":
        modalities = getattr(config, 'modalities', ['audio-only'])  

        for modality in modalities:
            print(f"\n=== Processing AVED {modality} modality ===")
            modality_acc = []
            for i in range(1, config.people_number + 1):
                name = 'S' + str(i)
                acc = main_AVED(name=name, dataset=config.dataset,
                                data_document_path=config.data_document_path,
                                time_len=config.time_len, modality=modality)
                modality_acc.append(acc)

            avg_acc = np.mean(modality_acc)
            std_acc = np.std(modality_acc)
            print(f"AVED {modality} - Average Acc: {avg_acc:.4f} ± {std_acc:.4f}")
            info_msg = f'AVED_{modality}_{str(config.time_len)}s avg_acc:{avg_acc:.4f} std:{std_acc:.4f}'
            result_logger.info(info_msg)
            all_test_acc.extend(modality_acc)

    print(f'avg_acc: {np.mean(all_test_acc):.4f}')
    info_msg = f'The average accuracy of {config.dataset}_{str(config.time_len)}s avg_acc:{np.mean(all_test_acc):.4f} std:{np.std(all_test_acc):.4f} '
    result_logger.info(info_msg)
