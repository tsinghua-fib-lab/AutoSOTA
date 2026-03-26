from environment import *
from utils.timefeatures import time_features


class TSDataset(Dataset):
    def __init__(self, CFG, flag='train'):
        self.CFG = CFG
        self.flag = {'train': 0, 'vali': 1, 'test': 2}[flag]
        self.scale, self.scaler = True, StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        path = os.path.join(self.CFG.root_path, self.CFG.data_path)
        if 'PEMS' in self.CFG.data_path:
            data = np.load(path, allow_pickle=True)['data'][:, :, 0]
            stamp = torch.zeros(data.shape[0], 1)
        elif 'Solar' in self.CFG.data_path:
            data = []
            with open(path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip('\n').split(',')
                    data.append(np.stack([float(i) for i in line]))
            data = np.stack(data, 0)
            stamp = torch.zeros(data.shape[0], 1)
        else:
            data = pd.read_csv(path)
            columns = list(data.columns)
            columns.remove('date')
            columns.remove(self.CFG.target)
            data = data[['date'] + columns + [self.CFG.target]]
            stamp = data[['date']][:]
            stamp['date'] = pd.to_datetime(stamp.date)
            if self.CFG.embed == 'fixed':
                stamp['month'] = stamp.date.apply(lambda row: row.month, 1)
                stamp['day'] = stamp.date.apply(lambda row: row.day, 1)
                stamp['weekday'] = stamp.date.apply(lambda row: row.weekday(), 1)
                stamp['hour'] = stamp.date.apply(lambda row: row.hour, 1)
                if self.CFG.freq == 't':
                    stamp['minute'] = stamp.date.apply(lambda row: row.minute, 1)
                    stamp['minute'] = stamp.minute.map(lambda x: x // 15)
                stamp = stamp.drop(['date'], 1).values
            else:
                stamp = time_features(pd.to_datetime(stamp['date'].values), freq=self.CFG.freq)
                stamp = stamp.transpose(1, 0)
            if self.CFG.mode == 'S':
                data = data[[self.CFG.target]].values
            else:
                data = data[data.columns[1:]].values

        num_train = int(len(data) * 0.7)
        num_vali = int(len(data) * 0.15)
        num_test = len(data) - num_train - num_vali
        border1s = [0, num_train - self.CFG.seq_len, num_train + num_vali - self.CFG.seq_len]
        border2s = [num_train, num_train + num_vali, num_train + num_vali + num_test]
        border1 = border1s[self.flag]
        border2 = border2s[self.flag]

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        if 'PEMS' in self.CFG.data_path:
            data = pd.DataFrame(data).ffill(limit=len(data)).bfill(limit=len(data)).values

        self.x = torch.FloatTensor(data[border1:border2])
        self.y = torch.FloatTensor(data[border1:border2])
        self.stamp = torch.FloatTensor(stamp[border1:border2])
        self.cycle_index = torch.LongTensor((np.arange(len(data)) % self.CFG.cycle)[border1:border2])

    def __len__(self):
        return len(self.x) - self.CFG.seq_len - self.CFG.pred_len + 1

    def __getitem__(self, idx):
        x_s = idx
        x_e = x_s + self.CFG.seq_len
        y_s = x_e - self.CFG.label_len
        y_e = x_e + self.CFG.pred_len

        x = self.x[x_s:x_e]
        x_stamp = self.stamp[x_s:x_e]
        y = self.y[y_s:y_e]
        y_stamp = self.stamp[y_s:y_e]
        if self.CFG.model == 'CycleNet':
            x_stamp = self.cycle_index[x_e]

        return x, x_stamp, y, y_stamp

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class TSFactory:
    def __init__(self, CFG):
        self.CFG = CFG

    def __call__(self, flag):
        shuffle = False if flag == 'test' else True
        dataset = TSDataset(self.CFG, flag=flag)
        dataset_loader = DataLoader(dataset, batch_size=self.CFG.batch_size, shuffle=shuffle, num_workers=self.CFG.num_workers)
        return dataset, dataset_loader
