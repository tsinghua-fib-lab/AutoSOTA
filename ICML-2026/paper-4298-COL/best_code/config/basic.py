import torchvision.transforms as transforms
from PIL import Image


class ConfigBasic:
    def __init__(self,):
        self.dataset = None
        self.setting = None
        self.logscale = False
        self.set_optimizer_parameters()
        self.set_training_opts()
        self.set_network()

    def set_dataset(self):
        if self.dataset == 'morph':
            self.setting = 'A'
            self.fold = '0'
        elif self.dataset == 'clap':
            self.setting = ''
            self.fold = 'eval_on_test'
        elif self.dataset == 'adience':
            self.setting= ''
            self.fold = '0'
        else:
            self.setting= ''
            self.fold = ''

        if self.dataset == 'morph':
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            if self.setting == 'A':
                self.is_filelist = True
                self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.fold}_train.txt'
                self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.fold}_test.txt'
                self.delimeter = ","
                self.img_idx = 4
                self.lb_idx = 3

            elif self.setting == 'B':
                self.delimeter = " "
                self.img_idx = 3
                self.lb_idx = 2
                if self.fold == 1:
                    self.is_filelist = True
                    self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1_train.txt'
                    self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2+S3_test.txt'
                else:
                    self.is_filelist = True
                    self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2_train.txt'
                    self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1+S3_test.txt'

            elif self.setting == 'C':
                self.delimeter = " "
                self.img_idx = 0
                self.lb_idx = 2
                self.is_filelist = True
                self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.fold}_train.txt'
                self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.fold}_test.txt'

            elif self.setting == 'D':
                self.delimeter = " "
                self.img_idx = 0
                self.lb_idx = 2
                self.is_filelist = True
                self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.fold}_train.txt'
                self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.fold}_test.txt'
            else:
                raise ValueError(f'setting {self.setting} is out of range.')

        elif self.dataset == 'adience':
            self.is_filelist = False
            self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.fold}_train_algn_[0_7].pickle'
            self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.fold}_test_algn_[0_7].pickle'
            # self.tau = 1

        elif self.dataset =='clap':
            self.delimeter = " "
            self.img_idx = 0
            self.lb_idx = 1
            self.is_filelist = True
            self.img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CLAP/2015'
            if self.fold == 'eval_on_test':
                self.train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_trainval.txt'
                self.test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_test.txt'
            elif self.fold == 'eval_on_val':
                self.train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_train.txt'
                self.test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_val.txt'
            else:
                raise ValueError(f'check fold: it should be [eval_on_test] or [eval_on_val], but {self.fold} is given.')

        elif self.dataset == 'ageDB':
            self.is_filelist = True
            self.img_root = '/hdd1/cwlee/datasets/OrderLearning/img'
            self.data_file = '/hdd1/cwlee/datasets/OrderLearning/index/AgeDB/agedb.csv'

        elif self.dataset == 'utk':
            self.is_filelist = True
            self.img_root = '/hdd1/cwlee/datasets/OrderLearning/img/utk'
            self.train_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_train_coral.csv'
            self.test_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_test_coral.csv'

        elif self.dataset == 'cacd':
            self.is_filelist = True
            self.img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CACD'
            self.data_file = '/hdd1/cwlee/datasets/OrderLearning/index/cacd/CACD.csv'

        elif self.dataset == 'koniq10k':
            self.split = 1
            self.is_filelist = True
            self.img_root = '/hdd1/cwlee/datasets/IQA/KonIQ10K/512x384/'
            _split_root = '/home/cwlee/datasets/IQA/KonIQ10K/splits'
            self.train_file = f'{_split_root}/KonIQ10K_train_split_{self.split}.csv'
            self.test_file = f'{_split_root}/KonIQ10K_test_split_{self.split}.csv'

        else:
            raise ValueError(f'{self.dataset} is out of range!')

        self.mean= [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        if self.dataset == 'koniq10k':
            self.transform_tr = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.transform_te = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            # numpy-based transforms for AGE datasets
            self.transform_tr = transforms.Compose([
                                                    lambda x: Image.fromarray(x),
                                                    transforms.RandomCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    self.normalize
                                                    ])
            self.transform_te = transforms.Compose([
                                                    lambda x: Image.fromarray(x),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    self.normalize
                                                    ])



    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.learning_rate = 0.0001
        # self.lr_decay_epochs = [30, 50, 100]
        self.lr_decay_epochs = [60, 90]
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # *** Scheduler
        self.scheduler = 'cosine'

    def set_network(self):
        self.model = 'T_v0'
        # self.backbone = 'vgg16bn'
        self.backbone = 'vitB16'
        self.ckpt = None

    def set_training_opts(self):
        # *** Print Option
        self.val_freq = 3
        self.print_freq = 50

        # *** Training
        self.batch_size = 32
        self.num_workers = 1
        self.epochs = 100

        # *** Save option
        self.save_freq = 100
        self.wandb = False

    def set_test_opts(self):
        self.ckpt = None

    def set_biqa_dataset(self):
        """Set transforms for BIQA datasets (BID, CLIVE, KonIQ10k, SPAQ, FLIVE)."""
        from torchvision import transforms
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform_tr = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        self.transform_te = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])
