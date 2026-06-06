import os
import sys
import time
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cuda as cudnn
import torch.nn.functional as F
import numpy as np

from utils.avgmeter import AverageMeter
from utils.ioueval import iouEval
from compute_point_level_ood import PointOODMetricsCalculator

from network.minkunet import MinkUNet

def get_maxlogit(logit): # ok but not great
    #probs = logit.softmax(dim=1)
    conf = torch.max(logit, dim=1).values
    scores = 1 - conf
    return scores

def get_rba(logit):
    rba = -torch.tanh(logit).sum(dim=1)
    rba[rba < -1] = -1
    rba = rba + 1
    return rba

def get_void_score(logit, void_idx):
    return logit[:, void_idx]

def softmax_thresholding(logit, threshold=0.45): # seems to detect more boundary cases between similar classes than real anomalies
    probs = logit.softmax(dim=1)
    count = (probs > threshold).sum(dim=1)
    mask = (count >= 2).long()
    print(f"Softmax thresholding: {mask.sum()} points above threshold {threshold}")
    return mask

class User():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir, split, save=False, eval=False, fp16=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.split = split
        self.save = save
        self.eval = eval
        self.fp16 = fp16

        self.nuscenes_flag = False

        # get data
        if self.ARCH['dataset']['pc_dataset_type'] == 'SemanticKITTI':
            from dataloader.kitti import Parser
        elif self.ARCH['dataset']['pc_dataset_type'] == 'SemanticPOSS':
            from dataloader.poss import Parser
        elif self.ARCH['dataset']['pc_dataset_type'] == 'nuScenes':
            from dataloader.nuscenes import Parser
            self.nuscenes_flag = True
        else:
            raise ValueError(f"Dataset type {self.ARCH['dataset']['pc_dataset_type']} not supported")
        
        self.parser = Parser(root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=self.DATA["split"]["test"],
            labels=self.DATA["labels"],
            color_map=self.DATA["color_map"],
            learning_map=self.DATA["learning_map"],
            learning_map_inv=self.DATA["learning_map_inv"],
            sensor=self.ARCH["dataset"]["sensor"],
            voxel_size=self.ARCH["model_params"]["voxel_size"],
            batch_size=1,
            workers=self.ARCH["train"]["workers"],
            gt=True,
            aug=False,
            shuffle_train=False)
        
        # load model
        with torch.no_grad():
            self.model = MinkUNet(in_dim=self.ARCH['model_params']['input_dims'],
                                  num_classes=self.parser.get_n_classes(),
                                  layer_num=self.ARCH['model_params']['layer_num'],
                                  cr=self.ARCH['model_params']['cr'])

        model_path = os.path.join(self.modeldir, 'checkpoint-best.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        try:
            #load model from .pt file
            self.model.load_state_dict(torch.load(model_path), strict=True)
            print(f"[INFO] Model loaded from {model_path}")
        except:
            # load model from checkpoint directory
            self.model.load_state_dict(torch.load(model_path)['model'], strict=True)
            print(f"[INFO] Model loaded from {model_path} checkpoint")

        # GPU
        self.gpu = False
        self.model_single = self.model
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        print('[INFO] Infering in device: ', self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

        # evaluation (ignore class 0)
        self.evaluator = iouEval(self.parser.get_n_classes(), self.device, self.ARCH['dataset']['ignore_label'])
        # ood evaluator
                # Determine anomaly class from the data config
        # STU uses label 1 for outlier; we extract from 'labels' config
        anomal_class = [k for k, v in self.DATA['labels'].items() if v == 'outlier']
        if anomal_class:
            anomal_class = anomal_class[0]
        else:
            anomal_class = 2  # default for KITTI, POSS
        self.ood_evaluator = PointOODMetricsCalculator(nuscenes=self.nuscenes_flag, anomal_class=anomal_class)

        # extract mavs
        with open(os.path.join(self.modeldir, "mavs.pickle"), 'rb') as h1:
            self.mavs = pickle.load(h1)
        with open(os.path.join(self.modeldir, "vars.pickle"), 'rb') as h2:
            self.vars = pickle.load(h2)
        self.mavs = torch.vstack(tuple(self.mavs.values())).cpu()
        self.vars = torch.vstack(tuple(self.vars.values())).cpu()
        print(f"[INFO] Loaded {self.mavs.shape[0]} mavs of dimension {self.mavs.shape[1]}")

    def infer(self):
        if self.split == 'train':
            # do train set
            acc, iou = self.infer_subset(loader=self.parser.get_train_set(),
                                         to_orig_fn=self.parser.to_original,
                                         evaluator=self.evaluator)
            print('Split: {} | acc: {:.2%} | iou: {:.2%}'.format(self.split, acc, iou))
        elif self.split == 'valid':
            acc, iou = self.infer_subset(loader=self.parser.get_valid_set(),
                                         to_orig_fn=self.parser.to_original,
                                         evaluator=self.evaluator)
            print('Split: {} | acc: {:.2%} | iou: {:.2%}'.format(self.split, acc, iou))
        elif self.split == 'test':
            self.infer_subset(loader=self.parser.get_test_set(),
                                         to_orig_fn=self.parser.to_original,
                                         evaluator=self.evaluator)
        else:
            raise SyntaxError('Invalid split chosen. Choose one of \'train\', \'valid\', \'test\'')

        print('Finished Infering')

    def infer_subset(self, loader, to_orig_fn, evaluator):
        # switch to evaluation mode
        self.model.eval()

        mean_time = AverageMeter()

        evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()

        with torch.inference_mode():
            for i, data_dict in tqdm(enumerate(loader), total=len(loader)):

                lidar = data_dict['lidar'].to(self.device)
                points = data_dict['points'].to(self.device)
                # voxel_labels = data_dict['targets'].F.to(self.device).long()
                point_labels = data_dict['targets_mapped'].to(self.device)
                orig_labels = data_dict['targets_original']
                # ref_index = data_dict['ref_index']
                # origin_len = data_dict['origin_len']
                # map = data_dict['map']
                # num_voxel = data_dict['num_voxel']
                invs = data_dict['inverse_map'].to(self.device)
                path_seq = data_dict['seq'][0]
                path_name = data_dict['name'][0]

                with torch.amp.autocast(device_type=self.device_name, enabled=self.fp16):
                    output, coutput = self.model(lidar)

                inv = invs.F # (N)
                out = output[inv] # (N, C)
                cout = coutput[inv] # (N, C)
                preds = out.argmax(dim=1) # (N)
                labels = point_labels.F # (N)
                points = points.F # (N, 4)
                orig_labels = orig_labels.F # (N)
                
                # using mavs with cosine similarity
                sim = cosine_similarity(out.cpu().float(), self.mavs)
                pred_classes = torch.argmax(sim, dim=1)

                preds = pred_classes.to(self.device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                mean_time.update(time.time() - end)
                end = time.time()

                # save scan
                pred_np = preds.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # simple post-processing ood techniques
                # scores = softmax_thresholding(out)
                # scores = get_maxlogit(out)

                # scores with mavs distance
                sim = cosine_similarity(out.float(), self.mavs.to(self.device))
                # IDEA-010: Use top-3 prototype similarities blended with top-1
                topk_sim, _ = sim.topk(k=3, dim=1)
                avg_sim = topk_sim.mean(dim=1)
                max_sim, _ = sim.max(dim=1)
                scores = 1 - (0.5 * max_sim + 0.5 * avg_sim)

                # normalized shannon entropy
                probs = out.softmax(dim=1)
                log_probs = out.log_softmax(dim=1)
                entropy = -(probs * log_probs).sum(dim=1) / np.log(self.parser.get_n_classes())

                scores = scores * entropy
                # IDEA-004: Removed per-frame max normalization

                # contrastive decoder scores
                cscores = F.relu(1-(torch.linalg.norm(cout, dim=1)**2 / 1000)) # r=1000

                scores = (scores + cscores) / 2.0

                scores = scores.cpu().numpy().astype(np.float32)

                # compute metrics
                if self.eval and self.split != 'test':
                    evaluator.addBatch(pred_np, labels)
                
                # update ood evaluator
                self.ood_evaluator.update(points.cpu().numpy(), scores, orig_labels.numpy())

                if self.save:
                    # map to original label
                    pred_np = to_orig_fn(pred_np)

                    # save scan
                    path = os.path.join(self.logdir, "sequences",
                                        path_seq, "predictions", path_name)
                    pred_np.tofile(path)

                    # save per-point probabilities (softmax scores) for deep ensemble OOD methods
                    # prob = out.softmax(dim=1).cpu().numpy()
                    # path_probs = os.path.join(self.logdir, "sequences",
                    #                           path_seq, "probs", path_name.replace('.label', '.npy'))
                    # np.save(path_probs, prob)

                    # save max logits
                    path_logits = os.path.join(self.logdir, "sequences",
                                               path_seq, "scores", path_name.replace('.label', '.txt'))
                    np.savetxt(path_logits, scores, fmt='%.6f')

        # print times
        print('Inference time per scan: {:.3f}'.format(mean_time.avg))

        ood_metrics = self.ood_evaluator.compute_metrics()
        print("OOD metrics: ", ood_metrics)

        # when done, do the evaluation
        if self.eval and self.split != 'test':
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoUMissingClass()
            # return also iou per class
            for i, jacc in enumerate(class_jaccard):
                print('IoU class {i:} {class_str:} = {jacc:.3f}'.format(
                    i=i, class_str=self.parser.get_xentropy_class_string(i), jacc=jacc))
    
            # print for spreadsheet
            # print("*" * 80)
            # print("below can be copied straight for paper table")
            # for i, jacc in enumerate(class_jaccard):
            #     if i not in [0]:
            #         sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            #         sys.stdout.write(",")
            # sys.stdout.write('{jacc:.3f}'.format(jacc=jaccard.item()))
            # sys.stdout.write(",")
            # sys.stdout.write('{acc:.3f}'.format(acc=accuracy.item()))
            # sys.stdout.write('\n')
            # sys.stdout.flush()

            return accuracy, jaccard
        return 0, 0
        
def cosine_similarity(a, b):
    a_norm = F.normalize(a, p=2, dim=1) # (N, C)
    b_norm = F.normalize(b, p=2, dim=1) # (C, C)
    return torch.matmul(a_norm, b_norm.T) # (N, C)