import torch
import torch.utils.data as dataloader
import numpy as np
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from model import GBMB
from DataHandler import DataHandler
from DataHandler import TrnData
import numpy as np
import pickle
from Utils.Utils import *
import os
import random
import setproctitle
from copy import deepcopy
from datetime import datetime
import time


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
if args.device == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'bprLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret
    
    def makePrintRes_test(self, name, ep, reses):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            if metric in ['Recall10', 'Recall20', 'NDCG10', 'NDCG20']:
                val = reses[metric]
                ret += '%s = %.4f, ' % (metric, val)
        ret = ret[:-2] + '  '
        return ret
    
    def makePrintRes(self, name, ep, reses):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        #保存每次训练测试的超参数和每一轮的结果    
        curTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fileName = f'trainingResult{args.data}-{curTime}'
        with open('./Result/' + fileName + '.txt', 'w') as f:
            hypeParameters = vars(args) 
            f.write("HyperParameters:\n")
            for k, v in hypeParameters.items():
                f.write(f"{k}: {v}\n")
        bestRes = None
        bestep = None
        train_time = 0
        earlyStop_count = 0
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses, eptime = self.trainEpoch(ep)
            train_time += eptime
            # Temperature1 annealing: linear decay from 1.0 to 0.2 over 20 epochs
            temp_warmup = 20
            temp_start, temp_end = 1.0, 0.2
            if ep < temp_warmup:
                self.model.temperature1 = temp_start - (temp_start - temp_end) * ep / temp_warmup
            else:
                self.model.temperature1 = temp_end
            print(f'Training time: {train_time:.2f} seconds')
            log(self.makePrint('Train', ep, reses, tstFlag))
            with open('./Result/' + fileName + '.txt', 'a') as f:
                f.write(f"{self.makePrintRes('Train', ep, reses)}\n")
            if tstFlag:
                with torch.no_grad():
                    reses = self.testEpoch()
                    print('')
                with open('./Result/' + fileName + '.txt', 'a') as f:
                    f.write(f"{self.makePrintRes_test('Test', ep, reses)}\n")
                log(self.makePrintRes_test('Test', ep, reses))

                # self.saveHistory()
                if bestRes is None:
                    bestRes = reses
                    bestep = ep
                elif (reses['NDCG20'] < bestRes['NDCG20']) or (reses['NDCG20'] == bestRes['NDCG20'] and reses['Recall20'] < bestRes['Recall20']):
                    earlyStop_count += 1
                    print('Early stop count: %d' % earlyStop_count)
                    with open('./Result/' + fileName + '.txt', 'a') as f:
                        f.write(f"Early stop count: {earlyStop_count}\n")
                    # if earlyStop_count >= args.earlyStop and ep > 50:
                    if earlyStop_count >= args.earlyStop:
                        print('************************************************************')
                        print('Early stop at epoch %d !!!!' % ep)
                        with open('./Result/' + fileName + '.txt', 'a') as f:
                            f.write("************************************************************\n")
                            f.write(f"Early stop at epoch {ep} !!!!\n")
                            f.write(f"{self.makePrintRes('Best Result', bestep, bestRes)}\n")
                        log(self.makePrint('Best Result', bestep, bestRes, True))
                        # torch.save({
                        #     'user_embeddings': self.model.user_embeddings.data,
                        #     'item_embeddings': self.model.item_embeddings.data
                        # }, f'{args.data}pretraining.pth')
                        break
                else:
                    earlyStop_count = 0
                    bestRes = reses
                    bestep = ep
                # bestRes = reses if bestRes is None or reses['Recall20'] > bestRes['Recall20'] else bestRes

            with open('./Result/' + fileName + '.txt', 'a') as f:
                f.write(f"{self.makePrintRes_test('Best Result', bestep, bestRes)}\n")
                f.write("###################################################\n")
            log(self.makePrintRes_test('Best Result', bestep, bestRes))
            print('###################################################')
            
        # with open('./Result/' + fileName + '.txt', 'a') as f:
        #     f.write(f"{self.makePrintRes('Test', ep, reses)}\n")           
        #     f.write(f"{self.makePrintRes('Best Result', args.epoch, bestRes)}")

        # self.saveHistory()

    def prepareModel(self):
        # main model
        self.model = GBMB(self.handler.torchBuyAdj, self.handler.auxAdj_list, self.handler.aux_allOneAdj_list,self.handler.torchSumAdj).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)

    def trainEpoch(self, ep):
        epLoss, epbprLoss, epreg_loss, epib_loss, epcontrast_loss = 0, 0, 0, 0, 0
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        steps = trnLoader.dataset.__len__() // args.batch
        self.model.train()
        epoch_start_time = time.time()
        for i, tem in enumerate(trnLoader):
            users, pos_items, neg_items = tem
            users = users.long().to(device)
            pos_items = pos_items.long().to(device)
            neg_items = neg_items.long().to(device)
            loss, bpr_loss, ib_loss, contrastive_loss, reg_loss = self.model.cal_loss(users, pos_items, neg_items)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            epLoss += loss.item()
            epbprLoss += bpr_loss.item()
            epib_loss += ib_loss.item()
            epreg_loss += reg_loss.item()
            epcontrast_loss += contrastive_loss.item()
            log('Step %d/%d: loss = %.3f, bprloss = %.3f,  ibloss = %.3f, regloss =  %.3f ' % (i, steps, loss.item(), bpr_loss.item(), ib_loss.item(), reg_loss.item()), save=False, oneline=True)
        epoch_end_time = time.time()
        print(f"Epoch time: {(epoch_end_time - epoch_start_time):.3f} seconds")
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['bprLoss'] = epbprLoss / steps
        ret['epib_loss'] = epib_loss / steps
        ret['epreg_loss'] = epreg_loss / steps
        ret['epcontrast_loss'] = epcontrast_loss / steps
        return ret, epoch_end_time - epoch_start_time

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        Topks = [5, 10, 20, 30, 40, 50]
        epLoss = 0
        epresults = {f'Recall{k}': 0 for k in Topks}  # 初始化Recall
        epresults.update({f'Hit rate{k}': 0 for k in Topks})  # 初始化NDCG
        epresults.update({f'NDCG{k}': 0 for k in Topks})  # 初始化NDCG
        i = 0
        num_true_samples = self.handler.tstMat.nnz
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        self.model.eval()
        _, _ ,_ ,_ ,_ ,_ , _,_, usrEmbeds, itmEmbeds = self.model.forward()


        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().to(device)
            trnMask = trnMask.to(device)

            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            for k in Topks:
                _, topLocsk = torch.topk(allPreds, k)
                recallk, ndcgk, hitsk = self.calcRes(topLocsk.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, k)
                epresults[f'Recall{k}'] += recallk
                epresults[f'NDCG{k}'] += ndcgk
                epresults[f'Hit rate{k}'] += hitsk
                log(f'Steps {i}/{steps}: Recall@{k} = {recallk:.1f}, num of Hits@{k} = {hitsk:.1f}, NDCG@{k} = {ndcgk:.1f}', save=False, oneline=True)
        ret = dict()
        for k in Topks:
            ret[f'Recall{k}'] = epresults[f'Recall{k}'] / num
            ret[f'Hits rate{k}'] = epresults[f'Hit rate{k}'] / num_true_samples
            ret[f'NDCG{k}'] = epresults[f'NDCG{k}'] / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds, topk):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = hits = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    hits += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg, hits

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('./History/' + args.save_path + '.txt', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        torch.save(content, './Models/' + args.save_path + '.mod')
        torch.save(self.model.state_dict(), './Models/' + args.save_path + '.pt')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        ckp = torch.load('./Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('./History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


    def saveRecord(self, reses, fileName):
        pass


if __name__ == '__main__':
    logger.saveDefault = True
    log('Start')
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    coach = Coach(handler)
    coach.run()
