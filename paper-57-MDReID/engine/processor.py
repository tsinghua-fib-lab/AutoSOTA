import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist
from layers.supcontrast import SupConLoss
from layers.triplet_loss import EnhancedTripletLoss


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 2) +
                     epsilon, 0.5).unsqueeze(2).expand_as(feature)
    return torch.div(feature, norm)


def orthonomal_loss(w):
    B, K, C = w.shape
    w_norm = featureL2Norm(w)
    WWT = torch.matmul(w_norm, w_norm.transpose(1, 2))
    AIM1 = torch.cat((torch.eye(K // 2, K // 2), torch.zeros(K // 2, K // 2)), dim=1)
    AIM2 = torch.cat((torch.zeros(K // 2, K // 2), torch.ones(K // 2, K // 2)), dim=1)
    AIM = torch.cat((AIM1, AIM2), dim=0).to(w.device)
    # AIM =torch.eye(K).unsqueeze(0).cuda()
    return torch.nn.functional.mse_loss(WWT - AIM, torch.zeros(B, K, K).to(w.device), size_average=False) / (K * K)


def CoRefine(output1, output2):
    output1_prob = output1 / output1.norm(dim=-1, keepdim=True)
    output2_prob = output2 / output2.norm(dim=-1, keepdim=True)

    kl = torch.bmm(output2_prob, output1_prob.permute(0, 2, 1))

    return kl


def cal_loss(loss_fn, output, target, target_cam, s_loc=-1):
    loss = 0
    for i in range(0, len(output), 2):
        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
        if i == s_loc:
            loss_tmp = 3 * loss_tmp
        loss = loss + loss_tmp
    return loss


def compute_sdm(fetures1, fetures2, pid, logit_scale=50, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = fetures1.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    fetures1_norm = fetures1 / fetures1.norm(dim=1, keepdim=True)
    fetures2_norm = fetures2 / fetures2.norm(dim=1, keepdim=True)

    t2i_cosine_theta = fetures2_norm @ fetures1_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = torch.nn.functional.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (
                torch.nn.functional.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = torch.nn.functional.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (
                torch.nn.functional.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda:" + cfg.MODEL.DEVICE_ID
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("MDReID.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(cfg.MODEL.DEVICE_ID)],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    tloss_meter = EnhancedTripletLoss(cfg.SOLVER.MARGIN)

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output = model(img, label=target, cam_label=target_cam, view_label=target_view)
                if cfg.MODEL.DIRECT:
                    if cfg.MODEL.ADD_SHARE is True:
                        ori_cls = output[1][:, :output[1].shape[1] // 2]
                        ori_share = output[1][:, output[1].shape[1] // 2:]
                    else:
                        ori_cls = output[1]
                    loss = cal_loss(loss_fn, output, target, target_cam)
                else:
                    ori_cls = torch.cat([output[1], output[3], output[5]], dim=-1)
                    if cfg.MODEL.ADD_SHARE is True:
                        ori_share = output[7]
                        loss = cal_loss(loss_fn, output[0:8], target, target_cam, 6)
                    else:
                        loss = cal_loss(loss_fn, output[0:6], target, target_cam)

                if cfg.MODEL.ADD_SHARE is True:
                    rnts = torch.chunk(ori_share, chunks=3, dim=1)
                    rnt = torch.chunk(ori_cls, chunks=3, dim=1)

                if cfg.MODEL.ADD_CLOSS is True:
                    lambda1 = 1.5
                    loss_a = 0
                    for i in range(ori_cls.shape[0]):
                        A_cls = torch.cat(
                            (rnt[0][i, :].unsqueeze(0), rnt[1][i, :].unsqueeze(0), rnt[2][i, :].unsqueeze(0)), dim=0)
                        A_s = torch.cat(
                            (rnts[0][i, :].unsqueeze(0), rnts[1][i, :].unsqueeze(0), rnts[2][i, :].unsqueeze(0)), dim=0)

                        W = CoRefine(torch.cat((A_cls, A_s), dim=0).unsqueeze(0),
                                     torch.cat((A_cls, A_s), dim=0).unsqueeze(0))
                        loss_a = loss_a + orthonomal_loss(W)
                    loss = loss + lambda1 * loss_a / ori_cls.shape[0]

                if cfg.MODEL.ADD_ELOSS is True:
                    lambda2 = 5.25
                    loss_e = tloss_meter(torch.cat([output[9], output[11], output[13]], dim=-1), ori_cls,
                                         torch.cat([rnts[0], rnts[1], rnts[2]], dim=-1), target)
                    loss = loss + lambda2 * loss_e

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger)
            else:
                mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                              return_pattern=3)
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                logger.info("~" * 50)
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                logger.info("~" * 50)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, return_pattern=1):
    device = "cuda:" + cfg.MODEL.DEVICE_ID
    logger = logging.getLogger("MDReID.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=True)
        evaluator.reset()
    if device:
        model.to(device)

    model.eval()
    img_path_list = []
    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            print(imgpath)
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feats = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern,
                          img_path=imgpath)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feats, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feats, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


from sklearn import manifold
import time
import matplotlib.pyplot as plt
import numpy as np

def visualize(features, labels, cameras, save_path='./save_tsne/', name='tnse.png'):
    """
    行人重识别特征可视化函数
    参数：
        features: 提取的特征向量(N, 512)
        labels: 对应的行人ID标签(N,)
        cameras: 摄像机编号(N,)
        save_path: 结果保存路径
    """
    # 创建保存目录
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # 数据标准化
    features = torch.nn.functional.normalize(features)

    # 使用t-SNE降维
    tsne = manifold.TSNE(n_components=2,
                         perplexity=30,
                         learning_rate=200,
                         init='pca',
                         random_state=42)

    start_time = time.time()
    embeddings = tsne.fit_transform(features.cpu().numpy())
    print(f't-SNE耗时: {time.time() - start_time:.2f}秒')

    # 可视化设置
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    plt.style.use('ggplot')  # 使用更清晰的背景样式

    # 颜色和形状配置
    unique_labels = np.unique(labels)
    color_palette = plt.cm.tab10  # 改用对比度更高的色板
    color_list = [color_palette(i % 10) for i in range(len(unique_labels))]
    # color_list = [tuple(np.array(c) * 0.85) for c in color_list]  # 统一加深颜色

    # 摄像机形状编码（6种预定义形状）
    camera_symbols = ['^', 'o', 'X', '^', 'o', 'X']
    symbol_sizes = [50, 50, 50, 50, 50, 50]  # 对应不同形状的尺寸，前三小后三大
    assert len(np.unique(cameras)) == 6, "数量必须为6个"

    # 绘制散点图
    for label_idx, label in enumerate(unique_labels):
        # 按行人ID筛选
        label_mask = labels == label
        label_cameras = cameras[label_mask]

        # 按摄像机类型绘制不同形状
        for cam in np.unique(label_cameras):
            cam_idx = int(cam)
            cam_mask = label_cameras == cam
            if cam_idx <= 2:
                ax.scatter(
                    embeddings[label_mask][cam_mask, 0],
                    embeddings[label_mask][cam_mask, 1],
                    color=color_list[label_idx],
                    marker=camera_symbols[cam_idx],  # 直接通过摄像机ID索引形状
                    s=symbol_sizes[cam_idx],  # 调整符号大小
                    alpha=0.9,  # 透明
                    edgecolors='none',  # 边框
                    label=f'ID{label}_Cam{cam}' if cam == np.unique(label_cameras)[0] else None
                )
            else:
                ax.scatter(
                    embeddings[label_mask][cam_mask, 0],
                    embeddings[label_mask][cam_mask, 1],
                    color=color_list[label_idx],
                    marker=camera_symbols[cam_idx],  # 直接通过摄像机ID索引形状
                    s=symbol_sizes[cam_idx],  # 调整符号大小
                    alpha=0.9,  # 透明
                    edgecolors='black',  # 边框
                    linewidths=0.6,  # 边框粗细
                    label=f'ID{label}_Cam{cam}' if cam == np.unique(label_cameras)[0] else None
                )

    # # 图例优化（按类别+摄像机组合）
    # handles, labels = ax.get_legend_handles_for_labels()
    # plt.legend(handles[:6 * 2], labels[:6 * 2], ncol=3, fontsize=9)  # 显示前12个图例项

    # 坐标轴美化
    # plt.title('t-SNE Visualization', fontsize=14)
    # plt.xlabel('t-SNE Dimension 1', fontsize=12)
    # plt.ylabel('t-SNE Dimension 2', fontsize=12)

    # 保存高清图像
    plt.savefig(os.path.join(save_path, name),
                dpi=400,
                bbox_inches='tight',
                facecolor='white')  # 强制白底
    plt.close()

def do_visualize(cfg,
                 model,
                 val_loader,
                 num_query, return_pattern=1):
    device = "cuda:" + cfg.MODEL.DEVICE_ID
    logger = logging.getLogger("MDReID.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    if device:
        model.to(device)

    model.eval()
    features = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    cameras = torch.tensor([]).to(device)
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            print(imgpath)
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            target_view = target_view.to(device)
            feats = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern,
                          img_path=imgpath)
            feats_new = torch.chunk(feats, chunks=6, dim=1)
            RNT_L = torch.cat(
                [torch.zeros([feats_new[0].shape[0], ]), torch.ones([feats_new[0].shape[0], ]), 2 * torch.ones([feats_new[0].shape[0], ]),
                 3 * torch.ones([feats_new[0].shape[0], ]), 4 * torch.ones([feats_new[0].shape[0], ]),
                 5 * torch.ones([feats_new[0].shape[0], ])], dim=0).to(device)
            features = torch.cat([features, torch.cat(feats_new, dim=0)], dim=0)
            labels = torch.cat([labels, torch.cat(
                [torch.tensor(pid).to(device), torch.tensor(pid).to(device), torch.tensor(pid).to(device),
                 torch.tensor(pid).to(device), torch.tensor(pid).to(device), torch.tensor(pid).to(device)], dim=0)],
                               dim=0)
            cameras = torch.cat([cameras, RNT_L], dim=0)
    point_num = cfg.TEST.IMS_PER_BATCH*4*6
    visualize(features[:point_num,].to("cpu"), labels[:point_num,].to("cpu"), cameras[:point_num,].to("cpu"), save_path='/home/tan/data/fyy_temp/result/vis/', name='tsne.svg')
    return

def do_inference_mismatch(cfg,
                          model,
                          val_loader,
                          num_query, return_pattern=1, query_ID=torch.tensor([1, 2], dtype=torch.int),
                          gallery_ID=torch.tensor([0, 2], dtype=torch.int)):
    device = "cuda:" + cfg.MODEL.DEVICE_ID
    logger = logging.getLogger("MDReID.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=True)
        evaluator.reset()
    if device:
        model.to(device)

    model.eval()
    img_path_list = []
    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)
    g_feats = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            print(imgpath)
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)

            # query feat
            model.miss_type = ID_to_MISS(query_ID)
            q_feats = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern,
                            img_path=imgpath)

            # gallery feat
            model.miss_type = ID_to_MISS(gallery_ID)
            g_feat = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern,
                           img_path=imgpath)

            if cfg.MODEL.ADD_SHARE is True:
                q_feats = transform_index(query_ID, q_feats)
                g_feat = transform_index(gallery_ID, g_feat)
                q_feats, g_feat = select_feature(query_ID, gallery_ID, q_feats, g_feat)
            else:
                q_feats = transform_index_org(query_ID, q_feats)
                g_feat = transform_index_org(gallery_ID, g_feat)

            g_feats.append(g_feat)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((q_feats, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((q_feats, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.mismatch_compute(g_feats)
    logger.info("Mismatch Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    return cmc[0], cmc[4]


def ID_to_MISS(ID):
    miss_type = "nothing"
    if torch.equal(ID, torch.tensor([1, 2], dtype=torch.int)) or torch.equal(ID, torch.tensor([2, 1], dtype=torch.int)):
        miss_type = "r"
    if torch.equal(ID, torch.tensor([0, 2], dtype=torch.int)) or torch.equal(ID, torch.tensor([2, 0], dtype=torch.int)):
        miss_type = "n"
    if torch.equal(ID, torch.tensor([0, 1], dtype=torch.int)) or torch.equal(ID, torch.tensor([1, 0], dtype=torch.int)):
        miss_type = "t"
    if torch.equal(ID, torch.tensor([2], dtype=torch.int)):
        miss_type = "rn"
    if torch.equal(ID, torch.tensor([1], dtype=torch.int)):
        miss_type = "rt"
    if torch.equal(ID, torch.tensor([0], dtype=torch.int)):
        miss_type = "nt"
    return miss_type


def transform_index_org(ID, feats):
    feats_out = feats
    if len(ID) > 1:
        is_increasing = True
        for i in range(len(ID) - 1):
            if ID[i] >= ID[i + 1]:
                is_increasing = False
                break
        _, indices = torch.sort(ID)
        ID_new = torch.arange(len(ID))[indices]

        if is_increasing is True:
            feats_all_swap = []
            feats_all = torch.chunk(feats, chunks=len(ID_new), dim=1)
            for i in range(len(ID_new)):
                feats_all_swap.append(feats_all[ID_new[i]])
            feats_out = torch.cat(feats_all_swap, dim=1).to(feats.device)
    return feats_out


def transform_index(ID, feats):
    feats_out = feats
    if len(ID) > 1:
        is_increasing = True
        for i in range(len(ID) - 1):
            if ID[i] >= ID[i + 1]:
                is_increasing = False
                break
        _, indices = torch.sort(ID)
        ID_new = torch.arange(len(ID))[indices]

        if is_increasing is True:
            feats_all_swap = []
            feats_all = torch.chunk(feats, chunks=len(ID_new) * 2, dim=1)
            for i in range(len(ID_new) * 2):
                if i < len(ID_new):
                    feats_all_swap.append(feats_all[ID_new[i]])
                else:
                    feats_all_swap.append(feats_all[ID_new[i - len(ID_new)] + len(ID_new)])
            feats_out = torch.cat(feats_all_swap, dim=1).to(feats.device)
    return feats_out


def select_feature(query_ID, gallery_ID, q_feats, g_feats):
    if torch.equal(query_ID, gallery_ID) is False:
        q_feats_r = []
        g_feats_r = []
        q_feats_s = q_feats[:, q_feats.shape[1] // 2:]
        g_feats_s = g_feats[:, g_feats.shape[1] // 2:]
        c_index = torch.arange(len(query_ID))[torch.where(query_ID == gallery_ID)[0]]
        if len(c_index) > 0:
            qfeats_all = torch.chunk(q_feats, chunks=len(query_ID) * 2, dim=1)
            gfeats_all = torch.chunk(g_feats, chunks=len(gallery_ID) * 2, dim=1)
            for i in range(len(c_index)):
                q_feats_r.append(qfeats_all[c_index[i]])
                g_feats_r.append(gfeats_all[c_index[i]])
            q_feats_r = torch.cat([torch.cat(q_feats_r, dim=1), q_feats_s], dim=1)
            g_feats_r = torch.cat([torch.cat(g_feats_r, dim=1), g_feats_s], dim=1)
        else:
            q_feats_r = q_feats_s
            g_feats_r = g_feats_s

        if q_feats_r.shape[1] > g_feats_r.shape[1]:
            g_feats_r = torch.cat(
                [g_feats_r, g_feats_r[:, g_feats_r.shape[1] - (q_feats_r.shape[1] - g_feats_r.shape[1]):]], dim=1)
        if q_feats_r.shape[1] < g_feats_r.shape[1]:
            q_feats_r = torch.cat(
                [q_feats_r, q_feats_r[:, q_feats_r.shape[1] - (g_feats_r.shape[1] - q_feats_r.shape[1]):]], dim=1)
        return q_feats_r.to(q_feats.device), g_feats_r.to(g_feats.device)
    else:
        return q_feats, g_feats


def simplify_index(query_ID, gallery_ID, q_feats, g_feats):
    # if len(query_ID) > len(gallery_ID):
    #     q_feats_all = torch.chunk(q_feats, chunks=(len(query_ID)+3), dim=1)
    #     q_feats_new = torch.tensor([]).to(q_feats.device)
    #     for i in range(len(q_feats_all)):
    #         if i <= abs(len(query_ID) - len(gallery_ID)) and i > 0:
    #             q_feats_new = q_feats_new + q_feats_all[i]
    #             if i == abs(len(query_ID) - len(gallery_ID)):
    #                 q_feats_new = q_feats_new/(abs(len(query_ID) - len(gallery_ID))+1)
    #         else:
    #             q_feats_new = torch.cat((q_feats_new, q_feats_all[i]), dim=1)
    #     return q_feats_new, g_feats
    # if len(query_ID) < len(gallery_ID):
    #     g_feats_all = torch.chunk(g_feats, chunks=(len(gallery_ID)+3), dim=1)
    #     g_feats_new = torch.tensor([]).to(g_feats.device)
    #     for i in range(len(g_feats_all)):
    #         if i <= abs(len(query_ID) - len(gallery_ID)) and i > 0:
    #             g_feats_new = g_feats_new + g_feats_all[i]
    #             if i == abs(len(query_ID) - len(gallery_ID)):
    #                 g_feats_new = g_feats_new/(abs(len(query_ID) - len(gallery_ID))+1)
    #         else:
    #             g_feats_new = torch.cat((g_feats_new, g_feats_all[i]), dim=1)
    #     return q_feats, g_feats_new
    if len(query_ID) > len(gallery_ID):
        g_feats_all = torch.chunk(g_feats, chunks=(len(gallery_ID) + 3), dim=1)
        g_feats_new = g_feats
        for i in range(abs(len(query_ID) - len(gallery_ID))):
            g_feats_new = torch.cat((g_feats_all[0], g_feats_new), dim=1)
        return q_feats, g_feats_new

    if len(query_ID) < len(gallery_ID):
        q_feats_all = torch.chunk(q_feats, chunks=(len(query_ID) + 3), dim=1)
        q_feats_new = q_feats
        for i in range(abs(len(query_ID) - len(gallery_ID))):
            q_feats_new = torch.cat((q_feats_all[0], q_feats_new), dim=1)
        return q_feats_new, g_feats

    return q_feats, g_feats


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator, epoch, logger, return_pattern=1):
    evaluator.reset()
    model.eval()
    logger.info("~" * 50)
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, scenceids, _))
            else:
                evaluator.update((feat, vid, camid, _))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("~" * 50)
    torch.cuda.empty_cache()
    return mAP, cmc
