import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_model
from engine.processor import do_inference,training_neat_eval,do_inference_mismatch
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDReID Testing")
    parser.add_argument(
        "--config_file", default="/home/tan/data/fyy_temp/code/MDReID-master_new/configs/RGBNT201/MDReID.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    model_path = "/data/RNTReID/result/w2_1.5_5.25/RGBNT201_w2=5.25/MDReIDbest.pth"

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("MDReID", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.eval()
    model.load_param(model_path)

    # from thop import profile
    # import torch
    # number=1
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # data = torch.randn([number, 3, 256, 128]).to(device)
    # img = {'RGB': data,'NI': data,'TI': data}
    # camids = torch.ones([number, ]).to(torch.int64).to(device)
    # target_view = torch.ones([number, ]).to(torch.int64).to(device)
    # return_pattern = 3
    # flops, params = profile(model, inputs=(img, None, camids, -1*target_view, return_pattern,))
    # print(flops / 1e9, params / 1e6)

    # do_inference(cfg, model, val_loader, num_query, return_pattern=3)
    do_visualize(cfg, model, val_loader, num_query, return_pattern=3)

    # mismatch validation
    # NT->RT: [1,2]->[0,2]; NT->NR: [1,2]->[1,0]; NT->R: [1,2]->[0]
    # RT->RN: [0,2]->[0,1]; RT->NT: [0,2]->[1,2]; RT->N: [0,2]->[1]
    # RN->RT: [0,1]->[0,2]; RN->TN: [0,1]->[2,1]; RN->T: [0,1]->[2]
    # T->R: [2]->[0]; T->N: [2]->[1]; T->RN: [2]->[0,1]
    # N->R: [1]->[0]; N->T: [1]->[2]; N->RT: [1]->[0,2]
    # R->N: [0]->[1]; R->T: [0]->[2]; R->NT: [0]->[1,2]

    logger.info("NT->NT:\n")
    query_ID = torch.tensor([1,2], dtype=torch.int)
    gallery_ID = torch.tensor([1,2], dtype=torch.int)
    do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)

    logger.info("RT->RT:\n")
    query_ID = torch.tensor([0,2], dtype=torch.int)
    gallery_ID = torch.tensor([0,2], dtype=torch.int)
    do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)

    logger.info("RN->RN:\n")
    query_ID = torch.tensor([0,1], dtype=torch.int)
    gallery_ID = torch.tensor([0,1], dtype=torch.int)
    do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)

    logger.info("T->T:\n")
    query_ID = torch.tensor([2], dtype=torch.int)
    gallery_ID = torch.tensor([2], dtype=torch.int)
    do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)

    logger.info("N->N:\n")
    query_ID = torch.tensor([1], dtype=torch.int)
    gallery_ID = torch.tensor([1], dtype=torch.int)
    do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)

    logger.info("R->R:\n")
    query_ID = torch.tensor([0], dtype=torch.int)
    gallery_ID = torch.tensor([0], dtype=torch.int)
    do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)

    # logger.info("RNT->RNT:\n")
    # query_ID = torch.tensor([0,1,2], dtype=torch.int)
    # gallery_ID = torch.tensor([0,1,2], dtype=torch.int)
    # do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)
    #
    # logger.info("RT->RT:\n")
    # query_ID = torch.tensor([0,2], dtype=torch.int)
    # gallery_ID = torch.tensor([0,2], dtype=torch.int)
    # do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)
    #
    # logger.info("RT->NT:\n")
    # query_ID = torch.tensor([0,2], dtype=torch.int)
    # gallery_ID = torch.tensor([1,2], dtype=torch.int)
    # do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)
    #
    # logger.info("RT->N:\n")
    # query_ID = torch.tensor([0,2], dtype=torch.int)
    # gallery_ID = torch.tensor([1], dtype=torch.int)
    # do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)
    #
    # logger.info("R->N:\n")
    # query_ID = torch.tensor([0], dtype=torch.int)
    # gallery_ID = torch.tensor([1], dtype=torch.int)
    # do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)
    #
    # logger.info("R->NT:\n")
    # query_ID = torch.tensor([0], dtype=torch.int)
    # gallery_ID = torch.tensor([1,2], dtype=torch.int)
    # do_inference_mismatch(cfg, model, val_loader, num_query, return_pattern=3, query_ID=query_ID, gallery_ID=gallery_ID)

