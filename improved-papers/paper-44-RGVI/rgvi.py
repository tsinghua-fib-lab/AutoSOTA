from fcnet import FCNet
from pfcnet import PFCNet
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from diffusers import StableDiffusionInpaintPipeline as SDI
from diffusers import DPMSolverMultistepScheduler


def backward_warp(x, flow):
    B, _, H, W = x.size()
    x = x.cuda()
    flow = flow.cuda()
    grid_h = torch.arange(0, H).view(1, H, 1).repeat(B, 1, W)
    grid_w = torch.arange(0, W).view(1, 1, W).repeat(B, H, 1)
    grid = torch.stack([grid_w, grid_h], 3).type_as(x)
    grid_flow = grid + flow.permute(0, 2, 3, 1)
    grid_flow_w = 2 * grid_flow[:, :, :, 0] / (W - 1) - 1
    grid_flow_h = 2 * grid_flow[:, :, :, 1] / (H - 1) - 1
    norm_grid_flow = torch.stack([grid_flow_w, grid_flow_h], dim=3)
    return F.grid_sample(x, norm_grid_flow, align_corners=True).cpu()


# RGVI model
class RGVI(nn.Module):
    def __init__(self):
        super().__init__()
        self.raft = tv.models.optical_flow.raft_large(pretrained=True)
        self.fcnet = FCNet('weights/FCNet.pth')
        self.sdi = SDI.from_pretrained('/sd2_inpaint', torch_dtype=torch.float16).to('cuda')
        self.sdi.scheduler = DPMSolverMultistepScheduler.from_config(self.sdi.scheduler.config, use_karras_sigmas=True)
        self.pfcnet = PFCNet('weights/PFCNet.pth')

    def forward(self, imgs, neg_masks, pos_masks, res, prompt):
        L, _, _, _ = imgs.size()

        # set resolution
        if res == '240p':
            H, W = 240, 432
        if res == '480p':
            H, W = 480, 864
        if res == '2K':
            H, W = 1200, 2160

        # resize input
        imgs = F.interpolate(imgs, size=(H, W), mode='bicubic', antialias=True)
        neg_masks = F.interpolate(neg_masks, size=(H, W), mode='bicubic', antialias=True)
        if pos_masks is not None:
            pos_masks = F.interpolate(pos_masks, size=(H, W), mode='bicubic', antialias=True)

        # remove positive mask as well
        if pos_masks is not None:
            masks = neg_masks + pos_masks
        else:
            masks = neg_masks
        masks = (masks != 0).float()
        cnts = 1 - masks

        # memorize original image
        org_imgs = imgs.clone()

        # optical flow generation (maximum 480p)
        fw_flows = {}
        bw_flows = {}
        for i in range(1, L):
            prev_img = F.interpolate(imgs[i - 1:i], size=(480, 864), mode='bicubic', antialias=True)
            curr_img = F.interpolate(imgs[i:i + 1], size=(480, 864), mode='bicubic', antialias=True)
            fw_flows[i - 1] = self.raft(2 * prev_img.cuda() - 1, 2 * curr_img.cuda() - 1)[-1].cpu()
            bw_flows[i] = self.raft(2 * curr_img.cuda() - 1, 2 * prev_img.cuda() - 1)[-1].cpu()

        # input masking
        imgs = imgs * cnts

        # flow completion (240p)
        s = H / 240
        fcnet_masks = F.interpolate(masks, size=(240, 432), mode='bicubic', antialias=True)
        fcnet_masks = F.avg_pool2d(fcnet_masks, 9, 1, 4)
        fcnet_masks = (fcnet_masks != 0).float().unsqueeze(0)
        fcnet_fw_flows = torch.zeros(1, L - 1, 2, 240, 432)
        fcnet_bw_flows = torch.zeros(1, L - 1, 2, 240, 432)
        for i in range(L - 1):
            fcnet_fw_flows[:, i] = F.interpolate(fw_flows[i], size=(240, 432), mode='bicubic', antialias=True) / 2
            fcnet_bw_flows[:, i] = F.interpolate(bw_flows[i + 1], size=(240, 432), mode='bicubic', antialias=True) / 2
        fcnet_fw_flows = (1 - fcnet_masks[:, :-1]) * fcnet_fw_flows
        fcnet_bw_flows = (1 - fcnet_masks[:, 1:]) * fcnet_bw_flows
        fcnet_flows = [fcnet_fw_flows.cuda(), fcnet_bw_flows.cuda()]
        fcnet_inp_flows = self.fcnet.forward_bidirect_flow(fcnet_flows, fcnet_masks.cuda())
        fcnet_inp_flows = self.fcnet.combine_flow(fcnet_flows, fcnet_inp_flows, fcnet_masks.cuda())

        # modify output to our format
        inp_fw_flows = {}
        inp_bw_flows = {}
        for i in range(L - 1):
            inp_fw_flows[i] = fcnet_inp_flows[0][:, i].cpu()
            inp_bw_flows[i + 1] = fcnet_inp_flows[1][:, i].cpu()

        # internal pixel propagation
        fw_imgs = imgs.clone()
        bw_imgs = imgs.clone()
        fw_cnts = cnts.clone()
        bw_cnts = cnts.clone()
        warp_masks = torch.zeros(L, 1, H, W)
        for i in range(L):

            # pulling from forward direction
            for j in range(i + 1, L):
                if j == i + 1:
                    acc_flow = inp_fw_flows[i]
                else:
                    acc_flow = backward_warp(inp_fw_flows[j - 1], acc_flow) + acc_flow
                acc_flow_s = F.interpolate(acc_flow, scale_factor=s, mode='bicubic', antialias=True) * s
                warp_img = backward_warp(imgs[j:j + 1], acc_flow_s)[0]
                warp_cnt = backward_warp(cnts[j:j + 1], acc_flow_s)[0]
                fw_imgs[i] = fw_imgs[i] + (1 - fw_cnts[i]) * warp_img
                fw_cnts[i] = fw_cnts[i] + (1 - fw_cnts[i]) * warp_cnt
                warp_masks[i] = warp_masks[i] + 1 - warp_cnt

            # pulling from backward direction
            for j in range(i - 1, -1, -1):
                if j == i - 1:
                    acc_flow = inp_bw_flows[i]
                else:
                    acc_flow = backward_warp(inp_bw_flows[j + 1], acc_flow) + acc_flow
                acc_flow_s = F.interpolate(acc_flow, scale_factor=s, mode='bicubic', antialias=True) * s
                warp_img = backward_warp(imgs[j:j + 1], acc_flow_s)[0]
                warp_cnt = backward_warp(cnts[j:j + 1], acc_flow_s)[0]
                bw_imgs[i] = bw_imgs[i] + (1 - bw_cnts[i]) * warp_img
                bw_cnts[i] = bw_cnts[i] + (1 - bw_cnts[i]) * warp_cnt
                warp_masks[i] = warp_masks[i] + 1 - warp_cnt

        # invalidate incomplete propagation
        fw_imgs[fw_cnts.repeat(1, 3, 1, 1) != 1] = 0
        fw_cnts[fw_cnts != 1] = 0
        bw_imgs[bw_cnts.repeat(1, 3, 1, 1) != 1] = 0
        bw_cnts[bw_cnts != 1] = 0

        # collect both directions
        imgs = (fw_imgs + bw_imgs) / (fw_cnts + bw_cnts).clamp(1e-7)
        masks = 1 - (fw_cnts + bw_cnts).clamp(0, 1)
        cnts = 1 - masks

        # propagation verification
        threshold = 1
        diff = torch.sum(abs(fw_imgs - bw_imgs), dim=1, keepdim=True)
        unsure = (diff > threshold).float() * (fw_cnts + bw_cnts - 1).clamp(0, 1)

        # count connected pixels
        con_num = torch.zeros(L)
        for i in range(L):
            con_num[i] = con_num[i] + torch.sum(masks[i]) + torch.sum(masks[i] * warp_masks[i])

        # select target frame to fill
        k = int(torch.argmax(con_num, dim=0))
        if 1 in masks[k].unique():

            # detach box for generation
            bbox = tv.ops.masks_to_boxes(masks[k])[0]
            if prompt is None:
                x1 = 0
                x2 = W
                y1 = 0
                y2 = H
            else:
                x1 = int(max(bbox[0] - 20 * s, 0))
                x2 = int(min(bbox[2] + 20 * s, W))
                y1 = int(max(bbox[1] - 20 * s, 0))
                y2 = int(min(bbox[3] + 20 * s, H))
            crop_img = imgs[k, :, y1:y2, x1:x2]
            crop_mask = masks[k, :, y1:y2, x1:x2]

            # generate reference frame
            img = tv.transforms.ToPILImage()(crop_img)
            mask = tv.transforms.ToPILImage()(crop_mask)
            if prompt is None:
                prompt = 'Empty background, high resolution'
            # Upscale to 512x512 for SD2's native resolution, then resize back
            orig_w, orig_h = img.size
            sd2_img = img.resize((512, 512), Image.BICUBIC)
            sd2_mask = mask.resize((512, 512), Image.NEAREST)
            # Multi-seed ensemble: generate 3 candidates, average in masked region
            import numpy as np
            seeds = [2024, 42, 1234, 777, 9999, 314, 8888]
            sum_arr = None
            for seed in seeds:
                generator = torch.Generator('cuda').manual_seed(seed)
                candidate = self.sdi(prompt=prompt, image=sd2_img, mask_image=sd2_mask, generator=generator, num_inference_steps=50).images[0]
                cand_np = np.array(candidate).astype(np.float32)
                if sum_arr is None:
                    sum_arr = cand_np
                else:
                    sum_arr = sum_arr + cand_np
            avg_np = np.clip(sum_arr / len(seeds), 0, 255).astype(np.uint8)
            from PIL import Image as PILImage
            sd2_out = PILImage.fromarray(avg_np)
            out = sd2_out.resize((orig_w, orig_h), Image.BICUBIC)
            imgs[k, :, y1:y2, x1:x2] = imgs[k, :, y1:y2, x1:x2] + masks[k, :, y1:y2, x1:x2] * tv.transforms.ToTensor()(out)
            cnts[k] = 1

            # pulling from forward direction
            for i in range(k - 1, -1, -1):
                if i == k - 1:
                    acc_flow = inp_fw_flows[i]
                else:
                    acc_flow = backward_warp(acc_flow, inp_fw_flows[i]) + inp_fw_flows[i]
                acc_flow_s = F.interpolate(acc_flow, scale_factor=s, mode='bicubic', antialias=True) * s
                warp_img = backward_warp(imgs[k:k + 1], acc_flow_s)[0]
                warp_cnt = backward_warp(cnts[k:k + 1], acc_flow_s)[0]
                imgs[i] = imgs[i] + (1 - cnts[i]) * warp_img
                cnts[i] = cnts[i] + (1 - cnts[i]) * warp_cnt

            # pulling from backward direction
            for i in range(k + 1, L):
                if i == k + 1:
                    acc_flow = inp_bw_flows[i]
                else:
                    acc_flow = backward_warp(acc_flow, inp_bw_flows[i]) + inp_bw_flows[i]
                acc_flow_s = F.interpolate(acc_flow, scale_factor=s, mode='bicubic', antialias=True) * s
                warp_img = backward_warp(imgs[k:k + 1], acc_flow_s)[0]
                warp_cnt = backward_warp(cnts[k:k + 1], acc_flow_s)[0]
                imgs[i] = imgs[i] + (1 - cnts[i]) * warp_img
                cnts[i] = cnts[i] + (1 - cnts[i]) * warp_cnt

            # invalidate incomplete propagation
            imgs[cnts.repeat(1, 3, 1, 1) != 1] = 0
            cnts[cnts != 1] = 0

        # propagation verification
        imgs = imgs * (1 - unsure)
        masks = 1 - cnts * (1 - unsure)

        # missing area completion
        for i in range(L):
            if 1 in masks[i].unique():
                imgs[i:i + 1] = self.pfcnet(imgs[i:i + 1].cuda(), masks[i:i + 1].cuda()).cpu()

        # attach back positive masks
        if pos_masks is not None:
            imgs = (1 - pos_masks) * imgs + pos_masks * org_imgs
        return imgs
