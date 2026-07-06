import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

from prompts.imagenet_template import get_text_template

from dinov3.utils.utils import UnNormalize, PAMR
from torchvision import transforms

from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import InterpolationMode


import sys
sys.path.append("..")


def resolve_text_template(text_template):
    if isinstance(text_template, str):
        return get_text_template(text_template)

    if isinstance(text_template, (list, tuple)) and all(callable(temp) for temp in text_template):
        return list(text_template)

    raise TypeError(
        'text_template must be a template name or a list/tuple of callables.')



@MODELS.register_module()
class DinotxtSegmentation(BaseSegmentor):
    def __init__(self, ckpt_path, name_path, cfg=None, background=False, pamr_steps=0, pamr_stride=(8,16),
                prob_thd=0.0, logit_scale=40, area_thd=None, slide_stride=112, slide_crop=336, bg_idx=0, tau=4.0,
                tem=1.0, text_template='openai_imagenet_template'):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        device=torch.device('cuda')

        self.cfg = cfg

        self.query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(self.query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        self.text_template = resolve_text_template(text_template)

        model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(ckpt_path)
        self.dinotxt = model.to(device).bfloat16()
        self.tokenizer = tokenizer

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        query_features = []
        # total_features = []
        with torch.autocast(dtype=torch.float16, device_type="cuda"):
            with torch.no_grad():
                for i,qw in enumerate(self.query_words):
                    text_prompt = [temp(qw) for temp in self.text_template]
                    query = self.tokenizer.tokenize(text_prompt).to(device)
                    total_feat = self.dinotxt.encode_text(query) # Part of text features that is aligned to patch feat

                    feature = total_feat[:, 1024:].clone()
                    feature /= feature.norm(dim=-1, keepdim=True)
                    query_features.append(feature.unsqueeze(0))


        self.query_features = torch.cat(query_features, dim=0).detach()

        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.bg_idx = bg_idx

        self.background = background
        self.tau = tau
        self.tem = tem

        if self.background:
            self.num_bkg_prompts = 1
        else:
            self.num_bkg_prompts = 0


        if pamr_steps > 0:
            self.dtype = self.query_features.dtype
            self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device)
        else:
            self.pamr = None

        # Gaussian weight map for sliding window blending (Idea-04)
        # Creates a 2D Gaussian centered on the crop for weighted overlap averaging
        self._create_slide_weight_map(device)



    def _create_slide_weight_map(self, device):
        """Create a 2D Gaussian weight map for sliding window blending.
        Pixels near the crop center get higher weight than boundary pixels,
        reducing boundary artifacts from uniform overlap averaging."""
        h, w = self.slide_crop, self.slide_crop
        sigma = min(h, w) / 3.0
        y = torch.arange(h, dtype=torch.float32, device=device)
        x = torch.arange(w, dtype=torch.float32, device=device)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        gy = torch.exp(-((y - cy) ** 2) / (2.0 * sigma ** 2))
        gx = torch.exp(-((x - cx) ** 2) / (2.0 * sigma ** 2))
        weight = torch.outer(gy, gx)
        weight = weight / weight.max()  # normalize to [0, 1]
        self.slide_weight_map = weight.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]

    @torch.inference_mode()
    def forward_feature(self, img, logit_size=None, img_metas=None, gt_masks=None):
        if type(img) == list:
            img = img[0]
        
        if self.slide_crop == 0:
            device = img.device
            img = _transform1(336)(img).to(device)
            

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.bfloat16()

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):

            image_class_tokens, image_features = self.dinotxt.encode_image_with_patch_tokens(imgs_norm)


            image_features /= image_features.norm(dim=-1, keepdim=True) # b,hw,d

        
            patch_size = self.dinotxt.visual_model.backbone.patch_size
            I, J = imgs_norm.shape[-2] // patch_size, imgs_norm.shape[-2] // patch_size 
            
            logits = torch.einsum('bnd,qkd->bnqk', image_features, self.query_features)
            logits = logits.mean(dim=-1)
    
            logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], I, J) 

            avg_feat = image_features.mean(dim=1)
            avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
            text_feat = self.query_features.mean(dim=1)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            cls_scores = (avg_feat @ text_feat.T).squeeze() / self.tem

            prob_logits = logits * self.logit_scale
            prob_logits = prob_logits.squeeze()

            map_logits = torch.zeros(self.num_classes, prob_logits.shape[1], prob_logits.shape[2]).to(prob_logits.device)
            for j in range(self.num_classes):
                weights = cls_scores[self.query_idx==j]
                weights = torch.softmax(weights,dim=0)
                alias_logits = prob_logits[self.query_idx==j, :, :].clone()  # [k_j, H, W]
                alias_logits = alias_logits * weights[:, None, None] * (1/weights.mean())
                map_logits[j] = (1/self.tau) * torch.logsumexp(self.tau * alias_logits, dim=0)

            logits = map_logits.unsqueeze(0)

            if logit_size == None:
                logits = F.interpolate(logits, size=img.shape[-2:], mode='bilinear')
            else:
                logits = F.interpolate(logits, size=logit_size, mode='bilinear')

        return logits
    

    def forward_slide(self, img, img_metas, stride=112, crop_size=224, gt_masks=None):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = F.pad(crop_img, pad)  # zero padding
                crop_seg_logit = self.forward_feature(crop_img, gt_masks=gt_masks).detach()

                torch.cuda.empty_cache()

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                # Gaussian-weighted blending (Idea-04): center-weighted accumulation
                # reduces boundary artifacts compared to uniform averaging
                weight = self.slide_weight_map.to(crop_seg_logit.device)
                weighted_logit = crop_seg_logit * weight
                preds += F.pad(weighted_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += weight
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = F.interpolate(preds, size=img_size, mode='bilinear')

        torch.cuda.empty_cache()

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]

        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop, None)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'], batch_img_metas[0], None)
        

        if self.pamr:
            img_size = batch_img_metas[0]['ori_shape']
            img = F.interpolate(inputs, size=img_size, mode='bilinear', align_corners=False)
            seg_logits = self.pamr(img, seg_logits.to(img.dtype)).to(self.dtype)


        return self.postprocess_result(inputs, seg_logits, data_samples)

    def postprocess_result(self, image, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i]

            seg_logits = seg_logits.softmax(0) # n_queries * w * h
            seg_pred = seg_logits.argmax(0, keepdim=True)

            low_conf = 255
            if self.background == True:
                low_conf = self.bg_idx
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = low_conf   
            
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': seg_pred})
            })

        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b
    

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """





def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices


def _transform1(n_px):
    BICUBIC = InterpolationMode.BICUBIC
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        # _convert_image_to_rgb,
    ])
