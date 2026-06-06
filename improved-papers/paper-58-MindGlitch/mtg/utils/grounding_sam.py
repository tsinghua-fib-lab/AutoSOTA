# You must install https://github.com/IDEA-Research/Grounded-Segment-Anything first
# sys.path.append('/datawaha/cggroup/eldesoa/code/Grounded-Segment-Anything')

import torch
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
import numpy as np

from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import annotate, predict
from groundingdino.util import box_ops


# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
sam_checkpoint = "sam_vit_h_4b8939.pth"


BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25

transform = T.Compose(
    [
        # T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    if isinstance(mask_image, torch.Tensor):
        mask_image = mask_image.cpu().numpy()
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def show_mask_with_pt(img, mask, pt, show_image=False):
    annotated_frame_with_mask = show_mask(mask, img)
    annotated_frame_with_mask = Image.fromarray(annotated_frame_with_mask)
    draw = ImageDraw.Draw(annotated_frame_with_mask)
    radius = 4
    draw.ellipse(
        (pt[0] - radius, pt[1] - radius, pt[0] + radius, pt[1] + radius),
        fill=(0, 120, 0),
        outline=(255, 255, 255),
    )

    # if show_image:
    #     plt.imshow(annotated_frame_with_mask)
    #     plt.show()

    return annotated_frame_with_mask


def segment_object_bb(img: Image.Image, groundingdino_model, sam_predictor, desc: str, DEVICE, visualize: bool = False):
    image, _ = transform(img, None)
    image_source = np.asarray(img)
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=desc,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=DEVICE,
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    # plt.imshow(annotated_frame)

    ## SAM
    sam_predictor.set_image(image_source)
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    annotated_frame_with_mask = show_mask(masks[0][0].cpu(), annotated_frame)
    # if visualize:
    #     plt.imshow(annotated_frame_with_mask)
    #     plt.show()

    mask_img = Image.fromarray(masks[0, 0].cpu().numpy())
    return masks[:1], mask_img, Image.fromarray(annotated_frame_with_mask)


def segment_object_pt(img, pt, sam_predictor):
    sam_predictor.set_image(img)

    pt_labels = np.array([1])
    masks, _, _ = sam_predictor.predict(
        point_coords=pt,
        point_labels=pt_labels,
        multimask_output=True,
    )
    return masks
