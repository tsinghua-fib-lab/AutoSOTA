from argparse import ArgumentParser

import cv2
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
from mmseg.utils import get_classes, get_palette
import numpy as np
import torch
import tqdm
import rein

classes = get_classes("cityscapes")
palette = get_palette("cityscapes")


def draw_sem_seg(sem_seg: torch.Tensor):
    num_classes = len(classes)
    sem_seg = sem_seg.data.squeeze(0)
    H, W = sem_seg.shape
    ids = torch.unique(sem_seg).cpu().numpy()
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)
    colors = [palette[label] for label in labels]
    colors = [torch.tensor(color, dtype=torch.uint8).view(1, 1, 3) for color in colors]
    result = torch.zeros([H, W, 3], dtype=torch.uint8)
    for label, color in zip(labels, colors):
        result[sem_seg == label, :] = color
    return result.cpu().numpy()


def main():
    parser = ArgumentParser()
    parser.add_argument("video", help="Video file or webcam id")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--palette",
        default="cityscapes",
        help="Color palette used for segmentation map",
    )
    parser.add_argument(
        "--show", action="store_true", help="Whether to show draw result"
    )
    parser.add_argument(
        "--show-wait-time", default=1, type=int, help="Wait time after imshow"
    )
    parser.add_argument(
        "--output-file", default=None, type=str, help="Output video file path"
    )
    parser.add_argument(
        "--output-fourcc", default="MJPG", type=str, help="Fourcc of the output video"
    )
    parser.add_argument(
        "--output-fps", default=-1, type=int, help="FPS of the output video"
    )
    parser.add_argument(
        "--output-height", default=-1, type=int, help="Frame height of the output video"
    )
    parser.add_argument(
        "--output-width", default=-1, type=int, help="Frame width of the output video"
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.3,
        help="Opacity of painted segmentation map. In (0, 1] range.",
    )
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == "cpu":
        model = revert_sync_batchnorm(model)

    # build input video
    if args.video.isdigit():
        args.video = int(args.video)
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()
    input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # init output video
    writer_fusion = None
    output_height = None
    output_width = None
    fusion_file = args.video.replace(".mp4", "_fusion.mp4")
    segmap_file = args.video.replace(".mp4", "_segmap.mp4")
    fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
    output_fps = args.output_fps if args.output_fps > 0 else input_fps
    output_height = args.output_height if args.output_height > 0 else int(input_height)
    output_width = args.output_width if args.output_width > 0 else int(input_width)
    writer_fusion = cv2.VideoWriter(
        fusion_file, fourcc, output_fps, (output_width, output_height), True
    )
    writer_segmap = cv2.VideoWriter(
        segmap_file, fourcc, output_fps, (output_width, output_height), True
    )
    print(writer_fusion)
    # start looping
    bar = tqdm.tqdm(total=input_length)
    try:
        while True:
            flag, frame = cap.read()
            if not flag:
                break
            bar.update(1)
            # test a single image
            result = inference_model(model, frame)

            # blend raw image and prediction
            pred = draw_sem_seg(result.pred_sem_seg)
            draw_img = (
                pred[:, :, ::-1] * (1 - args.opacity) + frame * args.opacity
            ).astype(np.uint8)

            if args.show:
                cv2.imshow("video_demo", draw_img)
                cv2.waitKey(args.show_wait_time)
            if writer_fusion and writer_segmap:
                if (
                    draw_img.shape[0] != output_height
                    or draw_img.shape[1] != output_width
                ):
                    draw_img = cv2.resize(draw_img, (output_width, output_height))
                writer_fusion.write(draw_img)
                writer_segmap.write(pred[:, :, ::-1].astype(np.uint8))
    finally:
        if writer_fusion and writer_segmap:
            writer_fusion.release()
            writer_segmap.release()
        cap.release()


if __name__ == "__main__":
    main()
