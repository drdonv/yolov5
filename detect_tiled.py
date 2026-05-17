#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run tiled YOLOv5 inference on large images.

This script splits each input image into overlapping tiles, runs detection on each
tile, remaps detections to full-image coordinates, then applies a final global NMS.
It is useful when objects become too small in full-image inference.
"""

import argparse
import os
import sys
from glob import glob, has_magic
from pathlib import Path

import numpy as np
import torch
import torchvision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS
from utils.general import (
    LOGGER,
    check_img_size,
    check_requirements,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


def resolve_image_paths(source):
    """Resolve image paths from file/dir/glob/list.txt source."""
    source = str(source)

    if has_magic(source):
        paths = [Path(p) for p in sorted(glob(source, recursive=True))]
    else:
        p = Path(source)
        if p.is_file():
            if p.suffix.lower() == ".txt":
                lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
                paths = [Path(line) for line in lines]
            else:
                paths = [p]
        elif p.is_dir():
            paths = [x for x in sorted(p.rglob("*")) if x.suffix[1:].lower() in IMG_FORMATS]
        else:
            raise FileNotFoundError(f"Source path '{source}' does not exist")

    image_paths = [p for p in paths if p.suffix[1:].lower() in IMG_FORMATS]
    if not image_paths:
        raise FileNotFoundError(f"No image files found for source '{source}'")
    return image_paths


def _tile_starts(length, tile_size, overlap):
    """Compute tile start positions that fully cover one axis."""
    if length <= tile_size:
        return [0]
    step = max(tile_size - overlap, 1)
    starts = list(range(0, length - tile_size + 1, step))
    end_start = length - tile_size
    if starts[-1] != end_start:
        starts.append(end_start)
    return starts


def _global_nms(dets, iou_thres=0.45, agnostic=False, max_det=1000):
    """Apply class-aware global NMS on full-image detections."""
    if dets.numel() == 0:
        return dets

    dets = dets[dets[:, 4].argsort(descending=True)]
    max_nms = 30000
    if dets.shape[0] > max_nms:
        dets = dets[:max_nms]

    max_wh = 7680
    class_offsets = dets[:, 5:6] * (0 if agnostic else max_wh)
    boxes = dets[:, :4] + class_offsets
    scores = dets[:, 4]
    keep = torchvision.ops.nms(boxes, scores, iou_thres)
    keep = keep[:max_det]
    return dets[keep]


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    merge_iou_thres=0.45,
    max_det=1000,
    device="",
    classes=None,
    agnostic_nms=False,
    augment=False,
    tile_size=640,
    tile_overlap=128,
    save_txt=False,
    save_conf=False,
    nosave=False,
    project=ROOT / "runs/detect",
    name="tiled",
    exist_ok=False,
    line_thickness=2,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
):
    """Run tiled inference and save merged detections."""
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    image_paths = resolve_image_paths(source)
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))

    for image_path in image_paths:
        im0 = cv2.imread(str(image_path))
        if im0 is None:
            LOGGER.warning(f"Skipping unreadable image: {image_path}")
            continue

        h, w = im0.shape[:2]
        x_starts = _tile_starts(w, tile_size, tile_overlap)
        y_starts = _tile_starts(h, tile_size, tile_overlap)

        merged = []
        tile_count = 0
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                tile = im0[y0:y1, x0:x1]
                tile_count += 1

                im = letterbox(tile, new_shape=imgsz, auto=False, stride=stride)[0]
                im = im.transpose((2, 0, 1))[::-1]
                im = np.ascontiguousarray(im)

                im_t = torch.from_numpy(im).to(model.device)
                im_t = im_t.half() if model.fp16 else im_t.float()
                im_t /= 255.0
                if im_t.ndim == 3:
                    im_t = im_t.unsqueeze(0)

                pred = model(im_t, augment=augment, visualize=False)
                det = non_max_suppression(
                    pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
                )[0]

                if len(det):
                    det = det.clone()
                    det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], tile.shape).round()
                    det[:, [0, 2]] += x0
                    det[:, [1, 3]] += y0
                    merged.append(det.cpu())

        if merged:
            det_full = torch.cat(merged, dim=0)
            det_full = _global_nms(det_full, iou_thres=merge_iou_thres, agnostic=agnostic_nms, max_det=max_det)
        else:
            det_full = torch.zeros((0, 6))

        # save labels
        if save_txt and len(det_full):
            txt_path = save_dir / "labels" / f"{image_path.stem}.txt"
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            lines = []
            for *xyxy, conf, cls in det_full.tolist():
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (int(cls), *xywh, conf) if save_conf else (int(cls), *xywh)
                lines.append(("%g " * len(line)).rstrip() % line)
            txt_path.write_text("\n".join(lines) + "\n")

        # annotate and save image
        if not nosave:
            annotator = Annotator(im0.copy(), line_width=line_thickness, example=str(names))
            for *xyxy, conf, cls in det_full.tolist():
                c = int(cls)
                label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                annotator.box_label(xyxy, label, color=colors(c, True))
            out_path = save_dir / image_path.name
            cv2.imwrite(str(out_path), annotator.result())

        # log summary
        if len(det_full):
            class_counts = {}
            for c in det_full[:, 5].tolist():
                class_counts[int(c)] = class_counts.get(int(c), 0) + 1
            class_text = ", ".join(f"{v} {names[k]}" for k, v in sorted(class_counts.items()))
            LOGGER.info(f"{image_path.name}: {tile_count} tiles, {len(det_full)} detections ({class_text})")
        else:
            LOGGER.info(f"{image_path.name}: {tile_count} tiles, (no detections)")

    label_count = len(list((save_dir / "labels").glob("*.txt"))) if save_txt else 0
    msg = f"\n{label_count} labels saved to {save_dir / 'labels'}" if save_txt else ""
    LOGGER.info(f"Results saved to {save_dir}{msg}")


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/glob/list.txt")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="tile NMS IoU threshold")
    parser.add_argument("--merge-iou-thres", type=float, default=0.45, help="global merge NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="max detections after global NMS")
    parser.add_argument("--device", default="", help="cuda device or cpu")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class indices")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--tile-size", type=int, default=640, help="tile size in pixels")
    parser.add_argument("--tile-overlap", type=int, default=128, help="tile overlap in pixels")
    parser.add_argument("--save-txt", action="store_true", help="save merged detections to labels/*.txt")
    parser.add_argument("--save-conf", action="store_true", help="include confidence in saved txt labels")
    parser.add_argument("--nosave", action="store_true", help="do not save annotated images")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="tiled", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="allow existing project/name")
    parser.add_argument("--line-thickness", default=2, type=int, help="bounding box thickness")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    """Run entrypoint with dependency checks."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
