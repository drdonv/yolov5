#!/usr/bin/env python3
"""
Prepare a leak-safe hybrid YOLO dataset from full images + tiled crops.

This script is designed for datasets laid out like:
    dataset/final/rebar/
      images/
      labels/

It will:
1) Split by original image (if splits do not already exist),
2) Tile selected splits with overlap,
3) Remap/clamp labels into each tile,
4) Keep a controlled amount of empty tiles (hard negatives),
5) Optionally keep a sample of full images in output (hybrid training),
6) Write a train-ready dataset YAML.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    yaml = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional fallback
    tqdm = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    rel_image_path: Path
    key: str
    explicit_split: Optional[str]


@dataclass(frozen=True)
class Box:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        w = max(0.0, self.x2 - self.x1)
        h = max(0.0, self.y2 - self.y1)
        return w * h


@dataclass
class RunStats:
    full_images_saved: int = 0
    tile_images_saved: int = 0
    positive_tiles_saved: int = 0
    empty_tiles_saved: int = 0
    labels_skipped_bad_lines: int = 0
    missing_label_files: int = 0
    source_images: int = 0
    class_ids_seen: set = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.class_ids_seen is None:
            self.class_ids_seen = set()


def normalize_split_name(name: str) -> Optional[str]:
    value = name.lower()
    if value == "train":
        return "train"
    if value in {"val", "valid", "validation"}:
        return "val"
    if value == "test":
        return "test"
    return None


def parse_csv_floats(text: str) -> List[float]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"No numeric values found in '{text}'")
    return values


def parse_csv_ints(text: str) -> List[int]:
    values = [int(v) for v in parse_csv_floats(text)]
    if any(v <= 0 for v in values):
        raise ValueError("Tile sizes must be positive integers.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create leak-safe hybrid (full + tiled) YOLO dataset."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("dataset/final/rebar"),
        help="Source dataset root containing images/ and labels/.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("dataset/final/rebar_hybrid_tiled"),
        help="Output dataset root.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Train ratio for random split if source has no explicit splits.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Val ratio for random split if source has no explicit splits.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.00,
        help="Test ratio for random split if source has no explicit splits.",
    )
    parser.add_argument(
        "--tile-sizes",
        type=str,
        default="640,960",
        help="Comma-separated tile sizes in pixels.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.30,
        help="Tile overlap fraction in [0, 0.95).",
    )
    parser.add_argument(
        "--tile-splits",
        type=str,
        default="train",
        help="Comma-separated splits to tile (e.g. train,val,test).",
    )
    parser.add_argument(
        "--min-box-coverage",
        type=float,
        default=0.40,
        help="Keep clipped box if intersection/original area is at least this.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=6.0,
        help="Drop clipped boxes smaller than this many pixels in tile space.",
    )
    parser.add_argument(
        "--empty-ratio",
        type=float,
        default=0.20,
        help="Max empty tiles kept per positive tile (per image and tile size).",
    )
    parser.add_argument(
        "--max-empty-no-positive",
        type=int,
        default=2,
        help="Max empty tiles to keep when an image yields zero positive tiles.",
    )
    parser.add_argument(
        "--include-full-train",
        type=float,
        default=0.30,
        help="Probability of copying full source images into train split.",
    )
    parser.add_argument(
        "--include-full-val",
        type=float,
        default=1.00,
        help="Probability of copying full source images into val split.",
    )
    parser.add_argument(
        "--include-full-test",
        type=float,
        default=1.00,
        help="Probability of copying full source images into test split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete destination directory first if it exists.",
    )
    return parser


def iter_images(images_root: Path) -> Iterable[Path]:
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def discover_samples(src: Path) -> List[Sample]:
    images_root = src / "images"
    labels_root = src / "labels"
    if not images_root.exists():
        raise FileNotFoundError(f"Missing images directory: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_root}")

    samples: List[Sample] = []
    for image_path in iter_images(images_root):
        rel = image_path.relative_to(images_root)
        label_path = labels_root / rel.with_suffix(".txt")
        explicit_split: Optional[str] = None
        if rel.parts:
            explicit_split = normalize_split_name(rel.parts[0])
        key = "__".join(rel.with_suffix("").parts)
        samples.append(
            Sample(
                image_path=image_path,
                label_path=label_path,
                rel_image_path=rel,
                key=key,
                explicit_split=explicit_split,
            )
        )

    if not samples:
        raise RuntimeError(f"No images found under: {images_root}")

    samples.sort(key=lambda s: str(s.rel_image_path))
    return samples


def split_samples(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train-ratio + val-ratio + test-ratio must equal 1.0.")

    explicit = [s.explicit_split for s in samples]
    all_explicit = all(x in SPLIT_ORDER for x in explicit)
    none_explicit = all(x is None for x in explicit)

    split_map: Dict[str, List[Sample]] = {k: [] for k in SPLIT_ORDER}

    if all_explicit:
        for sample in samples:
            split_map[sample.explicit_split or "train"].append(sample)
        return split_map

    if not none_explicit:
        print(
            "Warning: Mixed explicit and implicit split paths detected. "
            "Falling back to random split for all images."
        )

    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    split_map["train"] = shuffled[:n_train]
    split_map["val"] = shuffled[n_train : n_train + n_val]
    split_map["test"] = shuffled[n_train + n_val : n_train + n_val + n_test]
    return split_map


def parse_yolo_boxes(label_path: Path, img_w: int, img_h: int, stats: RunStats) -> List[Box]:
    boxes: List[Box] = []
    if not label_path.exists():
        stats.missing_label_files += 1
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                stats.labels_skipped_bad_lines += 1
                continue
            try:
                cls = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                stats.labels_skipped_bad_lines += 1
                continue

            if bw <= 0 or bh <= 0:
                continue
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)

            x1 = (cx - bw / 2.0) * img_w
            y1 = (cy - bh / 2.0) * img_h
            x2 = (cx + bw / 2.0) * img_w
            y2 = (cy + bh / 2.0) * img_h

            x1 = max(0.0, min(float(img_w), x1))
            y1 = max(0.0, min(float(img_h), y1))
            x2 = max(0.0, min(float(img_w), x2))
            y2 = max(0.0, min(float(img_h), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(Box(cls_id=cls, x1=x1, y1=y1, x2=x2, y2=y2))
            stats.class_ids_seen.add(cls)
    return boxes


def list_window_starts(total: int, tile: int, overlap: float) -> List[int]:
    tile = min(tile, total)
    if total <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def tile_windows(img_w: int, img_h: int, tile_size: int, overlap: float) -> Iterable[Tuple[int, int, int, int]]:
    tw = min(tile_size, img_w)
    th = min(tile_size, img_h)
    xs = list_window_starts(img_w, tw, overlap)
    ys = list_window_starts(img_h, th, overlap)
    for y in ys:
        for x in xs:
            yield (x, y, x + tw, y + th)


def remap_box_to_tile(
    box: Box,
    window: Tuple[int, int, int, int],
    min_coverage: float,
    min_box_size: float,
) -> Optional[Tuple[int, float, float, float, float]]:
    x0, y0, x1, y1 = window
    ix1 = max(box.x1, float(x0))
    iy1 = max(box.y1, float(y0))
    ix2 = min(box.x2, float(x1))
    iy2 = min(box.y2, float(y1))
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return None

    inter_area = iw * ih
    orig_area = box.area
    if orig_area <= 0:
        return None
    coverage = inter_area / orig_area
    if coverage < min_coverage:
        return None
    if iw < min_box_size or ih < min_box_size:
        return None

    tw = float(x1 - x0)
    th = float(y1 - y0)
    tx1 = ix1 - x0
    ty1 = iy1 - y0
    tx2 = ix2 - x0
    ty2 = iy2 - y0

    cx = ((tx1 + tx2) / 2.0) / tw
    cy = ((ty1 + ty2) / 2.0) / th
    bw = (tx2 - tx1) / tw
    bh = (ty2 - ty1) / th

    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    if bw <= 0 or bh <= 0:
        return None
    return (box.cls_id, cx, cy, bw, bh)


def save_label_file(path: Path, labels: Sequence[Tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cls, cx, cy, bw, bh in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def save_full_image_and_label(
    sample: Sample,
    split: str,
    dst: Path,
    include_prob: float,
    stats: RunStats,
    rng: random.Random,
) -> None:
    if include_prob <= 0:
        return
    if include_prob < 1.0 and rng.random() > include_prob:
        return

    img_out = dst / "images" / split / f"{sample.key}__full{sample.image_path.suffix.lower()}"
    lbl_out = dst / "labels" / split / f"{sample.key}__full.txt"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    lbl_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image_path, img_out)
    if sample.label_path.exists():
        shutil.copy2(sample.label_path, lbl_out)
    else:
        save_label_file(lbl_out, [])
    stats.full_images_saved += 1


def save_tiled_samples(
    sample: Sample,
    split: str,
    dst: Path,
    tile_sizes: Sequence[int],
    overlap: float,
    min_box_coverage: float,
    min_box_size: float,
    empty_ratio: float,
    max_empty_no_positive: int,
    stats: RunStats,
    rng: random.Random,
) -> None:
    image = Image.open(sample.image_path).convert("RGB")
    img_w, img_h = image.size
    boxes = parse_yolo_boxes(sample.label_path, img_w, img_h, stats)

    for tile_size in tile_sizes:
        positives: List[Tuple[Tuple[int, int, int, int], List[Tuple[int, float, float, float, float]]]] = []
        negatives: List[Tuple[int, int, int, int]] = []

        for window in tile_windows(img_w, img_h, tile_size, overlap):
            remapped: List[Tuple[int, float, float, float, float]] = []
            for box in boxes:
                label = remap_box_to_tile(
                    box=box,
                    window=window,
                    min_coverage=min_box_coverage,
                    min_box_size=min_box_size,
                )
                if label is not None:
                    remapped.append(label)
            if remapped:
                positives.append((window, remapped))
            else:
                negatives.append(window)

        for window, labels in positives:
            x0, y0, x1, y1 = window
            stem = f"{sample.key}__s{tile_size}__x{x0}_y{y0}"
            img_out = dst / "images" / split / f"{stem}{sample.image_path.suffix.lower()}"
            lbl_out = dst / "labels" / split / f"{stem}.txt"
            img_out.parent.mkdir(parents=True, exist_ok=True)
            lbl_out.parent.mkdir(parents=True, exist_ok=True)
            tile_img = image.crop((x0, y0, x1, y1))
            tile_img.save(img_out)
            save_label_file(lbl_out, labels)
            stats.tile_images_saved += 1
            stats.positive_tiles_saved += 1

        if positives:
            max_empty = int(math.ceil(len(positives) * empty_ratio))
        else:
            max_empty = max_empty_no_positive
        max_empty = min(max_empty, len(negatives))
        if max_empty > 0:
            selected_negatives = rng.sample(negatives, max_empty)
            for window in selected_negatives:
                x0, y0, x1, y1 = window
                stem = f"{sample.key}__s{tile_size}__x{x0}_y{y0}"
                img_out = dst / "images" / split / f"{stem}{sample.image_path.suffix.lower()}"
                lbl_out = dst / "labels" / split / f"{stem}.txt"
                img_out.parent.mkdir(parents=True, exist_ok=True)
                lbl_out.parent.mkdir(parents=True, exist_ok=True)
                tile_img = image.crop((x0, y0, x1, y1))
                tile_img.save(img_out)
                save_label_file(lbl_out, [])
                stats.tile_images_saved += 1
                stats.empty_tiles_saved += 1


def maybe_load_names_from_yaml(src: Path) -> Optional[List[str]]:
    if yaml is None:
        return None

    candidates = []
    for pattern in ("*.yaml", "*.yml"):
        candidates.extend(src.glob(pattern))
        candidates.extend(src.parent.glob(pattern))
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate.resolve())
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)

    for file_path in unique_candidates:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "names" not in data:
            continue
        names = data["names"]
        if isinstance(names, dict):
            try:
                max_idx = max(int(k) for k in names.keys())
            except Exception:
                continue
            ordered = [""] * (max_idx + 1)
            ok = True
            for k, v in names.items():
                try:
                    idx = int(k)
                except Exception:
                    ok = False
                    break
                if idx < 0 or idx >= len(ordered):
                    ok = False
                    break
                ordered[idx] = str(v)
            if ok and all(x != "" for x in ordered):
                return ordered
        elif isinstance(names, list) and names:
            return [str(x) for x in names]
    return None


def names_from_class_ids(class_ids: Sequence[int]) -> List[str]:
    if not class_ids:
        return ["class_0"]
    max_id = max(class_ids)
    names = [f"class_{i}" for i in range(max_id + 1)]
    return names


def write_dataset_yaml(dst: Path, names: List[str]) -> Path:
    data = {
        "path": str(dst.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    yaml_path = dst / "rebar_hybrid_tiled.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(data, f, sort_keys=False)
        else:
            f.write(f"path: {data['path']}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n")
            f.write(f"nc: {data['nc']}\n")
            f.write("names:\n")
            for i, name in enumerate(names):
                f.write(f"  {i}: {name}\n")
    return yaml_path


def dump_report(dst: Path, split_map: Dict[str, List[Sample]], stats: RunStats, args: argparse.Namespace) -> Path:
    report = {
        "source_images": stats.source_images,
        "split_counts": {k: len(v) for k, v in split_map.items()},
        "full_images_saved": stats.full_images_saved,
        "tile_images_saved": stats.tile_images_saved,
        "positive_tiles_saved": stats.positive_tiles_saved,
        "empty_tiles_saved": stats.empty_tiles_saved,
        "missing_label_files": stats.missing_label_files,
        "labels_skipped_bad_lines": stats.labels_skipped_bad_lines,
        "class_ids_seen": sorted(stats.class_ids_seen),
        "config": {
            "src": str(args.src),
            "dst": str(args.dst),
            "seed": args.seed,
            "tile_sizes": parse_csv_ints(args.tile_sizes),
            "tile_overlap": args.tile_overlap,
            "tile_splits": [normalize_split_name(x.strip()) for x in args.tile_splits.split(",") if x.strip()],
            "min_box_coverage": args.min_box_coverage,
            "min_box_size": args.min_box_size,
            "empty_ratio": args.empty_ratio,
            "max_empty_no_positive": args.max_empty_no_positive,
            "include_full_train": args.include_full_train,
            "include_full_val": args.include_full_val,
            "include_full_test": args.include_full_test,
        },
    }
    report_path = dst / "prepare_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.tile_overlap < 0.0 or args.tile_overlap >= 0.95:
        raise ValueError("--tile-overlap must be in [0.0, 0.95).")
    if not (0.0 <= args.min_box_coverage <= 1.0):
        raise ValueError("--min-box-coverage must be in [0.0, 1.0].")
    if args.min_box_size < 0.0:
        raise ValueError("--min-box-size must be non-negative.")
    if args.empty_ratio < 0.0:
        raise ValueError("--empty-ratio must be non-negative.")
    if args.max_empty_no_positive < 0:
        raise ValueError("--max-empty-no-positive must be non-negative.")

    for p in (args.include_full_train, args.include_full_val, args.include_full_test):
        if p < 0.0 or p > 1.0:
            raise ValueError("include-full probabilities must be in [0.0, 1.0].")

    tile_sizes = parse_csv_ints(args.tile_sizes)
    tile_splits_raw = [normalize_split_name(x.strip()) for x in args.tile_splits.split(",") if x.strip()]
    tile_splits = {x for x in tile_splits_raw if x is not None}
    if not tile_splits:
        raise ValueError("--tile-splits did not contain valid split names.")

    args.src = args.src.resolve()
    args.dst = args.dst.resolve()
    if args.dst.exists():
        if args.overwrite:
            shutil.rmtree(args.dst)
        else:
            raise FileExistsError(f"Destination exists: {args.dst}. Use --overwrite.")

    rng = random.Random(args.seed)
    stats = RunStats()

    samples = discover_samples(args.src)
    stats.source_images = len(samples)
    split_map = split_samples(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    include_probs = {
        "train": args.include_full_train,
        "val": args.include_full_val,
        "test": args.include_full_test,
    }

    for split in SPLIT_ORDER:
        (args.dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split in SPLIT_ORDER:
        samples_in_split = split_map.get(split, [])
        if not samples_in_split:
            continue
        iterator = samples_in_split
        if tqdm is not None:
            iterator = tqdm(samples_in_split, desc=f"Processing {split}", unit="img")
        for sample in iterator:
            save_full_image_and_label(
                sample=sample,
                split=split,
                dst=args.dst,
                include_prob=include_probs[split],
                stats=stats,
                rng=rng,
            )
            if split in tile_splits:
                save_tiled_samples(
                    sample=sample,
                    split=split,
                    dst=args.dst,
                    tile_sizes=tile_sizes,
                    overlap=args.tile_overlap,
                    min_box_coverage=args.min_box_coverage,
                    min_box_size=args.min_box_size,
                    empty_ratio=args.empty_ratio,
                    max_empty_no_positive=args.max_empty_no_positive,
                    stats=stats,
                    rng=rng,
                )

    names = maybe_load_names_from_yaml(args.src)
    if names is None:
        names = names_from_class_ids(sorted(stats.class_ids_seen))
    yaml_path = write_dataset_yaml(args.dst, names)
    report_path = dump_report(args.dst, split_map, stats, args)

    print("Done.")
    print(f"Output dataset: {args.dst}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Report: {report_path}")
    print(
        "Train command example:\n"
        f"  python train.py --data {yaml_path} --weights yolov5s.pt --img 960 --batch 16"
    )


if __name__ == "__main__":
    main()
