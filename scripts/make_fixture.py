#!/usr/bin/env python
"""Generate a tiny on-disk KITTI/BDD100K-shaped fixture for CI smoke tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

KITTI_NAMES = {1: "Car", 2: "Pedestrian", 3: "Cyclist", 4: "Truck"}
BDD_NAMES = {1: "car", 2: "person", 3: "rider", 4: "truck"}


def _scene(path: Path, width: int, height: int, seed: int, boxes):
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), (45, 45, 50), np.uint8)
    cv2.rectangle(image, (0, int(height * 0.55)), (width, height), (70, 70, 75), -1)
    for x1, y1, x2, y2, cid in boxes:
        color = {1: (180, 60, 60), 2: (60, 180, 60), 3: (60, 60, 200), 4: (200, 180, 60)}[cid]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (250, 250, 250), 1)
    image = np.clip(image.astype(np.int16) + rng.integers(-6, 6, image.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def make(root: Path, count: int) -> None:
    for split, offset in (("train", 0), ("val", 1000), ("test", 2000)):
        kimg = root / "kitti" / "images" / split
        klbl = root / "kitti" / "labels" / split
        bimg = root / "bdd100k" / "images" / split
        blbl = root / "bdd100k" / "labels" / split
        for d in (kimg, klbl, bimg, blbl, root / "kitti" / "splits", root / "bdd100k" / "splits"):
            d.mkdir(parents=True, exist_ok=True)
        knames, bnames = [], []
        for i in range(count):
            kname = f"{offset + i:06d}"
            boxes = [(30 + i % 3 * 10, 70, 100 + i % 3 * 10, 130, 1 + (i % 4)),
                     (150 + i % 2 * 8, 90, 195 + i % 2 * 8, 150, 2)]
            _scene(kimg / f"{kname}.png", 320, 180, offset + i, boxes)
            (klbl / f"{kname}.txt").write_text("\n".join(
                f"{KITTI_NAMES[c]} 0 0 0 {x1} {y1} {x2} {y2} 1.5 1.6 3.0 0 0 0 0"
                for x1, y1, x2, y2, c in boxes) + "\n")
            knames.append(kname)

            bname = f"bdd-{split}-{offset + i:05d}"
            boxes = [(60 + i % 4 * 6, 80, 130 + i % 4 * 6, 150, 1),
                     (200 + i % 2 * 8, 100, 245 + i % 2 * 8, 160, 2)]
            _scene(bimg / f"{bname}.jpg", 480, 270, offset + i, boxes)
            (blbl / f"{bname}.json").write_text(json.dumps({
                "image_id": bname, "width": 480, "height": 270,
                "weather": "clear", "timeofday": "daytime", "scene": "city street",
                "categories": [BDD_NAMES[c] for *_, c in boxes],
                "boxes": [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in boxes],
            }))
            bnames.append(bname)
        (root / "kitti" / "splits" / f"{split}.txt").write_text("\n".join(knames) + "\n")
        (root / "bdd100k" / "splits" / f"{split}.txt").write_text("\n".join(bnames) + "\n")
    print(f"fixture written to {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="datasets_fixture")
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    make(Path(args.out), args.count)
