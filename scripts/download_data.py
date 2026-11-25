#!/usr/bin/env python
"""Download a real, reduced KITTI + BDD100K obstacle-detection dataset.

Provenance
----------
KITTI   : official 2D object labels (data_object_label_2.zip) from the KITTI
          AWS bucket, paired with the matching frames served by the public
          Hugging Face mirror ``dpdl-benchmark/kitti``.
BDD100K : real BDD100K frames and detection annotations served by the public
          Hugging Face mirror ``lance-format/BDD100K-enriched`` (original
          release: https://bdd-data.berkeley.edu/, research licence).

Both mirrors expose the original, unmodified images.  No synthetic pixels are
used for training: the only synthetic step is the *camera style transfer*
applied at load time to simulate a truck-mounted sensor (see
``configs/truck_camera.yaml``).

Exact commands
--------------
    python scripts/download_data.py --kitti 900 --bdd 900

The script is resumable, writes SHA-256 checksums for every file to
``datasets/checksums.sha256`` and a provenance manifest to
``datasets/MANIFEST.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cv_obstacle.data import BDD_NAME_TO_ID, KITTI_NAME_TO_ID, KITTI_IGNORE  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kw):
        return x

KITTI_LABELS_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
KITTI_LABELS_SHA256 = "4efc76220d867e1c31bb980bbf8cbc02599f02a9cb4350effa98dbb04aaed880"
KITTI_ROWS_API = (
    "https://datasets-server.huggingface.co/rows?dataset=dpdl-benchmark/kitti"
    "&config=default&split=train&offset={offset}&length=100"
)
BDD_ROWS_API = (
    "https://datasets-server.huggingface.co/rows?dataset=lance-format/BDD100K-enriched"
    "&config=default&split=train&offset={offset}&length=100"
)
USER_AGENT = "cv-obstacle-downloader/1.0 (research dataset reproducibility)"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from last


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_name(value) -> str:
    """The HF rows API serialises bytes fields as the literal string ``b'name'``."""
    if isinstance(value, bytes):
        value = value.decode()
    value = str(value)
    if len(value) >= 3 and value[0] == "b" and value[1] in "'\"" and value[-1] == value[1]:
        value = value[2:-1]
    return value


def _cached_page(work: Path, tag: str, url: str) -> bytes:
    cache = work / "pages"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    path = cache / f"{tag}_{key}.json"
    if path.exists():
        return path.read_bytes()
    payload = _get(url)
    path.write_bytes(payload)
    return payload


def _split_for(name: str, seed: int = 42) -> str:
    value = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "val"
    return "test"


def _write_split_files(root: Path, split_map: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        names = sorted([n for n, s in split_map.items() if s == split])
        counts[split] = len(names)
        (root / "splits").mkdir(parents=True, exist_ok=True)
        (root / "splits" / f"{split}.txt").write_text("\n".join(names) + ("\n" if names else ""))
    return counts


# ---------------------------------------------------------------------------
# KITTI
# ---------------------------------------------------------------------------
def _extract_kitti_labels(work: Path) -> Dict[str, Path]:
    archive = work / "kitti_labels.zip"
    if not archive.exists() or _sha256(archive) != KITTI_LABELS_SHA256:
        print(f"[kitti] downloading official labels: {KITTI_LABELS_URL}")
        archive.write_bytes(_get(KITTI_LABELS_URL))
        digest = _sha256(archive)
        if digest != KITTI_LABELS_SHA256:
            raise RuntimeError(f"KITTI label checksum mismatch: {digest}")
    out_dir = work / "kitti_labels"
    if not (out_dir / "training" / "label_2").exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    return {p.stem: p for p in (out_dir / "training" / "label_2").glob("*.txt")}


def _parse_kitti(path: Path) -> List[Tuple[str, List[float]]]:
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 15 or parts[0] in KITTI_IGNORE:
            continue
        if parts[0] not in KITTI_NAME_TO_ID:
            continue
        out.append((parts[0], [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]))
    return out


def download_kitti(dest: Path, work: Path, target: int, seed: int = 42) -> Dict[str, Any]:
    labels = _extract_kitti_labels(work)
    print(f"[kitti] {len(labels)} official label files available")
    rng = random.Random(seed)
    offsets = list(range(0, 7000, 100))
    rng.shuffle(offsets)
    candidates: List[Tuple[str, List[Tuple[str, List[float]]], str]] = []
    seen: set[str] = set()
    pages_used = 0
    for offset in tqdm(offsets, desc="kitti index"):
        try:
            page = json.loads(_cached_page(work, "kitti", KITTI_ROWS_API.format(offset=offset)))
            pages_used += 1
        except RuntimeError:
            continue
        for item in page.get("rows", []):
            row = item["row"]
            name = _clean_name(row["image/file_name"])
            stem = Path(name).stem
            if stem in seen or stem not in labels:
                continue
            ann = _parse_kitti(labels[stem])
            if ann:
                candidates.append((stem, ann, row["image"]["src"]))
                seen.add(stem)
        if len(candidates) >= target * 2:
            break

    # deterministic, class-balanced-enough sample
    rng.shuffle(candidates)
    chosen = candidates[:target]
    out_root = dest

    def _fetch_one(item: Tuple[str, List[Tuple[str, List[float]]], str]) -> Tuple[str, int]:
        stem, ann, url = item
        split = _split_for(stem, seed)
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / f"{stem}.png"
        if not img_path.exists():
            img_path.write_bytes(_get(url))
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(
                f"{name} 0.0 0 0 {' '.join(f'{v:.2f}' for v in box)} 0 0 0 0 0 0 0"
                for name, box in ann
            ) + "\n"
        )
        return stem, img_path.stat().st_size

    total = 0
    split_map: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_fetch_one, item) for item in chosen]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="kitti images"):
            stem, size = fut.result()
            total += size
            split_map[stem] = _split_for(stem, seed)

    counts = _write_split_files(out_root, split_map)
    return {
        "source_labels": {"url": KITTI_LABELS_URL, "sha256": KITTI_LABELS_SHA256},
        "source_images": "https://huggingface.co/datasets/dpdl-benchmark/kitti (rows API)",
        "num_images": len(split_map),
        "bytes": total,
        "splits": counts,
        "pages_used": pages_used,
    }


# ---------------------------------------------------------------------------
# BDD100K
# ---------------------------------------------------------------------------
def download_bdd(dest: Path, work: Path, target: int, seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed + 1)
    offsets = list(range(0, 80000, 100))
    rng.shuffle(offsets)
    rows: List[Dict[str, Any]] = []
    for offset in tqdm(offsets, desc="bdd index"):
        try:
            page = json.loads(_cached_page(work, "bdd", BDD_ROWS_API.format(offset=offset)))
        except RuntimeError:
            continue
        for item in page.get("rows", []):
            row = item["row"]
            if row.get("is_duplicate"):
                continue
            mapped = [(c, b) for c, b in zip(row.get("ann_categories", []), row.get("ann_bboxes", []))
                      if c in BDD_NAME_TO_ID]
            if not mapped:
                continue
            rows.append({
                "image_id": row["image_id"],
                "url": row["image_bytes"]["src"],
                "height": row["height"],
                "width": row["width"],
                "weather": row.get("weather"),
                "timeofday": row.get("timeofday"),
                "scene": row.get("scene"),
                "ann": mapped,
            })
        if len(rows) >= target * 2:
            break

    # Prefer a balanced spread over weather / time of day.
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(f"{row['weather']}|{row['timeofday']}", []).append(row)
    selected: List[Dict[str, Any]] = []
    keys = sorted(by_condition)
    while len(selected) < target and any(by_condition.values()):
        for key in keys:
            bucket = by_condition[key]
            if bucket and len(selected) < target:
                selected.append(bucket.pop())
    selected = selected[:target]

    out_root = dest
    split_map: Dict[str, str] = {}
    total = 0

    def _fetch_one(row: Dict[str, Any]) -> Tuple[str, int, str]:
        name = row["image_id"]
        split = _split_for(name, seed)
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / f"{name}.jpg"
        if not img_path.exists():
            img_path.write_bytes(_get(row["url"]))
        (lbl_dir / f"{name}.json").write_text(json.dumps({
            "image_id": name,
            "width": row["width"],
            "height": row["height"],
            "weather": row["weather"],
            "timeofday": row["timeofday"],
            "scene": row["scene"],
            "categories": [c for c, _ in row["ann"]],
            "boxes": [[float(v) for v in b] for _, b in row["ann"]],
        }, indent=1))
        return name, img_path.stat().st_size, split

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_fetch_one, row) for row in selected]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="bdd images"):
            name, size, split = fut.result()
            total += size
            split_map[name] = split

    counts = _write_split_files(out_root, split_map)
    conditions: Dict[str, int] = {}
    for row in selected:
        key = f"{row['weather']}|{row['timeofday']}"
        conditions[key] = conditions.get(key, 0) + 1
    return {
        "source_images": "https://huggingface.co/datasets/lance-format/BDD100K-enriched (rows API)",
        "original_release": "https://bdd-data.berkeley.edu/",
        "num_images": len(selected),
        "bytes": total,
        "splits": counts,
        "conditions": dict(sorted(conditions.items())),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _write_checksums(datasets_root: Path) -> Tuple[int, int]:
    lines: List[str] = []
    total = 0
    for path in sorted(list(datasets_root.rglob("*.png")) + list(datasets_root.rglob("*.jpg")) +
                       list(datasets_root.rglob("*.json")) + list(datasets_root.rglob("*.txt"))):
        if path.name == "checksums.sha256" or path.name == "MANIFEST.json":
            continue
        total += path.stat().st_size
        lines.append(f"{_sha256(path)}  {path.relative_to(ROOT)}")
    (datasets_root / "checksums.sha256").write_text("\n".join(lines) + "\n")
    return len(lines), total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="datasets", help="output directory")
    parser.add_argument("--kitti", type=int, default=900, help="KITTI frames to keep")
    parser.add_argument("--bdd", type=int, default=900, help="BDD100K frames to keep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    root = (ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    kitti_dir, bdd_dir = root / "kitti", root / "bdd100k"
    work = root / ".work"
    work.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "commands": {
            "download": f"python scripts/download_data.py --kitti {args.kitti} --bdd {args.bdd}",
            "note": "checkpoints and SHA-256 for every file: datasets/checksums.sha256",
        },
        "ontology": {
            "1": "car", "2": "pedestrian", "3": "cyclist", "4": "truck",
            "kitti_mapping": {k: v for k, v in KITTI_NAME_TO_ID.items()},
            "bdd_mapping": {k: v for k, v in BDD_NAME_TO_ID.items()},
        },
    }

    if args.force or not (kitti_dir / "splits" / "train.txt").exists():
        manifest["kitti"] = download_kitti(kitti_dir, work, args.kitti, args.seed)
    else:
        print("[kitti] cached")
        manifest["kitti"] = json.loads((root / ".kitti_meta.json").read_text()) \
            if (root / ".kitti_meta.json").exists() else {"cached": True}
    (root / ".kitti_meta.json").write_text(json.dumps(manifest.get("kitti", {}), indent=1))

    if args.force or not (bdd_dir / "splits" / "train.txt").exists():
        manifest["bdd100k"] = download_bdd(bdd_dir, work, args.bdd, args.seed)
    else:
        print("[bdd100k] cached")
        manifest["bdd100k"] = json.loads((root / ".bdd_meta.json").read_text()) \
            if (root / ".bdd_meta.json").exists() else {"cached": True}
    (root / ".bdd_meta.json").write_text(json.dumps(manifest.get("bdd100k", {}), indent=1))

    n_files, total_bytes = _write_checksums(root)
    manifest["checksums"] = {"file": "datasets/checksums.sha256", "num_files": n_files}
    manifest["total_bytes"] = total_bytes
    manifest["total_mb"] = round(total_bytes / 1e6, 1)
    manifest["elapsed_s"] = round(time.time() - started, 1)
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("total_mb", "total_bytes", "elapsed_s")}, indent=2))
    print(f"wrote {root / 'MANIFEST.json'} and {root / 'checksums.sha256'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
