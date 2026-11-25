"""Tests for dataset loading, batching and augmentation."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cv_obstacle.data import (BDD100KDataset, CameraStyle, DomainAdaptationDataset,
                              KittiDataset, augment_image, make_loader)


def test_kitti_dataset_returns_expected_tuple(toy_dataset):
    ds = KittiDataset(toy_dataset / "kitti", split="train")
    assert len(ds) == 8
    image, boxes, class_ids, obstacles = ds[0]
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8
    assert boxes.shape[1] == 4 and boxes.dtype == np.float32
    assert class_ids.dtype == np.int64
    assert len(obstacles) == len(boxes)
    assert ((class_ids >= 1) & (class_ids <= 4)).all()
    assert all(b[2] > b[0] and b[3] > b[1] for b in boxes)


def test_bdd_dataset_and_ontology(toy_dataset):
    ds = BDD100KDataset(toy_dataset / "bdd100k", split="val")
    assert len(ds) == 4
    image, boxes, class_ids, _ = ds[0]
    assert image.dtype == np.uint8
    assert len(boxes) == len(class_ids) > 0
    assert set(np.unique(class_ids)).issubset({1, 2, 3, 4})


def test_batch_shape_and_types(toy_dataset):
    ds = KittiDataset(toy_dataset / "kitti", split="train")
    loader = make_loader(ds, batch_size=4, shuffle=False)
    images, boxes, labels = next(iter(loader))
    assert len(images) == 4
    assert torch.stack(images).shape == (4, 3, images[0].shape[1], images[0].shape[2])
    assert isinstance(boxes, list) and isinstance(labels, list)
    assert all(b.shape[1] == 4 for b in boxes)


def test_domain_adaptation_dataset_pairs(toy_dataset):
    source = KittiDataset(toy_dataset / "kitti", split="train")
    target = BDD100KDataset(toy_dataset / "bdd100k", split="train", camera="truck")
    paired = DomainAdaptationDataset(source, target)
    assert len(paired) == len(source)
    item = paired[0]
    assert set(item) == {"source_image", "target_image", "source_boxes",
                         "source_labels", "target_index"}
    assert item["source_image"].shape[2] == 3 and item["target_image"].shape[2] == 3
    # target indices wrap deterministically
    assert paired[len(source) - 1]["target_index"] == (len(source) - 1) % len(target)


def test_truck_camera_style_changes_pixels_and_boxes(toy_dataset):
    ds = KittiDataset(toy_dataset / "kitti", split="train")
    image, boxes, _, _ = ds[0]
    style = CameraStyle()
    styled, styled_boxes = style.apply(image, boxes)
    assert styled.shape == image.shape
    assert not np.array_equal(styled, image)
    assert styled_boxes.shape == boxes.shape
    assert (styled_boxes[:, 2] <= image.shape[1] + 1).all()


def test_augmentation_handles_edge_cases(toy_dataset):
    ds = KittiDataset(toy_dataset / "kitti", split="train")
    image, boxes, ids, _ = ds[0]

    # empty annotation list
    out_img, out_boxes, out_ids = augment_image(image, np.zeros((0, 4)), None,
                                                class_ids=np.zeros((0,), np.int64))
    assert out_img.shape == image.shape and out_boxes.shape[0] == out_ids.shape[0]

    # single-pixel box
    one = np.array([[10.0, 10.0, 11.0, 11.0]], np.float32)
    out_img, out_boxes, out_ids = augment_image(image, one, None,
                                                class_ids=np.array([1], np.int64))
    assert out_img.dtype == np.uint8
    assert out_boxes.shape[1] == 4

    # blank (all-black) image is still a valid image
    blank = np.zeros((64, 64, 3), np.uint8)
    out_img, _, _ = augment_image(blank, np.array([[2.0, 2.0, 20.0, 20.0]], np.float32),
                                  None, class_ids=np.array([2]))
    assert out_img.shape == blank.shape

    # normal path preserves labels / coordinates
    out_img, out_boxes, out_ids = augment_image(image, boxes, None, class_ids=ids, seed=7)
    assert out_img.dtype == np.uint8 and out_boxes.shape[0] == out_ids.shape[0]
    assert ((out_ids >= 1) & (out_ids <= 4)).all()


def test_empty_image_raises():
    with pytest.raises(ValueError):
        augment_image(np.zeros((0, 0, 3), np.uint8), np.zeros((0, 4)), None)


@pytest.mark.parametrize("split", ["train", "val"])
def test_all_class_ids_within_range(toy_dataset, split):
    for dataset in (KittiDataset(toy_dataset / "kitti", split=split),
                    BDD100KDataset(toy_dataset / "bdd100k", split=split)):
        for i in range(len(dataset)):
            _, _, ids, obstacles = dataset[i]
            assert ids.min(initial=1) >= 1
            assert ids.max(initial=1) <= 4
            assert all(1 <= o.class_id <= 4 for o in obstacles)
