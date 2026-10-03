"""计算车辆检测和跟踪指标。"""

import json

import cv2
import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

from preprocess import (
    Box,
    CONFIG,
    DATASETS,
    WEIGHTS,
    image_road_mask,
    on_road,
    test_frame_plan,
)
from infer import infer_detections, infer_tracks_on_images


def _iou(gt, pred):
    matrix = np.zeros((len(gt), len(pred)), dtype=np.float32)
    for i, box in enumerate(gt):
        for j, other in enumerate(pred):
            x1, y1 = max(box.x1, other.x1), max(box.y1, other.y1)
            x2, y2 = min(box.x2, other.x2), min(box.y2, other.y2)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            box_area = max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)
            other_area = max(0, other.x2 - other.x1) * max(0, other.y2 - other.y1)
            matrix[i, j] = area / max(box_area + other_area - area, 1e-9)
    return matrix


def _matches(iou, threshold):
    if not iou.size:
        return []
    bonus = min(iou.shape) + 1
    reward = np.where(iou >= threshold, bonus + iou, 0)
    rows, columns = linear_sum_assignment(reward, maximize=True)
    return [
        (row, column)
        for row, column in zip(rows, columns)
        if iou[row, column] >= threshold
    ]


def _ap(scores, count):
    scores.sort(key=lambda item: item[0], reverse=True)
    tp = np.cumsum([hit for _, hit in scores], dtype=np.float64)
    fp = np.cumsum([not hit for _, hit in scores], dtype=np.float64)
    recall = tp / count
    precision = tp / np.maximum(tp + fp, 1)
    return float(
        np.mean(
            [
                precision[recall >= point].max(initial=0)
                for point in np.linspace(0, 1, 101)
            ]
        )
    )


def detection_metrics(gt, pred):
    iou_threshold = CONFIG["iou_threshold"]
    conf_threshold = CONFIG["inference_confidence"]
    classes = range(len(CONFIG["classes"]))
    aps = []
    tp = fp = fn = 0

    for cls in classes:
        count = sum(box.cls == cls for boxes in gt.values() for box in boxes)
        if not count:
            continue
        ranked = []
        matched = {frame: set() for frame in gt}
        for frame, boxes in pred.items():
            for box in boxes:
                if box.cls == cls:
                    ranked.append((box.score, frame, box))
        ranked.sort(key=lambda item: item[0], reverse=True)
        ap_scores = []
        for score, frame, box in ranked:
            truth = [item for item in gt.get(frame, []) if item.cls == cls]
            overlaps = _iou(truth, [box]).ravel()
            order = np.argsort(overlaps)[::-1]
            hit = next(
                (
                    int(index)
                    for index in order
                    if overlaps[index] >= iou_threshold
                    and index not in matched.get(frame, set())
                ),
                None,
            )
            if hit is not None:
                matched.setdefault(frame, set()).add(hit)
            ap_scores.append((score, hit is not None))
        aps.append(_ap(ap_scores, count))

    for frame, truth in gt.items():
        boxes = [box for box in pred.get(frame, []) if box.score >= conf_threshold]
        for cls in classes:
            gt_ids = [i for i, box in enumerate(truth) if box.cls == cls]
            pred_ids = [i for i, box in enumerate(boxes) if box.cls == cls]
            pairs = _matches(
                _iou([truth[i] for i in gt_ids], [boxes[i] for i in pred_ids]),
                iou_threshold,
            )
            tp += len(pairs)
            fn += len(gt_ids) - len(pairs)
            fp += len(pred_ids) - len(pairs)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "map50": float(np.mean(aps)) if aps else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "confusion_matrix": np.array([[tp, fn], [fp, 0]], dtype=np.int64),
    }


def _read_boxes(path, with_ids=False):
    shapes = json.loads(path.read_text())["shapes"]
    boxes = []
    for shape in shapes:
        if shape["label"] not in CONFIG["classes"]:
            continue
        if with_ids and shape["group_id"] is None:
            raise ValueError(f"跟踪标注缺少跨帧 ID：{path}")
        xs, ys = zip(*shape["points"])
        boxes.append(
            Box(
                CONFIG["classes"].index(shape["label"]),
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                track_id=int(shape["group_id"]) if with_ids else None,
            )
        )
    return boxes


def _tracking_metrics(gt, pred):
    accumulator = mm.MOTAccumulator(auto_id=False)
    for frame_index, frame in enumerate(sorted(gt)):
        truth = gt[frame]
        boxes = pred.get(frame, [])
        overlap = _iou(truth, boxes)
        distance = np.where(overlap >= CONFIG["iou_threshold"], 1 - overlap, np.nan)
        for i, box in enumerate(truth):
            for j, other in enumerate(boxes):
                if box.cls != other.cls:
                    distance[i, j] = np.nan
        accumulator.update(
            [box.track_id for box in truth],
            [box.track_id for box in boxes],
            distance,
            frameid=frame_index,
        )
    result = (
        mm.metrics.create()
        .compute(
            accumulator,
            metrics=["mota", "num_switches", "idf1"],
            name="track",
        )
        .iloc[0]
    )
    return {
        "mota": float(result.mota),
        "idsw": int(result.num_switches),
        "idf1": float(result.idf1),
    }


def _test_images():
    images = {}
    for name, (video, root) in DATASETS.items():
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise FileNotFoundError(f"无法打开视频：{video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        frames, _ = test_frame_plan(name, fps, frame_count)
        directory = root / "vihicle_detection"
        paths = [directory / f"f{frame:06d}.jpg" for frame in sorted(frames)]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise ValueError(f"{name} 均匀抽样帧缺失：{missing[0]}")
        images[name] = paths
    return images


def _tracking_samples(name):
    video, root = DATASETS[name]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    truth = {}
    clips = {}
    _, tracking = test_frame_plan(name, fps, frame_count)
    for clip_index, frames in tracking.items():
        folder = root / "vihicle_tracking"
        paths = [folder / f"f{frame:06d}.jpg" for frame in frames]
        if not paths:
            raise ValueError(f"{name} 第 {clip_index + 1} 段没有跟踪帧")
        for frame_index, path in enumerate(paths):
            annotation = path.with_suffix(".json")
            if not annotation.exists():
                raise ValueError(f"跟踪帧缺少标注：{annotation}")
            frame_gt = _read_boxes(annotation, with_ids=True)
            truth[clip_index * 1_000_000 + frame_index] = [
                Box(
                    box.cls,
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                    track_id=clip_index * 1_000_000 + box.track_id,
                )
                for box in frame_gt
            ]
        clips[clip_index] = paths
    return truth, clips, fps


def evaluate():
    results = {}
    test_images = _test_images()

    for name, images in test_images.items():
        road = image_road_mask(name, cv2.imread(str(images[0])))
        gt = {
            index: [
                box
                for box in _read_boxes(image.with_suffix(".json"))
                if on_road(box, road)
            ]
            for index, image in enumerate(images)
        }
        pred, detector_fps = infer_detections(
            dict(enumerate(images)), det_weights=WEIGHTS["vehicle"], road=road
        )
        detection = detection_metrics(gt, pred)

        track_gt, clips, _ = _tracking_samples(name)
        track_road = image_road_mask(
            name, cv2.imread(str(next(iter(clips.values()))[0]))
        )
        track_gt = {
            frame: [box for box in boxes if on_road(box, track_road)]
            for frame, boxes in track_gt.items()
        }
        track_pred = {}
        elapsed = 0
        track_frames = 0
        for clip_index, paths in clips.items():
            predictions, fps = infer_tracks_on_images(
                paths, det_weights=WEIGHTS["vehicle"], road=track_road
            )
            elapsed += len(paths) / fps
            track_frames += len(paths)
            for frame_index, path in enumerate(paths):
                key = clip_index * 1_000_000 + frame_index
                track_pred[key] = [
                    Box(
                        box.cls,
                        box.x1,
                        box.y1,
                        box.x2,
                        box.y2,
                        box.score,
                        clip_index * 1_000_000 + box.track_id,
                    )
                    for box in predictions[int(path.stem[1:])]
                ]
        tracking = _tracking_metrics(track_gt, track_pred)
        result = {
            **detection,
            **tracking,
            "detection_test_images": len(images),
            "detection_gt_boxes": sum(map(len, gt.values())),
            "detection_pred_boxes": sum(
                box.score >= CONFIG["inference_confidence"]
                for boxes in pred.values()
                for box in boxes
            ),
            "detector_fps": detector_fps,
            "tracking_test_frames": track_frames,
            "tracking_fps": track_frames / elapsed,
        }
        results[name] = result

    metric_keys = (
        "map50",
        "precision",
        "recall",
        "f1",
        "detector_fps",
        "mota",
        "idsw",
        "idf1",
        "tracking_fps",
        "detection_test_images",
        "detection_gt_boxes",
        "detection_pred_boxes",
        "tracking_test_frames",
    )
    results["平均"] = {
        key: float(np.mean([result[key] for result in results.values()]))
        for key in metric_keys
    }
    results["平均"]["idsw_total"] = sum(results[name]["idsw"] for name in DATASETS)
    return results
