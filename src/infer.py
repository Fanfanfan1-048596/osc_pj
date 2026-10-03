"""运行车辆检测、车道分割和车辆跟踪。"""

import csv
import json
import math
import os
import shutil
import time
from collections import deque
from functools import cache

import cv2
import numpy as np
import torch
from matplotlib.colors import to_rgb
from ultralytics import YOLO

from preprocess import (
    Box,
    CONFIG,
    OUTPUTS,
    ROOT,
    WEIGHTS,
    image_road_mask,
    on_road,
    road_input,
)
from train import UNet


@cache
def _colors():
    return tuple(
        tuple(
            round(channel * 255)
            for channel in to_rgb(CONFIG["visualization"][key])[::-1]
        )
        for key in ("lane_color", "vehicle_color", "trail_color")
    )


def _vehicle_classes(model):
    names = set(CONFIG["vehicle_names"])
    return [cls for cls, name in model.names.items() if name.lower() in names]


def _boxes(result):
    ids = (
        result.boxes.id.cpu().tolist()
        if result.boxes.id is not None
        else [None] * len(result.boxes)
    )
    return [
        Box(0, *coords, float(score), int(track_id) if track_id is not None else None)
        for coords, score, track_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            ids,
        )
    ]


def load_lane_model(path):
    device = f"cuda:{CONFIG['device']}"
    model = UNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def predict_lane(model, image):
    height, width = image.shape[:2]
    settings = CONFIG["lane_model"]
    size = settings["tile_size"]
    stride = settings["stride"]
    ys = list(range(0, max(height - size, 0) + 1, stride))
    xs = list(range(0, max(width - size, 0) + 1, stride))
    if ys[-1] != height - size:
        ys.append(height - size)
    if xs[-1] != width - size:
        xs.append(width - size)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tiles = [(y, x) for y in ys for x in xs]
    score = np.zeros((height, width), np.float32)
    count = np.zeros((height, width), np.float32)
    device = next(model.parameters()).device
    mean = torch.tensor(CONFIG["training"]["image_mean"], device=device)[
        None, :, None, None
    ]
    std = torch.tensor(CONFIG["training"]["image_std"], device=device)[
        None, :, None, None
    ]

    with torch.inference_mode():
        for start in range(0, len(tiles), settings["batch_size"]):
            positions = tiles[start : start + settings["batch_size"]]
            batch = np.stack([rgb[y : y + size, x : x + size] for y, x in positions])
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
            batch = (batch / 255 - mean) / std
            with torch.autocast("cuda", dtype=torch.float16):
                probs = model(batch).sigmoid().float().cpu().numpy()
            for (y, x), prob in zip(positions, probs):
                score[y : y + size, x : x + size] += prob
                count[y : y + size, x : x + size] += 1

    return (score / count >= settings["threshold"]).astype(np.uint8)


def draw_lane(mask):
    line = mask.copy()
    center = np.zeros_like(line)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(line):
        opened = cv2.morphologyEx(line, cv2.MORPH_OPEN, kernel)
        center |= line - opened
        line = cv2.erode(line, kernel)
    width = CONFIG["visualization"]["lane_width"]
    return cv2.dilate(center, np.ones((width, width), np.uint8))


def detect_image(model, image, confidence=None):
    result = model.predict(
        image,
        imgsz=int(model.model.args["imgsz"]),
        conf=confidence if confidence is not None else CONFIG["inference_confidence"],
        classes=_vehicle_classes(model),
        device=CONFIG["device"],
        verbose=False,
    )[0]
    return _boxes(result)


class Trail:
    def __init__(self, fps):
        self.length = max(2, round(fps * CONFIG["visualization"]["trail_seconds"]))
        self.points = {}

    def draw(self, image, boxes):
        mask = np.zeros(image.shape[:2], np.uint8)
        active = {}
        radius = CONFIG["visualization"]["trail_head_width"] / 2
        opacity = CONFIG["visualization"]["trail_alpha"]

        for box in boxes:
            if box.track_id is None:
                continue
            path = self.points.get(box.track_id, deque(maxlen=self.length))
            path.append((round((box.x1 + box.x2) / 2), round((box.y1 + box.y2) / 2)))
            active[box.track_id] = path
            if len(path) < 2 or math.dist(path[0], path[-1]) < 6:
                continue

            steps = len(path) - 1
            for index, (start, end) in enumerate(zip(path, list(path)[1:]), 1):
                dx, dy = end[0] - start[0], end[1] - start[1]
                distance = math.hypot(dx, dy)
                if distance < 1:
                    continue
                nx, ny = -dy / distance, dx / distance
                left = radius * (index - 1) / steps
                right = radius * index / steps
                polygon = np.rint(
                    [
                        (start[0] + nx * left, start[1] + ny * left),
                        (start[0] - nx * left, start[1] - ny * left),
                        (end[0] - nx * right, end[1] - ny * right),
                        (end[0] + nx * right, end[1] + ny * right),
                    ]
                ).astype(np.int32)
                cv2.fillConvexPoly(
                    mask, polygon, round(255 * opacity * index / steps), cv2.LINE_AA
                )
            cv2.circle(
                mask, path[-1], round(radius), round(255 * opacity), -1, cv2.LINE_AA
            )

        self.points = active
        visible = mask > 0
        blend = mask[visible, None].astype(np.float32) / 255
        image[visible] = (
            image[visible] * (1 - blend) + np.array(_colors()[2]) * blend
        ).astype(np.uint8)


def _draw(image, lane, boxes, trail=None):
    colors = _colors()
    canvas = image.copy()
    lane_layer = canvas.copy()
    lane_layer[lane == 1] = colors[0]
    canvas = cv2.addWeighted(
        canvas,
        1 - CONFIG["visualization"]["lane_alpha"],
        lane_layer,
        CONFIG["visualization"]["lane_alpha"],
        0,
    )

    if trail is not None:
        trail.draw(canvas, boxes)

    for box in boxes:
        x1, y1, x2, y2 = map(round, (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            colors[1],
            CONFIG["visualization"]["vehicle_width"],
        )
    return canvas


def export_figures():
    folder = OUTPUTS / "report_figures"
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in CONFIG["report_frames"].items():
        capture = cv2.VideoCapture(str(OUTPUTS / f"{name}_visualization.mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        capture.release()
        if not ok:
            raise ValueError(f"无法读取 {name} 第 {frame} 帧")
        cv2.imwrite(str(folder / f"{name}.png"), image)
    shutil.copyfile(
        OUTPUTS / "XAM-S_road_region.png", folder / "XAM-S_road_region.png"
    )


def infer_video(video, name, det_weights=None, lane_weights=None):
    video = ROOT / video
    detector = YOLO(str(det_weights or WEIGHTS["vehicle"]))
    lane_model = load_lane_model(lane_weights or WEIGHTS["lane"])
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = int(detector.model.args["imgsz"])
    classes = _vehicle_classes(detector)
    ok, first = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"视频没有可读帧：{video}")
    road = image_road_mask(name, first) if name in CONFIG["road_regions"] else None
    lane_area = None
    if road is not None:
        margin = CONFIG["road_regions"][name]["lane_margin"]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
        )
        lane_area = cv2.erode(road, kernel)
    x, y, crop_width, crop_height = (
        cv2.boundingRect(road) if road is not None else (0, 0, width, height)
    )
    crop_road = (
        road[y : y + crop_height, x : x + crop_width] if road is not None else None
    )
    first_roi = first[y : y + crop_height, x : x + crop_width]
    if crop_road is not None:
        first_roi = road_input(first_roi, crop_road)
    detector.predict(
        first_roi,
        imgsz=image_size,
        classes=classes,
        device=CONFIG["device"],
        verbose=False,
    )
    lane = predict_lane(lane_model, first)
    if lane_area is not None:
        lane &= lane_area
    lane = draw_lane(lane)
    if lane_area is not None:
        lane &= lane_area

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    output = OUTPUTS / f"{name}_visualization.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建视频：{output}")

    trail = Trail(source_fps)
    count = 0
    frame_index = 0
    start = time.perf_counter()
    try:
        with (OUTPUTS / f"{name}_predictions.csv").open("w", newline="") as file:
            rows = csv.writer(file)
            rows.writerow(
                ["frame", "time_s", "id", "class", "x1", "y1", "x2", "y2", "score"]
            )
            while True:
                if frame_index:
                    ok, image = capture.read()
                    if not ok:
                        break
                else:
                    image = first

                roi = image[y : y + crop_height, x : x + crop_width]
                if crop_road is not None:
                    roi = road_input(roi, crop_road)
                result = detector.track(
                    roi,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=image_size,
                    conf=CONFIG["tracking_confidence"],
                    classes=classes,
                    device=CONFIG["device"],
                    verbose=False,
                )[0]
                boxes = _boxes(result)
                if road is not None:
                    for box in boxes:
                        box.x1 += x
                        box.x2 += x
                        box.y1 += y
                        box.y2 += y
                    boxes = [box for box in boxes if on_road(box, road)]
                if (
                    frame_index
                    and frame_index % CONFIG["lane_model"]["video_step"] == 0
                ):
                    lane = predict_lane(lane_model, image)
                    if lane_area is not None:
                        lane &= lane_area
                    lane = draw_lane(lane)
                    if lane_area is not None:
                        lane &= lane_area
                writer.write(_draw(image, lane, boxes, trail))
                for box in boxes:
                    rows.writerow(
                        [
                            frame_index,
                            f"{frame_index / source_fps:.3f}",
                            box.track_id,
                            box.cls,
                            box.x1,
                            box.y1,
                            box.x2,
                            box.y2,
                            box.score,
                        ]
                    )
                count += 1
                if count % round(source_fps * 10) == 0:
                    print(f"{name}：已处理 {count} 帧", flush=True)
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start
    timing = {
        "video": os.path.relpath(video, ROOT),
        "processed_frames": count,
        "source_frames": frame_index,
        "source_fps": source_fps,
        "output_fps": source_fps,
        "detector_image_size": image_size,
        "seconds": elapsed,
        "pipeline_fps": count / elapsed,
    }
    (OUTPUTS / f".{name}_timing.json").write_text(json.dumps(timing, indent=2))
    return timing


def infer_detections(images, det_weights=None, road=None):
    detector = YOLO(str(det_weights or WEIGHTS["vehicle"]))
    predictions = {}
    frames = list(images)
    first = cv2.imread(str(images[frames[0]]))
    detect_image(
        detector,
        road_input(first, road) if road is not None else first,
        CONFIG["ap_confidence"],
    )
    torch.cuda.synchronize(CONFIG["device"])
    start = time.perf_counter()
    for frame in frames:
        image = cv2.imread(str(images[frame]))
        boxes = detect_image(
            detector,
            road_input(image, road) if road is not None else image,
            CONFIG["ap_confidence"],
        )
        predictions[frame] = (
            [box for box in boxes if on_road(box, road)] if road is not None else boxes
        )
    torch.cuda.synchronize(CONFIG["device"])
    return predictions, len(predictions) / (time.perf_counter() - start)


def infer_tracks_on_images(images, det_weights=None, road=None):
    detector = YOLO(str(det_weights or WEIGHTS["vehicle"]))
    classes = _vehicle_classes(detector)
    image_size = int(detector.model.args["imgsz"])
    first = cv2.imread(str(images[0]))
    detector.predict(
        road_input(first, road) if road is not None else first,
        imgsz=image_size,
        classes=classes,
        device=CONFIG["device"],
        verbose=False,
    )
    torch.cuda.synchronize(CONFIG["device"])
    start = time.perf_counter()
    predictions = {}
    for path in images:
        image = cv2.imread(str(path))
        result = detector.track(
            road_input(image, road) if road is not None else image,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=image_size,
            conf=CONFIG["tracking_confidence"],
            classes=classes,
            device=CONFIG["device"],
            verbose=False,
        )[0]
        frame = int(path.stem[1:])
        predictions[frame] = [
            box
            for box in _boxes(result)
            if box.track_id is not None and (road is None or on_road(box, road))
        ]
    torch.cuda.synchronize(CONFIG["device"])
    elapsed = time.perf_counter() - start
    return predictions, len(images) / elapsed
