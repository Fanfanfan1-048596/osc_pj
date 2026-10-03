"""准备测试帧、车辆预标注和主干道区域。"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from matplotlib.colors import to_rgb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATASETS = {
    "XAM-S": (DATA / "XAM-S.mp4", DATA / "train" / "xam-s"),
    "Anthem": (DATA / "Anthem.mp4", DATA / "train" / "Anthem"),
    "Local": (DATA / "local.mp4", DATA / "train" / "local"),
}
CROP_X, CROP_Y, WIDTH, HEIGHT = CONFIG["source"]["crop"]
WEIGHTS = {name: ROOT / path for name, path in CONFIG["paths"].items()}


class Box:
    def __init__(self, cls, x1, y1, x2, y2, score=1.0, track_id=None):
        self.cls = cls
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.track_id = track_id


def test_frame_plan(name, fps, frame_count):
    step = max(1, round(fps * CONFIG["evaluation"]["detection_interval_seconds"]))
    detection = set(range(0, frame_count, step))
    stride = max(1, round(fps / 10))
    tracking = {
        index: [
            frame
            for frame in range(0, frame_count, stride)
            if start <= frame / fps < end
        ]
        for index, (start, end) in enumerate(
            CONFIG["evaluation"]["tracking_segments"][name]
        )
    }
    return detection, tracking


def crop_frame(name, image):
    if name == "XAM-S":
        return image[CROP_Y : CROP_Y + HEIGHT, CROP_X : CROP_X + WIDTH]
    return image


def road_path(name):
    return DATA / "road_masks" / f"{name}.png"


def build_road_mask(name):
    settings = CONFIG["road_regions"][name]
    capture = cv2.VideoCapture(str(ROOT / settings["video"]))
    frames = []
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for index in np.linspace(0, count - 1, 9).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = capture.read()
        if ok:
            frames.append(cv2.resize(image, None, fx=0.5, fy=0.5))
    capture.release()
    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    height, width = background.shape[:2]

    # 引导线只给出大致范围，边界由静态背景上的路面颜色和路缘确定。
    points = np.rint(np.array(settings["guide"]) * 0.5).astype(np.int32)
    guide = np.zeros((height, width), np.uint8)
    cv2.fillPoly(guide, [points], 255)
    band = settings["edge_band"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band, band))
    labels = np.full((height, width), cv2.GC_PR_FGD, np.uint8)
    labels[cv2.dilate(guide, kernel) == 0] = cv2.GC_BGD
    labels[cv2.erode(guide, kernel) > 0] = cv2.GC_FGD
    cv2.grabCut(
        background,
        labels,
        None,
        np.zeros((1, 65), np.float64),
        np.zeros((1, 65), np.float64),
        4,
        cv2.GC_INIT_WITH_MASK,
    )
    mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask.fill(0)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    mask = cv2.resize(mask, (width * 2, height * 2), interpolation=cv2.INTER_NEAREST)

    path = road_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask * 255)
    preview = cv2.resize(frames[len(frames) // 2], (width * 2, height * 2))
    preview[mask == 0] = (preview[mask == 0] * 0.45).astype(np.uint8)
    edge = cv2.Canny(mask * 255, 100, 200)
    preview[edge > 0] = tuple(
        round(value * 255)
        for value in to_rgb(CONFIG["visualization"]["road_edge_color"])[::-1]
    )
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUTS / f"{name}_road_region.png"), preview)
    return mask


def load_road_mask(name):
    path = road_path(name)
    if not path.exists():
        return build_road_mask(name)
    return (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)


def image_road_mask(name, image):
    mask = load_road_mask(name)
    if mask.shape != image.shape[:2]:
        height, width = image.shape[:2]
        mask = mask[CROP_Y : CROP_Y + height, CROP_X : CROP_X + width]
    return mask


def road_input(image, mask):
    result = image.copy()
    result[mask == 0] = 0
    return result


def on_road(box, mask):
    x = round((box.x1 + box.x2) / 2)
    y = round((box.y1 + box.y2) / 2)
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])


def _shapes(result, road, tracking=False):
    ids = (
        result.boxes.id.cpu().tolist()
        if tracking and result.boxes.id is not None
        else [None] * len(result.boxes)
    )
    shapes = []
    for coords, score, track_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        ids,
    ):
        if tracking and track_id is None:
            continue
        x1, y1, x2, y2 = coords
        if not on_road(Box(0, x1, y1, x2, y2), road):
            continue
        shapes.append(
            {
                "label": "vehicle",
                "score": score,
                "points": [[x1, y1], [x2, y2]],
                "group_id": int(track_id) if track_id is not None else None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )
    return shapes


def _save(folder, frame, image, shapes):
    path = folder / f"f{frame:06d}.jpg"
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": CONFIG["annotation_version"],
                "flags": {},
                "shapes": shapes,
                "imagePath": path.name,
                "imageData": None,
                "imageHeight": image.shape[0],
                "imageWidth": image.shape[1],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def prelabel_dataset(dataset="all"):
    from ultralytics import YOLO

    names = (
        ("XAM-S", "Anthem", "Local")
        if dataset == "all"
        else ({"xam-s": "XAM-S", "anthem": "Anthem", "local": "Local"}[dataset],)
    )
    total_frames = total_boxes = 0
    for name in names:
        video, root = DATASETS[name]
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        detection, clips = test_frame_plan(name, fps, frame_count)
        tracking = {frame: clip for clip, frames in clips.items() for frame in frames}
        for folder in (root / "vihicle_detection", root / "vihicle_tracking"):
            folder.mkdir(parents=True, exist_ok=True)

        detector = YOLO(str(WEIGHTS["vehicle"]))
        vehicle_names = set(CONFIG["vehicle_names"])
        classes = [
            cls
            for cls, label in detector.names.items()
            if label.lower() in vehicle_names
        ]
        tracker = None
        current_clip = None
        road = None
        written = 0
        started = time.perf_counter()
        try:
            for frame in range(frame_count):
                ok, source = capture.read()
                if not ok:
                    break
                if frame not in detection and frame not in tracking:
                    continue

                image = crop_frame(name, source)
                if road is None:
                    road = image_road_mask(name, image)
                masked = road_input(image, road)

                if frame in detection:
                    result = detector.predict(
                        masked,
                        imgsz=CONFIG["training"]["detection_image_size"],
                        conf=CONFIG["inference_confidence"],
                        classes=classes,
                        device=CONFIG["device"],
                        verbose=False,
                    )[0]
                    shapes = _shapes(result, road)
                    _save(root / "vihicle_detection", frame, image, shapes)
                    total_boxes += len(shapes)
                    written += 1

                if frame in tracking:
                    clip = tracking[frame]
                    if clip != current_clip:
                        tracker = YOLO(str(WEIGHTS["vehicle"]))
                        current_clip = clip
                    result = tracker.track(
                        masked,
                        persist=True,
                        tracker="bytetrack.yaml",
                        imgsz=CONFIG["training"]["detection_image_size"],
                        conf=CONFIG["tracking_confidence"],
                        classes=classes,
                        device=CONFIG["device"],
                        verbose=False,
                    )[0]
                    shapes = _shapes(result, road, tracking=True)
                    _save(root / "vihicle_tracking", frame, image, shapes)
                    total_boxes += len(shapes)
                    written += 1

                if written % 100 == 0:
                    print(
                        f"{name}：{written} 帧，{written / (time.perf_counter() - started):.1f} FPS",
                        flush=True,
                    )
        finally:
            capture.release()
        total_frames += written
        print(f"{name}：完成 {written} 帧", flush=True)
    return total_frames, total_boxes
