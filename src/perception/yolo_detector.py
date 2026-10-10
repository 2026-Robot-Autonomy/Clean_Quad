"""
CNN detector: YOLOv8-nano fine-tuned on the simulator dataset.

Training starts from COCO-pretrained weights, so only the
litter classes have to be learned.
"""

from pathlib import Path
import shutil

from src.config import PROJECT_ROOT
from src.perception.detection import Detection, Detector


MODEL_PATH = PROJECT_ROOT / "models" / "litter_yolov8n.pt"
RUNS_DIR = PROJECT_ROOT / "models" / "runs"

BASE_WEIGHTS = "yolov8n.pt"
IMAGE_SIZE = 640


def train_yolo(data_yaml, epochs=50, batch=16, device=0, workers=4):
    """Fine-tune and copy the best weights to MODEL_PATH."""

    from ultralytics import YOLO

    model = YOLO(BASE_WEIGHTS)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMAGE_SIZE,
        batch=batch,
        device=device,
        workers=workers,
        project=str(RUNS_DIR),
        name="litter",
        exist_ok=True,
        seed=0,
        # Simulator colours carry class information (and are
        # varied in the data), so keep colour augmentation mild.
        hsv_h=0.005,
        fliplr=0.5,
        mosaic=1.0,
        plots=True,
    )

    best = RUNS_DIR / "litter" / "weights" / "best.pt"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, MODEL_PATH)

    return MODEL_PATH


class YoloDetector(Detector):

    name = "YOLOv8n (CNN)"

    def __init__(self, weights=MODEL_PATH, conf=0.001, device=None, batch=16):
        """
        conf: minimum score kept. Low for mAP evaluation; the
        operating threshold is applied by the caller.
        """

        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(
                f"{weights} not found: run experiments/23_train_detectors.py first"
            )

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device
        self.batch = batch

    def detect_batch(self, images):

        detections = []

        for start in range(0, len(images), self.batch):

            # Ultralytics treats numpy images as BGR (OpenCV order).
            chunk = [rgb[..., ::-1] for rgb in images[start:start + self.batch]]

            results = self.model.predict(
                chunk, imgsz=IMAGE_SIZE, conf=self.conf, device=self.device, verbose=False
            )

            for r in results:
                boxes = r.boxes
                detections.append([
                    Detection(int(c), float(s), tuple(float(v) for v in xyxy))
                    for xyxy, c, s in zip(
                        boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()
                    )
                ])

        return detections

    def detect(self, rgb):
        return self.detect_batch([rgb])[0]
