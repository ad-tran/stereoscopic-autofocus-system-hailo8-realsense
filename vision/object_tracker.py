import sys
import numpy as np
import degirum as dg

from utils.config import (
    DG_MODEL_PERSON_NAME,
    DG_MODEL_FACE_NAME,
    DG_MODEL_SEG_NAME,
    DG_ZOO_PERSON_URL,
    DG_ZOO_FACE_URL,
    DG_ZOO_SEG_URL,
    DG_DEVICE_TYPE,
    DG_TOKEN,
    DG_INFERENCE_HOST,
)

#  SORT direkt von festem Pfad laden
sys.path.append('/home/amacus/hailo_examples/sort')
from sort import Sort

class DetectionPipeline:
    def __init__(self):
        #  Modelle hart laden
        self.model_person = dg.load_model(
            model_name=DG_MODEL_PERSON_NAME,
            inference_host_address=DG_INFERENCE_HOST,
            zoo_url=DG_ZOO_PERSON_URL,
            token=DG_TOKEN,
            device_type=DG_DEVICE_TYPE,
        )
        self.model_face = dg.load_model(
            model_name=DG_MODEL_FACE_NAME,
            inference_host_address=DG_INFERENCE_HOST,
            zoo_url=DG_ZOO_FACE_URL,
            token=DG_TOKEN,
            device_type=DG_DEVICE_TYPE,
        )
        self.model_seg = dg.load_model(
            model_name=DG_MODEL_SEG_NAME,
            inference_host_address=DG_INFERENCE_HOST,
            zoo_url=DG_ZOO_SEG_URL,
            token=DG_TOKEN,
            device_type=DG_DEVICE_TYPE,
        )
        #  Warmup (Face + Seg)
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _ = self.model_face(dummy)
        _ = self.model_seg(dummy)

    def detect_person_bboxes(self, image_bgr) -> list[tuple[int, int, int, int]]:
        #  nur x1,y1,x2,y2 (ohne Score)
        results = self.model_person(image_bgr)
        boxes: list[tuple[int, int, int, int]] = []
        for r in results.results:
            if r.get('label') == 'person':
                x1, y1, x2, y2 = r['bbox']
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    def detect_faces(self, image_bgr) -> list[tuple[int, int, int, int, float]]:
        results = self.model_face(image_bgr)
        faces: list[tuple[int, int, int, int, float]] = []
        for r in results.results:
            if r.get('label') == 'face' and float(r.get('score', 0)) > 0.3:
                x1, y1, x2, y2 = r['bbox']
                score = float(r.get('score', 0.0))
                faces.append((int(x1), int(y1), int(x2, int(y2), score)))
        return faces

    def segment_person(self, image_bgr):
        results = self.model_seg(image_bgr)
        for r in results.results:
            if r.get('label') == 'person':
                mask = r.get('mask') or r.get('segmentation_mask')
                return mask
        return None


class SortTracker:
    def __init__(self):
        self._sort = Sort()

    def update(self, dets: np.ndarray) -> np.ndarray:
        #  Dets unverändert an SORT weitergeben (kein Score anhängen)
        if dets is None:
            dets = np.empty((0, 5))
        return self._sort.update(dets)