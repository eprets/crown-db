import uuid
import json
import math
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from ..database import SessionLocal
from ..models import Observation, Annotation, Image
from ..utils.image_io import read_image_unicode


def crop_roi(img, x0: float, y0: float, a: float, b: float, padding_px: int):
    """Вырезает ROI по bounding box эллипса + padding"""
    h, w = img.shape[:2]
    xmin = int(max(0, math.floor(x0 - a - padding_px)))
    xmax = int(min(w, math.ceil(x0 + a + padding_px)))
    ymin = int(max(0, math.floor(y0 - b - padding_px)))
    ymax = int(min(h, math.ceil(y0 + b + padding_px)))
    roi = img[ymin:ymax, xmin:xmax].copy()
    bbox = (xmin, ymin, xmax, ymax)
    return roi, bbox


def compute_features(roi, a: float, b: float):
    """Вычисляет простые признаки ROI"""
    area_ellipse = float(math.pi * a * b)
    axis_ratio = float(a / b) if b != 0 else None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_gray = float(np.mean(gray))
    std_gray = float(np.std(gray))
    return {
        "ellipse_area": area_ellipse,
        "axis_ratio": axis_ratio,
        "roi_mean_gray": mean_gray,
        "roi_std_gray": std_gray,
    }


def build_observations(padding_px: int = 20) -> int:
    """
    Создаёт наблюдения (ROI) для всех аннотаций, у которых ещё нет наблюдения.
    Возвращает количество созданных наблюдений.
    """
    db = SessionLocal()
    added = 0

    # Находим аннотации без наблюдений (используем подзапрос через db.query)
    annotations = db.query(Annotation).filter(
        ~db.query(Observation).filter(Observation.annotation_id == Annotation.annotation_id).exists()
    ).all()

    for ann in annotations:
        img_record = db.query(Image).filter(Image.image_id == ann.image_id).first()
        if not img_record:
            logging.warning(f"Изображение не найдено для аннотации {ann.annotation_id}")
            continue

        try:
            img_cv = read_image_unicode(img_record.path)
        except Exception as e:
            logging.warning(f"Не удалось прочитать {img_record.path}: {e}")
            continue

        if img_cv is None:
            continue

        roi, bbox = crop_roi(
            img_cv,
            x0=ann.x0,
            y0=ann.y0,
            a=ann.a,
            b=ann.b,
            padding_px=padding_px,
        )

        obs_id = str(uuid.uuid4())
        roi_raw_dir = Path("data/roi_raw")
        roi_raw_dir.mkdir(parents=True, exist_ok=True)
        roi_path = roi_raw_dir / f"{obs_id}.png"

        cv2.imwrite(str(roi_path), roi)

        features = compute_features(roi, ann.a, ann.b)
        features["bbox"] = bbox

        observation = Observation(
            obs_id=obs_id,
            annotation_id=ann.annotation_id,
            image_id=ann.image_id,
            tree_id=ann.tree_id,
            roi_raw_path=str(roi_path),
            obs_height=img_record.flight_altitude,
            features_json=json.dumps(features, ensure_ascii=False),
            created_at=datetime.now().isoformat()
        )
        db.add(observation)
        added += 1
        logging.info(f"Создано наблюдение {obs_id} для аннотации {ann.annotation_id}")

    db.commit()
    db.close()
    return added