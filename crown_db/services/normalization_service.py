import logging
from pathlib import Path

import cv2
import numpy as np

from ..database import SessionLocal
from ..models import Observation
from ..utils.image_io import read_image_unicode


def normalize_observations(target_size: int = 256) -> int:
    """
    Нормализует все наблюдения (ROI) к единому размеру target_size x target_size.
    Сохраняет нормализованные ROI в data/roi_norm/ и обновляет поле roi_norm_path.
    Возвращает количество обработанных наблюдений.
    """
    db = SessionLocal()
    processed = 0

    observations = db.query(Observation).filter(
        Observation.roi_norm_path.is_(None)
    ).all()

    if not observations:
        logging.info("Нет наблюдений для нормализации.")
        return 0

    roi_norm_dir = Path("data/roi_norm")
    roi_norm_dir.mkdir(parents=True, exist_ok=True)

    for obs in observations:
        roi_raw_path = Path(obs.roi_raw_path)
        if not roi_raw_path.exists():
            logging.warning(f"ROI файл не найден: {roi_raw_path}")
            continue

        img = read_image_unicode(str(roi_raw_path))
        if img is None:
            logging.warning(f"Не удалось прочитать ROI: {roi_raw_path}")
            continue

        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        norm_filename = f"{obs.obs_id}_norm.png"
        norm_path = roi_norm_dir / norm_filename
        cv2.imwrite(str(norm_path), canvas)

        obs.roi_norm_path = str(norm_path)
        db.add(obs)
        processed += 1
        logging.info(f"Нормализовано наблюдение {obs.obs_id}")

    db.commit()
    db.close()
    return processed