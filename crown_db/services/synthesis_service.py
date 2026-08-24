import uuid
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from ..database import SessionLocal
from ..models import Level, Observation


def interpolate_images(img1_path, img2_path, alpha):
    """Линейная интерполяция двух изображений с коэффициентом alpha (0..1)"""
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        return None
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    blended = cv2.addWeighted(img1, 1 - alpha, img2_resized, alpha, 0)
    return blended


def synthesize_missing(tree_id: str, step: int = 5) -> int:
    """
    Заполняет пропущенные уровни для дерева с шагом step (по умолчанию 5 м).
    Использует линейную интерполяцию между соседними REAL-уровнями.
    Возвращает количество синтезированных уровней.
    """
    db = SessionLocal()
    added = 0

    real_levels = db.query(Level).filter(
        Level.tree_id == tree_id,
        Level.data_type == "REAL"
    ).order_by(Level.h_level).all()

    if not real_levels:
        logging.warning(f"Нет реальных уровней для дерева {tree_id}")
        return 0

    existing_h = {level.h_level for level in db.query(Level).filter(Level.tree_id == tree_id).all()}

    h_min = real_levels[0].h_level
    h_max = real_levels[-1].h_level

    # Генерируем сетку с шагом step
    grid = list(range(h_min, h_max + 1, step))

    for h in grid:
        if h in existing_h:
            continue

        # Находим соседние реальные уровни
        left = None
        right = None
        for lvl in real_levels:
            if lvl.h_level < h:
                left = lvl
            elif lvl.h_level > h:
                right = lvl
                break

        if left is None or right is None:
            continue

        alpha = (h - left.h_level) / (right.h_level - left.h_level)

        left_obs = db.query(Observation).filter(Observation.obs_id == left.source_obs_id).first()
        right_obs = db.query(Observation).filter(Observation.obs_id == right.source_obs_id).first()

        if not left_obs or not right_obs:
            continue
        if not left_obs.roi_norm_path or not right_obs.roi_norm_path:
            continue

        blended = interpolate_images(
            Path(left_obs.roi_norm_path),
            Path(right_obs.roi_norm_path),
            alpha
        )
        if blended is None:
            continue

        synth_dir = Path("data/roi_synth")
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth_filename = f"{tree_id}_{h}m_synth_linear.png"
        synth_path = synth_dir / synth_filename
        cv2.imwrite(str(synth_path), blended)

        level = Level(
            level_id=str(uuid.uuid4()),
            tree_id=tree_id,
            h_level=h,
            source_obs_id=None,
            data_type="SYNTH",
            mapping_error=None,
            roi_norm_path=str(synth_path),
            synth_method="linear_interp",
            synth_src_h=f"{left.h_level},{right.h_level}",
            created_at=datetime.now().isoformat()
        )
        db.add(level)
        added += 1
        logging.info(f"Синтезирован уровень {h}м для дерева {tree_id}")

    db.commit()
    db.close()
    return added