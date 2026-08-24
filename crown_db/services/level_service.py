import uuid
import logging
from pathlib import Path
from datetime import datetime

from ..database import SessionLocal
from ..models import Observation, Level, Tree


def build_levels(height_grid: list = None) -> int:
    if height_grid is None:
        height_grid = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 150]

    db = SessionLocal()
    added = 0

    tree_ids = db.query(Observation.tree_id).distinct().all()
    tree_ids = [t[0] for t in tree_ids]
    print(f"Найдено деревьев: {tree_ids}")

    for tree_id in tree_ids:
        observations = db.query(Observation).filter(Observation.tree_id == tree_id).all()
        print(f"Для дерева {tree_id} найдено наблюдений: {len(observations)}")
        print("Высоты наблюдений:", [obs.obs_height for obs in observations])

        for h_level in height_grid:
            best_obs = None
            best_diff = float('inf')
            for obs in observations:
                if obs.obs_height is None:
                    continue
                diff = abs(obs.obs_height - h_level)
                if diff < best_diff:
                    best_diff = diff
                    best_obs = obs

            print(f"Уровень {h_level}м: лучшая разница = {best_diff}, наблюдение = {best_obs.obs_id if best_obs else None}")

            if best_obs is not None and best_diff <= 5.0:
                existing = db.query(Level).filter(
                    Level.tree_id == tree_id,
                    Level.h_level == h_level
                ).first()
                if existing:
                    print(f"Уровень {h_level}м уже существует, пропускаем.")
                    continue

                level = Level(
                    level_id=str(uuid.uuid4()),
                    tree_id=tree_id,
                    h_level=h_level,
                    source_obs_id=best_obs.obs_id,
                    data_type="REAL",
                    mapping_error=best_diff,
                    roi_norm_path=best_obs.roi_norm_path,
                    created_at=datetime.now().isoformat()
                )
                db.add(level)
                added += 1
                print(f"Добавлен уровень {h_level}м")
            else:
                print(f"Уровень {h_level}м не добавлен (причина: best_obs={best_obs is not None}, best_diff={best_diff})")

    db.commit()
    db.close()
    return added