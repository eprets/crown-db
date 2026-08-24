import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ...database import SessionLocal
from ...models import Level


class Pix2PixDataset(Dataset):
    """
    Датасет для Pix2Pix.
    Формирует пары (input, target) из реальных уровней всех доступных деревьев.
    """
    def __init__(self, tree_ids=None, min_height_diff: int = 5, max_height_diff: int = 50, augment: bool = True):
        self.min_diff = min_height_diff
        self.max_diff = max_height_diff
        self.augment = augment

        db = SessionLocal()

        # Если tree_ids не указан, получаем все tree_id с REAL уровнями
        if tree_ids is None:
            result = db.query(Level.tree_id).filter(Level.data_type == "REAL").distinct().all()
            tree_ids = [r[0] for r in result]

        if not tree_ids:
            db.close()
            raise ValueError("Нет деревьев с REAL уровнями для обучения")

        self.pairs = []

        for tree_id in tree_ids:
            levels = db.query(Level).filter(
                Level.tree_id == tree_id,
                Level.data_type == "REAL",
                Level.roi_norm_path.isnot(None)
            ).order_by(Level.h_level).all()

            if len(levels) < 2:
                continue

            # Формируем все возможные пары (вход, выход) для данного дерева
            for i in range(len(levels)):
                for j in range(len(levels)):
                    if i == j:
                        continue
                    h1 = levels[i].h_level
                    h2 = levels[j].h_level
                    diff = abs(h2 - h1)
                    if self.min_diff <= diff <= self.max_diff:
                        self.pairs.append((levels[i], levels[j]))

        db.close()

        if not self.pairs:
            raise ValueError(f"Не найдено пар для обучения. Проверьте, что есть REAL уровни.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_level, target_level = self.pairs[idx]

        input_path = input_level.roi_norm_path
        target_path = target_level.roi_norm_path

        input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        target_img = cv2.imread(target_path, cv2.IMREAD_COLOR)

        if input_img is None or target_img is None:
            # fallback
            input_img = cv2.imread(str(Path(input_path)), cv2.IMREAD_COLOR)
            target_img = cv2.imread(str(Path(target_path)), cv2.IMREAD_COLOR)

        # BGR -> RGB, нормализация [-1, 1]
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        input_img = (input_img.astype(np.float32) / 127.5) - 1.0
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0

        if self.augment and random.random() > 0.5:
            # Горизонтальное отражение
            input_img = np.fliplr(input_img).copy()
            target_img = np.fliplr(target_img).copy()

        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float()
        target_tensor = torch.from_numpy(target_img.transpose(2, 0, 1)).float()

        return input_tensor, target_tensor