import torch
import cv2
import numpy as np
from pathlib import Path

from .model import GeneratorUNet
from ...database import SessionLocal
from ...models import Level


def apply_pix2pix(tree_id: str, checkpoint_path: str = None):
    """
    Применяет обученную модель Pix2Pix для синтеза всех пропущенных уровней дерева.
    Если checkpoint_path не указан, ищет 'generator_final.pth' в папке all_trees.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")

    # Загружаем модель (обученную на всех деревьях)
    model = GeneratorUNet().to(device)
    if checkpoint_path is None:
        checkpoint_path = "data/models/pix2pix/all_trees/generator_final.pth"
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Модель не найдена: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Получаем все REAL уровни дерева
    db = SessionLocal()
    levels = db.query(Level).filter(
        Level.tree_id == tree_id,
        Level.data_type == "REAL",
        Level.roi_norm_path.isnot(None)
    ).order_by(Level.h_level).all()
    db.close()

    if not levels:
        print(f"Нет реальных уровней для дерева {tree_id}")
        return

    synth_dir = Path("data/roi_synth_pix2pix")
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Для каждой пары соседних реальных уровней генерируем промежуточные SYNTH-уровни
    for i in range(len(levels) - 1):
        h_left = levels[i].h_level
        h_right = levels[i+1].h_level
        # Генерируем все промежуточные уровни с шагом 5
        for h_mid in range(h_left + 5, h_right, 5):
            # Используем левый уровень как вход
            input_img = cv2.imread(levels[i].roi_norm_path, cv2.IMREAD_COLOR)
            if input_img is None:
                continue
            # Нормализуем
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = (input_img.astype(np.float32) / 127.5) - 1.0
            input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            with torch.no_grad():
                fake = model(input_tensor)
            fake = fake.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            fake = (fake + 1.0) * 127.5
            fake = np.clip(fake, 0, 255).astype(np.uint8)
            fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)

            synth_path = synth_dir / f"{tree_id}_{h_mid}m_synth_pix2pix.png"
            cv2.imwrite(str(synth_path), fake)
            print(f"Синтезирован уровень {h_mid}м с помощью Pix2Pix, сохранён в {synth_path}")

    print("Синтез Pix2Pix завершён.")