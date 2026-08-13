import re
import uuid
import logging
from pathlib import Path
from datetime import datetime

from ..database import SessionLocal
from ..models import Image as ImageModel
from .image_utils import get_exif_data, get_gps_coordinates

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

def iter_images(raw_dir: Path):
    """Обходит папку и возвращает пути ко всем изображениям"""
    if not raw_dir.exists():
        return
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p

def import_images(raw_dir: Path) -> int:
    """Импортирует изображения из папки в БД, возвращает количество добавленных"""
    db = SessionLocal()
    added = 0

    for img_path in iter_images(raw_dir):
        # Проверяем, не импортировано ли уже (по пути)
        existing = db.query(ImageModel).filter(ImageModel.path == str(img_path)).first()
        if existing:
            logging.info(f"Пропущено (уже есть): {img_path}")
            continue

        # Извлекаем высоту из имени файла (например "Береза 8м.JPG" → 8)
        height_from_name = None
        match = re.search(r'(\d+)\s*м', img_path.stem, re.IGNORECASE)
        if match:
            height_from_name = float(match.group(1))

        # Читаем EXIF
        exif = get_exif_data(img_path)
        lat, lon = get_gps_coordinates(exif)

        # Дата/время (из EXIF или из времени создания файла)
        timestamp = exif.get("DateTime")
        if not timestamp:
            timestamp = datetime.fromtimestamp(img_path.stat().st_ctime).isoformat()

        camera_model = exif.get("Model", None)

        # Создаём запись
        image = ImageModel(
            image_id=str(uuid.uuid4()),
            path=str(img_path),
            original_name=img_path.name,          # ← оригинальное имя файла
            lat=lat,
            lon=lon,
            flight_altitude=height_from_name,     # ← высота из имени
            timestamp=timestamp,
            camera_model=camera_model,
            created_at=datetime.now().isoformat()
        )

        db.add(image)
        added += 1
        logging.info(f"Импортировано: {img_path} (высота={height_from_name})")

    db.commit()
    db.close()
    return added