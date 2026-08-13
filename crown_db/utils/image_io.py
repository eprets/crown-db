import cv2
import numpy as np

def read_image_unicode(path: str):
    """
    Читает изображение по пути, поддерживая Unicode (кириллицу).
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise ValueError(f"Не удалось прочитать изображение: {path}") from e