import uuid
import json
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import cv2
import numpy as np

from .database import SessionLocal, get_db
from .models import Image, Tree, Annotation, Mask
from .services.import_service import iter_images
from .utils.image_io import read_image_unicode

app = FastAPI()

# Подключаем статику (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="crown_db/static"), name="static")


# ===================== ИЗОБРАЖЕНИЯ =====================

@app.get("/api/images")
def get_images(db: Session = Depends(get_db)):
    """Возвращает список изображений без аннотаций"""
    images = db.query(Image).filter(~Image.annotations.any()).all()
    result = []
    for img in images:
        result.append({
            "image_id": img.image_id,
            "original_name": img.original_name or Path(img.path).name,
            "path": img.path,
            "timestamp": img.timestamp,
            "flight_altitude": img.flight_altitude
        })
    return result


@app.get("/api/image/{image_id}")
def get_image(image_id: str, db: Session = Depends(get_db)):
    img = db.query(Image).filter(Image.image_id == image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    return FileResponse(img.path)


# ===================== АННОТАЦИИ =====================

@app.post("/api/annotate")
def save_annotation(
    image_id: str = Form(...),
    tree_id: str = Form(...),
    tree_type: str = Form(...),
    x0: float = Form(...),
    y0: float = Form(...),
    a: float = Form(...),
    b: float = Form(...),
    theta: float = Form(...),
    db: Session = Depends(get_db)
):
    """Сохраняет аннотацию и генерирует маску"""

    # Проверяем/создаём дерево
    tree = db.query(Tree).filter(Tree.tree_id == tree_id).first()
    if not tree:
        tree = Tree(
            tree_id=tree_id,
            tree_type=tree_type,
            created_at=datetime.now().isoformat()
        )
        db.add(tree)
        db.commit()

    # Проверяем, нет ли уже аннотации на этом изображении для этого дерева
    existing = db.query(Annotation).filter(
        Annotation.image_id == image_id,
        Annotation.tree_id == tree_id
    ).first()

    if existing:
        # Обновляем существующую
        existing.x0 = x0
        existing.y0 = y0
        existing.a = a
        existing.b = b
        existing.theta = theta
        existing.tree_type = tree_type
        existing.created_at = datetime.now().isoformat()
        annotation_id = existing.annotation_id
        db.commit()
    else:
        # Создаём новую
        annotation_id = str(uuid.uuid4())
        annotation = Annotation(
            annotation_id=annotation_id,
            image_id=image_id,
            tree_id=tree_id,
            tree_type=tree_type,
            x0=x0, y0=y0, a=a, b=b, theta=theta,
            created_at=datetime.now().isoformat()
        )
        db.add(annotation)
        db.commit()

    # Генерируем маску
    img_record = db.query(Image).filter(Image.image_id == image_id).first()
    if not img_record:
        raise HTTPException(404, "Image not found")

    try:
        img_cv = read_image_unicode(img_record.path)
    except Exception as e:
        raise HTTPException(500, f"Не удалось прочитать файл: {e}")

    if img_cv is None:
        raise HTTPException(500, "Не удалось декодировать изображение")

    mask_dir = Path("data/masks")
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{annotation_id}.png"

    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(x0), int(y0))
    axes = (int(a), int(b))
    angle = theta * 180.0 / np.pi
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    cv2.imwrite(str(mask_path), mask)

    # Сохраняем запись в Mask
    mask_record = db.query(Mask).filter(Mask.annotation_id == annotation_id).first()
    if mask_record:
        mask_record.mask_path = str(mask_path)
        mask_record.created_at = datetime.now().isoformat()
    else:
        mask_record = Mask(
            mask_id=str(uuid.uuid4()),
            annotation_id=annotation_id,
            mask_path=str(mask_path),
            created_at=datetime.now().isoformat()
        )
        db.add(mask_record)
    db.commit()

    return {
        "status": "ok",
        "annotation_id": annotation_id,
        "tree_id": tree_id,
        "image_original_name": img_record.original_name or Path(img_record.path).name
    }


@app.get("/api/annotations/all")
def get_all_annotations(db: Session = Depends(get_db)):
    """Возвращает список всех аннотаций с информацией об изображениях"""
    annotations = db.query(Annotation).order_by(Annotation.created_at.desc()).all()
    result = []
    for ann in annotations:
        img = db.query(Image).filter(Image.image_id == ann.image_id).first()
        result.append({
            "annotation_id": ann.annotation_id,
            "image_id": ann.image_id,
            "image_original_name": img.original_name if img else None,
            "tree_id": ann.tree_id,
            "tree_type": ann.tree_type,
            "x0": ann.x0,
            "y0": ann.y0,
            "a": ann.a,
            "b": ann.b,
            "theta": ann.theta,
            "created_at": ann.created_at,
            "has_mask": db.query(Mask).filter(Mask.annotation_id == ann.annotation_id).count() > 0
        })
    return result


@app.get("/api/annotations/{annotation_id}")
def get_annotation(annotation_id: str, db: Session = Depends(get_db)):
    """Возвращает данные одной аннотации с путём к изображению"""
    ann = db.query(Annotation).filter(Annotation.annotation_id == annotation_id).first()
    if not ann:
        raise HTTPException(404, "Annotation not found")
    img = db.query(Image).filter(Image.image_id == ann.image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    return {
        "annotation_id": ann.annotation_id,
        "image_id": ann.image_id,
        "image_path": img.path,
        "tree_id": ann.tree_id,
        "tree_type": ann.tree_type,
        "x0": ann.x0,
        "y0": ann.y0,
        "a": ann.a,
        "b": ann.b,
        "theta": ann.theta
    }


@app.put("/api/annotations/{annotation_id}")
def update_annotation(
    annotation_id: str,
    tree_id: str = Form(...),
    tree_type: str = Form(...),
    x0: float = Form(...),
    y0: float = Form(...),
    a: float = Form(...),
    b: float = Form(...),
    theta: float = Form(...),
    db: Session = Depends(get_db)
):
    """Обновляет аннотацию и перегенерирует маску"""
    ann = db.query(Annotation).filter(Annotation.annotation_id == annotation_id).first()
    if not ann:
        raise HTTPException(404, "Annotation not found")

    # Обновляем поля
    ann.tree_id = tree_id
    ann.tree_type = tree_type
    ann.x0 = x0
    ann.y0 = y0
    ann.a = a
    ann.b = b
    ann.theta = theta
    ann.created_at = datetime.now().isoformat()
    db.commit()

    # Перегенерируем маску
    img_record = db.query(Image).filter(Image.image_id == ann.image_id).first()
    if img_record:
        try:
            img_cv = read_image_unicode(img_record.path)
        except Exception as e:
            raise HTTPException(500, f"Не удалось прочитать файл: {e}")

        if img_cv is not None:
            mask_dir = Path("data/masks")
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{annotation_id}.png"
            h, w = img_cv.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (int(x0), int(y0))
            axes = (int(a), int(b))
            angle = theta * 180.0 / np.pi
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
            cv2.imwrite(str(mask_path), mask)

            mask_record = db.query(Mask).filter(Mask.annotation_id == annotation_id).first()
            if mask_record:
                mask_record.mask_path = str(mask_path)
                mask_record.created_at = datetime.now().isoformat()
            else:
                mask_record = Mask(
                    mask_id=str(uuid.uuid4()),
                    annotation_id=annotation_id,
                    mask_path=str(mask_path),
                    created_at=datetime.now().isoformat()
                )
                db.add(mask_record)
            db.commit()

    return {"status": "ok", "annotation_id": annotation_id}


@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: str, db: Session = Depends(get_db)):
    """Удаляет аннотацию и связанную маску"""
    ann = db.query(Annotation).filter(Annotation.annotation_id == annotation_id).first()
    if not ann:
        raise HTTPException(404, "Annotation not found")

    # Удаляем связанную маску
    mask = db.query(Mask).filter(Mask.annotation_id == annotation_id).first()
    if mask:
        if mask.mask_path and Path(mask.mask_path).exists():
            Path(mask.mask_path).unlink()
        db.delete(mask)

    db.delete(ann)
    db.commit()
    return {"status": "ok"}