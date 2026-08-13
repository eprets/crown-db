from ..database import SessionLocal
from ..models import Image

def list_images(limit=10):
    db = SessionLocal()
    images = db.query(Image).order_by(Image.created_at.desc()).limit(limit).all()
    db.close()
    return images