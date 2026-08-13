from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Базовый класс для всех моделей
Base = declarative_base()

# Путь к БД (из корня проекта)
DB_PATH = Path("data/db/crowns.sqlite3")

# Создаём папку для БД, если её нет
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Подключаемся к SQLite
engine = create_engine(f"sqlite:///{DB_PATH}", echo=True)

# Фабрика сессий
SessionLocal = sessionmaker(bind=engine)

# Функция для получения сессии (используем в API)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()