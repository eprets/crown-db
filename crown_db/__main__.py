import sys
from pathlib import Path

from .database import engine, Base
from .services.import_service import import_images
from .services.query_service import list_images

def init_db():
    print("🗄️ Создание таблиц...")
    Base.metadata.create_all(bind=engine)
    print("✅ Таблицы созданы!")

def import_cmd():
    print("📂 Импорт изображений...")
    raw_dir = Path("data/raw_images")
    if not raw_dir.exists():
        print("❌ Папка data/raw_images не найдена! Создайте её и положите изображения.")
        return
    count = import_images(raw_dir)
    print(f"✅ Импортировано {count} изображений.")

def list_cmd():
    print("📋 Список импортированных изображений (последние 10):")
    images = list_images(10)
    if not images:
        print("   Нет изображений в БД.")
        return
    for img in images:
        print(f"   {img.image_id} | {img.path} | высота: {img.flight_altitude} | {img.timestamp}")

def start_server():
    print("🚀 Запуск сервера...")
    print("🌐 Откройте в браузере: http://localhost:8000/static/annotator.html")
    import uvicorn
    uvicorn.run("crown_db.api:app", host="0.0.0.0", port=8000, reload=True)

def main():
    if len(sys.argv) < 2:
        print("🐍 Crown DB v2.0")
        print("Доступные команды:")
        print("  init-db        - создать таблицы")
        print("  import         - импортировать изображения")
        print("  list-images    - показать список изображений")
        print("  start          - запустить сервер")
        return

    command = sys.argv[1]
    if command == "init-db":
        init_db()
    elif command == "import":
        import_cmd()
    elif command == "list-images":
        list_cmd()
    elif command == "start":
        start_server()
    else:
        print(f"❌ Неизвестная команда: {command}")

if __name__ == "__main__":
    main()