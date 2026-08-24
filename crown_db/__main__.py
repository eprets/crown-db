import sys
from pathlib import Path

from .database import engine, Base
from .services.import_service import import_images
from .services.query_service import list_images
from .services.observation_service import build_observations
from .services.normalization_service import normalize_observations
from .services.level_service import build_levels
from .services.synthesis_service import synthesize_missing
from .database import SessionLocal
from .models import Level
from .ml.pix2pix.train import train_pix2pix
from .ml.pix2pix.infer import apply_pix2pix

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


def build_obs_cmd():
    print("📦 Сборка наблюдений (ROI)...")
    count = build_observations(padding_px=20)
    print(f"✅ Создано {count} наблюдений.")


def normalize_cmd():
    print("📏 Нормализация масштаба ROI...")
    count = normalize_observations(target_size=256)
    print(f"✅ Нормализовано {count} наблюдений.")


def build_levels_cmd():
    print("📊 Построение высотной сетки...")
    count = build_levels()
    print(f"✅ Создано {count} уровней.")


def get_all_tree_ids():
    """Возвращает список всех tree_id, у которых есть REAL уровни."""
    db = SessionLocal()
    tree_ids = db.query(Level.tree_id).distinct().filter(Level.data_type == "REAL").all()
    db.close()
    return [t[0] for t in tree_ids]


def synthesize_cmd(tree_id: str = None):
    """
    Синтез пропусков для указанного дерева.
    Если tree_id не указан, выводит список доступных деревьев.
    """
    if tree_id is None:
        tree_ids = get_all_tree_ids()
        if not tree_ids:
            print("❌ Нет деревьев с REAL уровнями. Сначала выполните build-levels.")
            return
        print("Доступные деревья для синтеза:")
        for tid in tree_ids:
            print(f"  {tid}")
        print("\nУкажите tree_id: python -m crown_db synthesize <tree_id>")
        return

    print(f"🧪 Синтез пропусков для дерева {tree_id}...")
    count = synthesize_missing(tree_id, step=5)
    print(f"✅ Синтезировано {count} уровней.")


def start_server():
    print("🚀 Запуск сервера...")
    print("🌐 Откройте в браузере: http://localhost:8000/static/index.html")
    import uvicorn
    uvicorn.run("crown_db.api:app", host="0.0.0.0", port=8000, reload=True)

def train_pix2pix_cmd():
    print("Обучение Pix2Pix на всех доступных деревьях...")
    epochs = int(input("Количество эпох (по умолчанию 100): ") or 100)
    batch_size = int(input("Размер батча (по умолчанию 4): ") or 4)
    train_pix2pix(None, epochs=epochs, batch_size=batch_size, device='cuda')

def apply_pix2pix_cmd():
    tree_id = input("Введите tree_id для синтеза: ").strip()
    if not tree_id:
        print("❌ tree_id не указан.")
        return
    apply_pix2pix(tree_id)

def main():
    if len(sys.argv) < 2:
        print("🐍 Crown DB v2.0")
        print("Доступные команды:")
        print("  init-db           - создать таблицы")
        print("  import            - импортировать изображения")
        print("  list-images       - показать список изображений")
        print("  build-observations - создать ROI для всех аннотаций")
        print("  normalize         - нормализовать масштаб ROI")
        print("  build-levels      - построить высотную сетку")
        print("  synthesize [tree_id] - синтезировать пропущенные уровни")
        print("  start             - запустить сервер")
        return

    command = sys.argv[1]

    if command == "init-db":
        init_db()
    elif command == "import":
        import_cmd()
    elif command == "list-images":
        list_cmd()
    elif command == "build-observations":
        build_obs_cmd()
    elif command == "normalize":
        normalize_cmd()
    elif command == "build-levels":
        build_levels_cmd()
    elif command == "synthesize":
        # Проверяем, передан ли tree_id
        if len(sys.argv) > 2:
            tree_id = sys.argv[2]
            synthesize_cmd(tree_id)
        else:
            synthesize_cmd()  # вызов без аргумента покажет список
    elif command == "start":
        start_server()
    elif command == "train-pix2pix":
        train_pix2pix_cmd()
    elif command == "apply-pix2pix":
        apply_pix2pix_cmd()
    else:
        print(f"❌ Неизвестная команда: {command}")


if __name__ == "__main__":
    main()