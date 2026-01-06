"""Точка входа в приложение системы биометрической идентификации."""
import sys
from utils.logger import setup_logger

logger = setup_logger()


def main():
    """Главная функция приложения."""
    logger.info("Запуск системы биометрической идентификации...")
    logger.info("Мини-релиз 1: Инфраструктура и БД готовы!")
    logger.info("Для проверки БД запустите: python scripts/test_db.py")
    logger.info("Для создания миграций: alembic revision --autogenerate -m 'Initial migration'")
    logger.info("Для применения миграций: alembic upgrade head")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)

