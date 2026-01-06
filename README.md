# Система биометрической идентификации по распознаванию лиц

Система для автоматической идентификации пользователей по биометрическим данным лица с использованием алгоритмов компьютерного зрения и машинного обучения.

## Текущий статус: Мини-релиз 1 ✅

**Мини-релиз 1: Инфраструктура и БД** - завершен
- ✅ Структура проекта
- ✅ Конфигурация (config.py)
- ✅ PostgreSQL в Docker
- ✅ SQLAlchemy 2.0 (async) модели
- ✅ Alembic миграции
- ✅ Репозитории для CRUD операций
- ✅ Логирование

## Требования

- Python 3.13 (минимум 3.12)
- Docker и docker-compose
- PostgreSQL 15+ (в Docker)

## Установка

### 1. Клонирование и установка зависимостей

```bash
# Установка зависимостей
pip install -r requirements.txt
```

**Важно:** Для установки `dlib` может потребоваться компилятор C++ и CMake. На Linux:
```bash
sudo apt-get install build-essential cmake
```

### 2. Запуск PostgreSQL в Docker

```bash
# Запуск контейнера с PostgreSQL
docker-compose up -d

# Проверка статуса
docker-compose ps
```

### 3. Настройка переменных окружения

Создайте файл `.env` (можно скопировать из `.env.example`):
```bash
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=face_recognition_db
DATABASE_USER=user
DATABASE_PASSWORD=password
```

### 4. Инициализация базы данных

```bash
# Создание миграций
alembic revision --autogenerate -m "Initial migration"

# Применение миграций
alembic upgrade head

# Или используйте скрипт инициализации (создает таблицы напрямую)
python scripts/init_db.py
```

### 5. Тестирование подключения к БД

```bash
python scripts/test_db.py
```

## Структура проекта

```
BioFaceIdentificationSystem/
├── main.py                 # Точка входа
├── config.py              # Конфигурация
├── requirements.txt        # Зависимости
├── docker-compose.yml     # Docker конфигурация
├── alembic.ini            # Конфигурация Alembic
│
├── core/                   # Модули ядра (следующий релиз)
├── database/               # Работа с БД
│   ├── models.py          # SQLAlchemy модели
│   ├── connection.py      # Подключение к БД
│   └── repositories.py    # CRUD операции
├── ui/                     # UI (будущие релизы)
├── utils/                  # Утилиты
│   └── logger.py          # Логирование
├── scripts/                # Вспомогательные скрипты
│   ├── init_db.py         # Инициализация БД
│   └── test_db.py         # Тест подключения
└── alembic/               # Миграции БД
```

## Следующие мини-релизы

- **Мини-релиз 2:** Модули ядра (VideoCapture, FaceDetector, FaceEncoder, QualityValidator, FaceRecognizer)
- **Мини-релиз 3:** Интеграция модулей ядра с БД
- **Мини-релиз 4:** Регистрация пользователей
- **Мини-релиз 5:** Идентификация (консольная версия)
- **Мини-релиз 6:** UI - главное окно
- **Мини-релиз 7:** Админ-панель

## Полезные команды

```bash
# Запуск приложения
python main.py

# Работа с миграциями
alembic revision --autogenerate -m "Описание изменений"
alembic upgrade head
alembic downgrade -1

# Остановка PostgreSQL
docker-compose down

# Просмотр логов PostgreSQL
docker-compose logs postgres
```

## Архитектура

Подробное описание архитектуры см. в [ARCHITECTURE.md](ARCHITECTURE.md)
