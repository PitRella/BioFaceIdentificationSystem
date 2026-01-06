"""Script for database initialization."""
import asyncio
from database.connection import init_db, close_db
from utils.logger import setup_logger

logger = setup_logger()


async def main():
    """Initialize database."""
    logger.info("Initializing database...")
    try:
        await init_db()
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
