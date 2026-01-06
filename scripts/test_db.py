"""Script for testing database connection."""
import asyncio
from database.connection import AsyncSessionLocal, close_db
from database.repositories import UserRepository, BiometricTemplateRepository
from utils.logger import setup_logger

logger = setup_logger()


async def test_connection():
    """Test connection and basic operations."""
    logger.info("Testing database connection...")
    
    async with AsyncSessionLocal() as session:
        try:
            # Test 1: Get all users
            users = await UserRepository.get_all(session)
            logger.info(f"Found users: {len(users)}")
            
            # Test 2: Create test user
            test_user = await UserRepository.create(
                session,
                name="Test",
                surname="User",
                access_level=0
            )
            logger.info(f"Created test user: {test_user}")
            
            # Test 3: Get user by ID
            found_user = await UserRepository.get_by_id(session, test_user.id)
            logger.info(f"Found user: {found_user}")
            
            # Test 4: Create test template
            test_descriptor = [0.1] * 128  # Test descriptor
            template = await BiometricTemplateRepository.create(
                session,
                user_id=test_user.id,
                descriptor=test_descriptor
            )
            logger.info(f"Created test template: {template}")
            
            # Test 5: Get user templates
            templates = await BiometricTemplateRepository.get_by_user_id(
                session,
                test_user.id
            )
            logger.info(f"Found templates for user: {len(templates)}")
            
            # Commit changes
            await session.commit()
            logger.info("All tests passed successfully!")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error during testing: {e}")
            raise
        finally:
            await close_db()


if __name__ == "__main__":
    asyncio.run(test_connection())
