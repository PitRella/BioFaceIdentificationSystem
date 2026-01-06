"""Database repositories (async CRUD operations)."""
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from datetime import datetime

from database.models import User, BiometricTemplate, AccessLog


class UserRepository:
    """Repository for user operations."""
    
    @staticmethod
    async def create(
        session: AsyncSession,
        name: str,
        surname: str,
        photo_path: Optional[str] = None,
        access_level: int = 0
    ) -> User:
        """Create a new user."""
        user = User(
            name=name,
            surname=surname,
            photo_path=photo_path,
            access_level=access_level,
            registration_date=datetime.utcnow()
        )
        session.add(user)
        await session.flush()
        await session.refresh(user)
        return user
    
    @staticmethod
    async def get_by_id(session: AsyncSession, user_id: int) -> Optional[User]:
        """Get user by ID."""
        result = await session.execute(
            select(User)
            .options(selectinload(User.templates))
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_all(session: AsyncSession) -> List[User]:
        """Get all users."""
        result = await session.execute(
            select(User)
            .options(selectinload(User.templates))
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def update(
        session: AsyncSession,
        user_id: int,
        name: Optional[str] = None,
        surname: Optional[str] = None,
        photo_path: Optional[str] = None,
        access_level: Optional[int] = None
    ) -> Optional[User]:
        """Update user data."""
        user = await UserRepository.get_by_id(session, user_id)
        if not user:
            return None
        
        if name is not None:
            user.name = name
        if surname is not None:
            user.surname = surname
        if photo_path is not None:
            user.photo_path = photo_path
        if access_level is not None:
            user.access_level = access_level
        
        await session.flush()
        await session.refresh(user)
        return user
    
    @staticmethod
    async def delete(session: AsyncSession, user_id: int) -> bool:
        """Delete user."""
        result = await session.execute(
            delete(User).where(User.id == user_id)
        )
        await session.flush()
        return result.rowcount > 0


class BiometricTemplateRepository:
    """Repository for biometric template operations."""
    
    @staticmethod
    async def create(
        session: AsyncSession,
        user_id: int,
        descriptor: List[float]
    ) -> BiometricTemplate:
        """Create biometric template."""
        template = BiometricTemplate(
            user_id=user_id,
            descriptor=descriptor
        )
        session.add(template)
        await session.flush()
        await session.refresh(template)
        return template
    
    @staticmethod
    async def get_by_user_id(session: AsyncSession, user_id: int) -> List[BiometricTemplate]:
        """Get all templates for a user."""
        result = await session.execute(
            select(BiometricTemplate)
            .where(BiometricTemplate.user_id == user_id)
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_all(session: AsyncSession) -> List[BiometricTemplate]:
        """Get all templates with related users."""
        result = await session.execute(
            select(BiometricTemplate)
            .options(selectinload(BiometricTemplate.user))
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def delete_by_user_id(session: AsyncSession, user_id: int) -> int:
        """Delete all templates for a user."""
        result = await session.execute(
            delete(BiometricTemplate).where(BiometricTemplate.user_id == user_id)
        )
        await session.flush()
        return result.rowcount


class AccessLogRepository:
    """Repository for access log operations."""
    
    @staticmethod
    async def create(
        session: AsyncSession,
        user_id: Optional[int],
        access_type: str,
        result: str,
        confidence: Optional[float] = None
    ) -> AccessLog:
        """Create access log entry."""
        log_entry = AccessLog(
            user_id=user_id,
            access_type=access_type,
            result=result,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
        session.add(log_entry)
        await session.flush()
        await session.refresh(log_entry)
        return log_entry
    
    @staticmethod
    async def get_recent(
        session: AsyncSession,
        limit: int = 100
    ) -> List[AccessLog]:
        """Get recent log entries."""
        result = await session.execute(
            select(AccessLog)
            .options(selectinload(AccessLog.user))
            .order_by(AccessLog.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_by_user_id(
        session: AsyncSession,
        user_id: int,
        limit: int = 100
    ) -> List[AccessLog]:
        """Get log entries for a user."""
        result = await session.execute(
            select(AccessLog)
            .where(AccessLog.user_id == user_id)
            .order_by(AccessLog.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
