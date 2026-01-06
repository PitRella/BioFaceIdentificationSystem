"""SQLAlchemy models for database."""
from datetime import datetime
from typing import List
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, JSON, ForeignKey, Float, Text


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    surname: Mapped[str] = mapped_column(String(255), nullable=False)
    registration_date: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow,
        nullable=False
    )
    photo_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    access_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow,
        nullable=False
    )
    
    # Relationships
    templates: Mapped[List["BiometricTemplate"]] = relationship(
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    access_logs: Mapped[List["AccessLog"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, name='{self.name}', surname='{self.surname}')>"


class BiometricTemplate(Base):
    """Biometric template model (face descriptor)."""
    __tablename__ = "biometric_templates"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    descriptor: Mapped[List[float]] = mapped_column(JSON, nullable=False)  # 128 numbers
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="templates")
    
    def __repr__(self) -> str:
        return f"<BiometricTemplate(id={self.id}, user_id={self.user_id})>"


class AccessLog(Base):
    """Access log model."""
    __tablename__ = "access_log"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    access_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'verification' or 'identification'
    result: Mapped[str] = mapped_column(String(20), nullable=False)  # 'success' or 'failure'
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Relationships
    user: Mapped["User | None"] = relationship(back_populates="access_logs")
    
    def __repr__(self) -> str:
        return f"<AccessLog(id={self.id}, user_id={self.user_id}, result='{self.result}')>"
