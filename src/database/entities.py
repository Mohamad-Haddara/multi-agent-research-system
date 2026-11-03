"""
Entities

Our table classes (SQLAlchemy models)

In entities is where the database models liveL: the classes inherit from Base and map to tables

- Entities (SQLAlchemy) define how data is stored in the database.
- Schemas (Pydantic) define what the API receives and returns.

The model alone doesnt create the table in the database. That happens when you run Base.metadata.create_all(engine) or an Alembic migration.

Mapped[str] -> tells SQLAlchemy the Python type. It uses it to pick the SQL type (str becomes VARCHAR)
mapped_column() -> holds the database settings for that colum

""" 

from datetime import UTC, datetime
from mailbox import Message
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# Declarative base is the parent class our ORM model inherit from. Declarative means I describe the table right on the Python class (columns, types, keys) and SQLAlchemy builds the table metadata and maps rows to objects
# Declare a declarative base class for creating SQLAlchemy models for its ORM engine. Turns a normal Python class into an ORM model
class Base(DeclarativeBase):
    """
    Base itself is not a model. It has no table; it's only the foundation.
    
    Each subclass becomes a model. Any class that inherit from Base and has __tablname__ and a PK is mapped to a table.

    Without the base, Conversation would be a plain class that SQLAlchemy knows nothing about.
    """
    pass

# Create the conversation model specifying the table columns, primary key, and indexes
class Conversation(Base):
    __tablename__="conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column()
    model_type: Mapped[str] = mapped_column(index = True) # Index the `model_type` in case I want faster filtering of conversations by model type
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC) # Specify defaults and update operations for datetime columns
    )

    messages: Mapped[list["Message"]] = relationship("Message", back_populates="conversation", cascade="all, delete-orphan") # indicate that all orphan messages must be deleted if a conversation is deleted through a CASCADE DELETE operation.
    # It means when the conversation is deleted, its messages are deleted with it. A message whose conversation no longer exists is an "orphan"
    # cascade = "all, delete-orphan" ~ makes SQLAlchemy delete the messages when I call session.delete(conversation). The delete-orphan also deletes a messages If I want to remove it from conversation.messages


# Create a Message model specifying table columns, PK, secondary indexes, table relationships, and FK
class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True   # Indicate all orphan messages must be deleted if a conversation is deleted through CASCADE DELETE operation
    )

    prompt_content: Mapped[str] = mapped_column()
    response_content: Mapped[str] = mapped_column()
    prompt_tokens: Mapped[int | None] = mapped_column() # Specify Mapped[int|None] to declare optional typing so the column will allow NULL values (nullable=True)
    response_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    is_success: Mapped[bool | None] = mapped_column()
    status_code: Mapped[int | None] = mapped_column() # The messages tables will contain both the LLM prompts and response, usage token, and cost along side the status code
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC)) # I use lambda to run on each insert, instead of every row got the same time
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC), onupdate=datetime.now(UTC))

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

