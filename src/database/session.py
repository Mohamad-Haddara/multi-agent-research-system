"""
database.py/session.py -> engine, session, get_db

Create the SQLAlchemy database engine

3 things to watch out for production:
1) Don't create tables at start up at all; use Alembic migrations
2) create_all only create tables but never modify existing ones, so a new column would never reach the database
3) dont use hardcoded database url
4) If the database is in the cloud, also add ?sslmode=require to the URL.

"""

from curses import echo
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from entities import Base

from typing import Annotated
from fastapi import Depends




# For Postgres databases, the connection string is defined using the following template: <driver>://<username>:<password>@<origin>/<database>
database_url = (
    "postgresql+psycopg://fastapi:mysecretpassword@localhost:5432/backend_db"
)


# Create an async database engine using the database connection string. Turn on debug logging with echo=True
engine = create_async_engine(database_url, echo=True)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all) # Drop any existing tables and then create all database tables using the defined SQLAlchemy models


# Create a database session FastAPI dependency

# Create an async database session factory bound to the database engine to asynchronously connect to our Postgres instance.
# Disable automatic commiting of transactions with (autocommit=false)
# Disable automatic flushing of changes to the database with (autoflush=false)
# Disabling both will give us more control, helps prevent unintended data updates, and allows us to implement more robust transaction management

async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, autocommit=False, autoflush=False
)


# Define a dependency function to reuse and inject across our FastAPI app into route controller
# Since the function uses the yield with the async with -> async context manager
# FastAPI will internally decorate the get_db_session as context manager when it is used as dependency
async def get_db_session():
    try:
        # Use the database session factory to create async session. The context manager helps to manage database session lifecycle such as opening, interacting, and closing the database conenections in each session.
        async with async_session() as session:
            # yield the database session to the caller of the get_db_session
            yield session

    except:
        # If there are any exceptions, roll back the transaction and reraise the exception
        await session.rollback()
        raise

    finally:
        # In any case, close the database session dependency that can be reused across different controllers
        await session.close()