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
from sqlalchemy.ext.asyncio import create_async_engine
from entities import Base

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