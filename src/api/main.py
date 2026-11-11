"""
This file contains the main application entry point. ASGI entrypoint.

FastAPI; lifespan builds
"""

import logging
from uuid import uuid4


from fastapi import (
    FastAPI,
    Request,
    status,
    Depends,
    HTTPException,
)
from fastapi.routing import JSONResponse

from src.core.config import get_settings
from datetime import datetime

from openai import OpenAI
from src.core.config import get_settings

from src.schemas.chat import ChatRequest, ChatResponse


from contextlib import asynccontextmanager, contextmanager
from src.database.session import engine, init_db



# CRUD endpoints
from typing import Annotated
from src.database.session import DBSessionDep
from src.database.entities import Conversation
from src.schemas.database import ConversationCreate, ConversationOut, ConversationUpdate
from sqlalchemy import select



settings = get_settings()



logger = logging.getLogger(__name__)

client = OpenAI(api_key = settings.openai_api_key)


@asynccontextmanager
async def lifespan(_:FastAPI):
    """
    Handle application startup and shutdown events.
    """
    # Initialize database connection
    try:
        await  init_db()
    except Exception as e:
        logger.exception("database_initialization_failed")


    # Other startup operations within the lifespan will be added later on ...

    yield

    # Cleanup and shutdown
    await engine.dispose() # Dispose of the database engine during the server shutdown process. Any code after yield is executed when the server shutdown is requested




app = FastAPI(
   # title= settings.project_name,
    version= settings.version,
    description= settings.description,
    lifespan= lifespan

)


@app.get("/")
async def root(request: Request) -> dict[str, str]:
    """Root endpoint returning basic API information"""
    logger.info("root endpoint called")
    return {
        "name": settings.project_name,
        "version": settings.version,
        "status": "healthy",
        "environment": settings.app_env,
        "docs_url": "/docs",
    }




@app.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """
    Health check endpoint with environment-specific information.

    Here I should check if the dependencies (database) are reachable

    Returns:
        JSONResponse: Health status payload, with HTTP 503 when the database is 
        unreachable so load balancer can drop the instance
    """

    logger.info("health_check endpoint called")
    
    response = {
        "status": "healthy",
        "version": settings.version,
        "environment": settings.app_env,
        "timestamp": datetime.now().isoformat(),
    }

    return JSONResponse(content=response)




# API Routes
@app.post("/chat")
async def chat(body: ChatRequest, request: Request ) -> ChatResponse:
    """
    Chat endpoint using OpenAI API
    """
    logger.info("Chat endpoint called")
    
    messages = [
        {
            "role": "system",
            "content" : " You are a helpful assistant that can answer questions and help with tasks."
        },
        {
            "role":"user",
            "content": body.messages
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        logprobs=None,
        logit_bias=None,
        user=None,
        response_format=None,
    )

    # Pydantic validates the value is UUID and converts it to a string in the JSON output. Recommendation use UUID in both request and response model
    return ChatResponse(
        request_id= uuid4(),
        thread_id=body.thread_id or uuid4(), 
        content = response.choices[0].message.content
    )




"""
Implementing resource-based CRUD endpoints fot the conversation tables
"""


# Define a dependency to check if the conversation record exists. Raise a 404 HTTPException if a record is not found; otherwise return the retrieved record
# This dependency can be reused across several CRUD endpoints through dependency injection
async def get_conversation(
    conversation_id: int,
    session: DBSessionDep
) -> Conversation:
    # Begin the async session within an async context manager during each request
    async with session.begin():
        result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )

        conversation = result.scalars().first()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        return conversation


GetConversationDep = Annotated[Conversation, Depends(get_conversation)]



@app.get("/conversations")
async def list_conversations_controller(
    session: DBSessionDep,
    skip: int = 0,
    take: int = 100
) -> list[ConversationOut]:

    async with session.begin():
        result = await session.execute(
            select(Conversation).offset(skip).limit(take)
        )
        # When listing records, it is more efficient to retrieve only a subset of records. By default, SQLAlchemy ORM returns a subset of most recent records in the database
        # But I can use the .offset(skip) and .limit(take) chained methods to retrieve any subset records
    return [
        ConversationOut.model_validate(conversation)
        for conversation in result.scalars().all() ]



@app.get("/conversations/{id}")
async def list_conversations_controller(
    conversation: GetConversationDep,
) -> ConversationOut:
    return ConversationOut.model_validate(conversation)  #Create a Pydantic model from SQLAlchemy model using model_validate. Raises a ValidationError if the SQLAlchemy object passed can’t be created or doesn’t pass Pydantic’s data validation checks.




@app.post("/conversations", status_code=status.HTTP_201_CREATED)
async def create_conversation_controller(
    conversation: ConversationCreate,
    session: DBSessionDep,
) -> ConversationOut:
    new_conversation = Conversation(**conversation.model_dump())

    async with session.begin():
        session.add(new_conversation)
        await session.commit() # For operations (Create, Update , Delete) - commit the transaction then send the refreshed record to the client, except for the successful delete operation that should return None
        await session.refresh(new_conversation)

    return ConversationOut.model_validate(new_conversation)


@app.put("/conversation/{id}")
async def update_conversation_controller(
    updated_conversation: ConversationUpdate,
    conversation: GetConversationDep,
    session: DBSessionDep,
) -> ConversationOut:
    for key, value in updated_conversation.model_dump().items():
        setattr(conversation, key, value)
    
    async with session.begin():
        await session.commit()
        await session.refresh(conversation)

    return ConversationOut.model_validate(conversation)



@app.delete("/conversation/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_controller(
    conversation: GetConversationDep,
    session: DBSessionDep
) -> None:
    async with session.begin():
        await session.delete(conversation)
        await session.commit()

if __name__ == "__main__":
    import uvicorn
    import asyncio
    # Use reload in development only. In production, remove it.
    # Open the auto-generated API docs at http://localhost:8000/docs.
    asyncio.run(uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000), loop_factory=asyncio.SelectorEventLoop)